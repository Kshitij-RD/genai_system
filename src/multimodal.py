"""Multimodal pipeline — VLM visual QA with grounding, confidence scoring, and OCR augmentation.

Uses Groq's Llama-4-Scout vision model (free tier) via OpenAI-compatible API.
Accepts image + question, returns structured answer with grounding and confidence.
"""

import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from PIL import Image

from src.utils import append_jsonl, utc_now_iso

logger = logging.getLogger("genai_system.multimodal")

# ---------------------------------------------------------------------------
# System prompt for grounded VQA with confidence scoring
# ---------------------------------------------------------------------------

MULTIMODAL_SYSTEM = (
    "You are a precise visual question-answering assistant.\n"
    "Answer the question about the image. You MUST:\n"
    "1. Identify the specific region of the image you are reasoning about.\n"
    "2. Provide a confidence score between 0.0 and 1.0 for your answer.\n"
    "3. Include a grounding field describing the location "
    "(e.g., 'upper-right quadrant', 'central foreground object', "
    "'lower-left region, spanning approximately 30% of the image width').\n\n"
    "Return ONLY a valid JSON object with these exact keys:\n"
    '- "reasoning": step-by-step visual analysis of the image (string)\n'
    '- "answer": your final answer to the question (string)\n'
    '- "confidence": a float between 0.0 and 1.0 indicating certainty (number)\n'
    '- "grounding": description of the image region you are reasoning about (string)\n'
    "Do NOT include any text outside the JSON object."
)


class MultimodalPipeline:
    """Wraps VLM calls for visual question answering with grounding and confidence."""

    def __init__(self, config: dict):
        self.config = config
        mm_cfg = config.get("multimodal", {})
        llm_cfg = config.get("llm", {})

        # Vision model config — use Groq's Llama-4-Scout (vision-capable)
        self.model = mm_cfg.get("model", "meta-llama/llama-4-scout-17b-16e-instruct")
        self.temperature = mm_cfg.get("temperature", 0)
        self.max_tokens = mm_cfg.get("max_tokens", 1024)
        self.seed = config.get("seed", 42)

        # API client — same Groq endpoint
        api_key = os.getenv(
            mm_cfg.get("api_key_env", llm_cfg.get("api_key_env", "GROQ_API_KEY")), ""
        )
        base_url = mm_cfg.get("base_url", llm_cfg.get("base_url", "https://api.groq.com/openai/v1"))

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._available = bool(api_key)

        # Inference log
        self.inference_log = config.get("logging", {}).get(
            "multimodal_log", "logs/multimodal_log.jsonl"
        )

        if not self._available:
            logger.warning("Multimodal pipeline: no API key set — VLM calls will fail")

        logger.info(
            "MultimodalPipeline initialised (model=%s, available=%s)",
            self.model, self._available,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        image_path: str,
        question: str,
        return_grounding: bool = True,
        sample_id: str = "",
    ) -> dict:
        """Answer a visual question about an image.

        Returns dict matching /v1/multimodal response schema:
            answer, confidence, grounding, safety_triggered, latency_ms
        """
        start = time.perf_counter()

        if not self._available:
            return self._error_result("VLM API key not configured", start)

        # 1. Load and encode image as base64
        image_b64, media_type = self._encode_image(image_path)
        if image_b64 is None:
            return self._error_result(f"Failed to load image: {image_path}", start)

        # 2. Optional OCR augmentation
        ocr_text = self._extract_ocr(image_path)
        augmented_question = question
        if ocr_text:
            augmented_question = f"{question}\n\nOCR-extracted text from image: {ocr_text}"

        # 3. Build system prompt
        system = MULTIMODAL_SYSTEM
        if not return_grounding:
            system = system.replace(
                '- "grounding": description of the image region you are reasoning about (string)\n',
                ""
            )

        # 4. Call VLM with retry
        structured = self._call_vlm(image_b64, media_type, augmented_question, system)
        latency_ms = int((time.perf_counter() - start) * 1000)

        if structured is None:
            return self._error_result("VLM call failed", start)

        result = {
            "answer": structured.get("answer", ""),
            "reasoning": structured.get("reasoning", ""),
            "confidence": self._clamp_confidence(structured.get("confidence", 0.5)),
            "grounding": structured.get("grounding", ""),
            "safety_triggered": False,
            "latency_ms": latency_ms,
        }

        # 5. Log inference
        self._log_inference(sample_id, question, result)

        return result

    @property
    def status(self) -> str:
        return "ok" if self._available else "no_token"

    # ------------------------------------------------------------------
    # VLM call with retry
    # ------------------------------------------------------------------

    def _call_vlm(
        self,
        image_b64: str,
        media_type: str,
        question: str,
        system: str,
    ) -> Optional[dict]:
        """Call the vision-language model and parse structured JSON response."""
        import time as _time

        data_url = f"data:{media_type};base64,{image_b64}"
        raw = ""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait = 2 ** attempt
                    logger.info("VLM retry %d/%d after %ds…", attempt + 1, max_retries, wait)
                    _time.sleep(wait)

                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": data_url},
                                },
                            ],
                        },
                    ],
                )

                raw = resp.choices[0].message.content.strip()
                logger.debug("VLM raw response: %s", raw[:300])

                # Parse JSON — handle possible markdown wrapping
                cleaned = raw
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

                return json.loads(cleaned)

            except json.JSONDecodeError as exc:
                logger.error("VLM response not valid JSON: %s — raw: %s", exc, raw[:200])
                # Last attempt: return best-effort parse
                if attempt == max_retries - 1:
                    return {
                        "answer": raw[:500] if raw else "",
                        "reasoning": "",
                        "confidence": 0.3,
                        "grounding": "unable to parse structured response",
                    }
            except Exception as exc:
                logger.error("VLM call attempt %d failed: %s", attempt + 1, exc)
                if attempt == max_retries - 1:
                    return None

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def _encode_image(self, image_path: str) -> tuple[Optional[str], str]:
        """Load image from path and return base64 string + media type."""
        try:
            path = Path(image_path)
            if not path.exists():
                logger.error("Image not found: %s", image_path)
                return None, ""

            # Determine media type
            suffix = path.suffix.lower()
            media_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            media_type = media_map.get(suffix, "image/jpeg")

            # Read and encode
            with open(path, "rb") as f:
                image_bytes = f.read()

            # Resize if too large (>10MB)
            if len(image_bytes) > 10 * 1024 * 1024:
                img = Image.open(io.BytesIO(image_bytes))
                img.thumbnail((1024, 1024), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                image_bytes = buf.getvalue()
                media_type = "image/jpeg"

            b64 = base64.b64encode(image_bytes).decode("utf-8")
            return b64, media_type

        except Exception as exc:
            logger.error("Image encoding failed: %s", exc)
            return None, ""

    @staticmethod
    def _encode_image_bytes(image_bytes: bytes, content_type: str = "image/jpeg") -> tuple[str, str]:
        """Encode raw image bytes to base64."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return b64, content_type

    # ------------------------------------------------------------------
    # OCR augmentation
    # ------------------------------------------------------------------

    def _extract_ocr(self, image_path: str) -> str:
        """Extract text from image using pytesseract (optional)."""
        try:
            import pytesseract
            text = pytesseract.image_to_string(Image.open(image_path)).strip()
            if text and len(text) > 3:
                logger.debug("OCR extracted %d chars from %s", len(text), image_path)
                return text
        except ImportError:
            logger.debug("pytesseract not installed — skipping OCR augmentation")
        except Exception as exc:
            logger.debug("OCR extraction failed: %s", exc)
        return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_confidence(value) -> float:
        """Ensure confidence is a float in [0.0, 1.0]."""
        try:
            v = float(value)
            return max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            return 0.5

    def _error_result(self, error_msg: str, start: float) -> dict:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "answer": "",
            "reasoning": "",
            "confidence": 0.0,
            "grounding": "",
            "safety_triggered": False,
            "latency_ms": latency_ms,
            "error": error_msg,
        }

    def _log_inference(self, sample_id: str, question: str, result: dict):
        record = {
            "ts": utc_now_iso(),
            "id": sample_id,
            "question": question[:500],
            "answer": result.get("answer", "")[:500],
            "confidence": result.get("confidence", 0.0),
            "grounding": result.get("grounding", ""),
            "latency_ms": result.get("latency_ms", 0),
        }
        append_jsonl(self.inference_log, record)
