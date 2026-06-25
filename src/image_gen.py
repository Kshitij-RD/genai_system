"""Image generation pipeline — prompt optimisation, HuggingFace SDXL, CLIP scoring, sidecar JSON."""

import io
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from openai import OpenAI
from PIL import Image

from src.utils import append_jsonl, utc_now_iso

logger = logging.getLogger("genai_system.image_gen")

# ---------------------------------------------------------------------------
# Prompt optimisation system prompt (CLIP ≤77 token constraint)
# ---------------------------------------------------------------------------
OPTIMISE_SYSTEM = (
    "You are an expert Stable Diffusion XL prompt engineer.\n"
    "Rewrite the user's prompt into a rich, highly detailed generation prompt.\n"
    "Add specific details about: lighting style, camera angle, art style, "
    "texture, mood, color palette, and background environment.\n"
    "Return ONLY the prompt text, no explanation, no quotes.\n"
    "CRITICAL: Keep the output under 77 tokens (CLIP limit)."
)

DEFAULT_NEGATIVE_PROMPT = "blurry, watermark, low resolution, deformed, artifacts"


class ImageGenPipeline:
    """Wraps image generation with prompt optimisation, CLIP scoring, and sidecar metadata."""

    def __init__(self, config: dict):
        self.config = config
        img_cfg = config.get("image_gen", {})
        llm_cfg = config.get("llm", {})

        # Paths
        self.output_dir = Path(img_cfg.get("output_dir", "data/generated/"))
        self.static_dir = Path(img_cfg.get("static_dir", "static/generated/"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir.mkdir(parents=True, exist_ok=True)

        # Image generation config
        self.model_name = img_cfg.get("model", "@cf/stabilityai/stable-diffusion-xl-base-1.0")
        self.steps = img_cfg.get("steps", 20)
        self.guidance_scale = img_cfg.get("guidance_scale", 7.5)
        self.negative_prompt = img_cfg.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        self.default_seed = config.get("seed", 42)

        # Cloudflare Workers AI
        self.cf_account_id = os.getenv(img_cfg.get("account_id_env", "CF_ACCOUNT_ID"), "")
        self.cf_token = os.getenv(img_cfg.get("api_key_env", "CF_API_TOKEN"), "")
        self._cf_available = bool(self.cf_account_id and self.cf_token)

        # Groq LLM client for prompt optimisation
        self._llm_client = OpenAI(
            base_url=llm_cfg["base_url"],
            api_key=os.getenv(llm_cfg["api_key_env"], ""),
        )
        self._llm_model = llm_cfg["model"]

        # CLIP model (lazy loaded)
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._clip_available = False

        # Inference log
        self.inference_log = config.get("logging", {}).get(
            "image_log", "logs/image_log.jsonl"
        )

        if not self._cf_available:
            logger.warning(
                "CF_ACCOUNT_ID or CF_API_TOKEN not set. Image generation unavailable. "
                "Set both in .env to enable Cloudflare Workers AI."
            )

        logger.info("ImageGenPipeline initialised (model=%s, cf_available=%s)",
                     self.model_name, self._cf_available)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        sample_id: str = "",
    ) -> dict:
        """Generate an image from a text prompt.

        Returns dict with all /v1/image response fields.
        """
        start = time.perf_counter()
        seed = seed if seed is not None else self.default_seed
        if not sample_id:
            sample_id = f"img_{uuid.uuid4().hex[:5]}"

        # 1. Optimise prompt via LLM
        optimised_prompt = self._optimise_prompt(prompt)

        # 2. Generate image via HuggingFace Inference API
        image = self._generate_image(optimised_prompt, seed)

        generation_time_ms = int((time.perf_counter() - start) * 1000)

        if image is None:
            return {
                "image_url": "",
                "optimised_prompt": optimised_prompt,
                "clip_score": 0.0,
                "safety_triggered": False,
                "generation_time_ms": generation_time_ms,
                "seed": seed,
                "error": "Image generation failed — check HF_API_TOKEN",
            }

        # 3. Save image to data/generated/ and static/generated/
        image_filename = f"{sample_id}.png"
        data_path = self.output_dir / image_filename
        static_path = self.static_dir / image_filename
        image.save(str(data_path))
        shutil.copy2(str(data_path), str(static_path))

        # 4. Compute CLIP scores
        clip_raw = self._clip_score(str(data_path), prompt)
        clip_optimised = self._clip_score(str(data_path), optimised_prompt)

        generation_time_ms = int((time.perf_counter() - start) * 1000)

        # 5. Save sidecar JSON
        sidecar = {
            "id": sample_id,
            "raw_prompt": prompt,
            "optimised_prompt": optimised_prompt,
            "negative_prompt": self.negative_prompt,
            "seed": seed,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "model": self.model_name,
            "image_path": str(data_path).replace("\\", "/"),
            "clip_score_raw": round(clip_raw, 4),
            "clip_score_optimised": round(clip_optimised, 4),
            "clip_delta": round(clip_optimised - clip_raw, 4),
        }
        sidecar_path = self.output_dir / f"{sample_id}.json"
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)

        # 6. Log inference
        self._log_inference(sample_id, prompt, optimised_prompt,
                            clip_raw, clip_optimised, generation_time_ms, seed)

        image_url = f"/static/generated/{image_filename}"
        return {
            "image_url": image_url,
            "optimised_prompt": optimised_prompt,
            "clip_score": round(clip_optimised, 4),
            "safety_triggered": False,
            "generation_time_ms": generation_time_ms,
            "seed": seed,
        }

    @property
    def status(self) -> str:
        """Health check status."""
        if self._cf_available:
            return "ok"
        return "no_token"

    # ------------------------------------------------------------------
    # Prompt optimisation
    # ------------------------------------------------------------------

    def _optimise_prompt(self, raw_prompt: str) -> str:
        """Expand a short prompt into a detailed, CLIP-constrained image prompt."""
        try:
            resp = self._llm_client.chat.completions.create(
                model=self._llm_model,
                temperature=0.7,
                max_tokens=150,
                messages=[
                    {"role": "system", "content": OPTIMISE_SYSTEM},
                    {"role": "user", "content": raw_prompt},
                ],
            )
            optimised = resp.choices[0].message.content.strip()
            # Strip any wrapping quotes the LLM might add
            if optimised.startswith('"') and optimised.endswith('"'):
                optimised = optimised[1:-1]
            logger.info("Prompt optimised: '%s' → '%s'", raw_prompt[:60], optimised[:80])
            return optimised
        except Exception as exc:
            logger.warning("Prompt optimisation failed, using raw: %s", exc)
            return raw_prompt

    # ------------------------------------------------------------------
    # Image generation via Cloudflare Workers AI
    # ------------------------------------------------------------------

    def _generate_image(self, prompt: str, seed: int) -> Optional[Image.Image]:
        """Call Cloudflare Workers AI to generate an image with retry."""
        if not self._cf_available:
            logger.error("Cannot generate image: CF_ACCOUNT_ID or CF_API_TOKEN not set")
            return None

        import requests as req_lib

        api_url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self.cf_account_id}/ai/run/{self.model_name}"
        )
        headers = {
            "Authorization": f"Bearer {self.cf_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "negative_prompt": self.negative_prompt,
            "num_steps": self.steps,
            "guidance": self.guidance_scale,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                wait = 2 ** attempt
                if attempt > 0:
                    logger.info("Retry %d/%d after %ds…", attempt + 1, max_retries, wait)
                    time.sleep(wait)

                logger.info("Calling CF Workers AI: %s (attempt %d)", self.model_name, attempt + 1)
                response = req_lib.post(api_url, headers=headers, json=payload, timeout=180)

                # 429 = rate limited
                if response.status_code == 429:
                    logger.warning("Rate limited, backing off…")
                    time.sleep(wait * 5)
                    continue

                if response.status_code != 200:
                    logger.error("CF API returned %d: %s",
                                 response.status_code, response.text[:500])
                    if attempt < max_retries - 1:
                        continue
                    return None

                # CF returns raw PNG bytes for image models
                content_type = response.headers.get("content-type", "")
                if "image" in content_type:
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                else:
                    # Some CF responses wrap in JSON with base64
                    try:
                        import base64
                        data = response.json()
                        if "result" in data and "image" in data["result"]:
                            img_bytes = base64.b64decode(data["result"]["image"])
                            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        else:
                            logger.error("Unexpected CF response format: %s", str(data)[:300])
                            if attempt < max_retries - 1:
                                continue
                            return None
                    except Exception:
                        logger.error("Could not parse CF response as image or JSON")
                        if attempt < max_retries - 1:
                            continue
                        return None

                logger.info("Image generated successfully (%dx%d)", image.width, image.height)
                return image

            except Exception as exc:
                logger.error("Image generation attempt %d failed: %s", attempt + 1, exc)
                if attempt == max_retries - 1:
                    return None

    # ------------------------------------------------------------------
    # CLIP scoring
    # ------------------------------------------------------------------

    def _ensure_clip_loaded(self):
        """Lazy-load the CLIP model."""
        if self._clip_model is not None:
            return
        try:
            import open_clip
            import torch  # noqa: F401

            clip_cfg = self.config.get("image_gen", {})
            model_name = clip_cfg.get("clip_model", "ViT-B-32")
            pretrained = clip_cfg.get("clip_pretrained", "laion2b_s34b_b79k")

            logger.info("Loading CLIP model: %s (%s)", model_name, pretrained)
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            tokenizer = open_clip.get_tokenizer(model_name)

            self._clip_model = model
            self._clip_preprocess = preprocess
            self._clip_tokenizer = tokenizer
            self._clip_available = True
            logger.info("CLIP model loaded successfully")
        except Exception as exc:
            logger.warning("CLIP model not available: %s", exc)
            self._clip_available = False

    def _clip_score(self, image_path: str, prompt: str) -> float:
        """Compute cosine similarity between image and text using CLIP."""
        try:
            self._ensure_clip_loaded()
            if not self._clip_available:
                return 0.0

            import torch

            image = self._clip_preprocess(
                Image.open(image_path).convert("RGB")
            ).unsqueeze(0)
            text = self._clip_tokenizer([prompt])

            with torch.no_grad():
                img_feat = self._clip_model.encode_image(image)
                txt_feat = self._clip_model.encode_text(text)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            score = (img_feat @ txt_feat.T).item()
            return score
        except Exception as exc:
            logger.warning("CLIP scoring failed: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_inference(self, sample_id, raw_prompt, optimised_prompt,
                       clip_raw, clip_optimised, generation_time_ms, seed):
        record = {
            "ts": utc_now_iso(),
            "id": sample_id,
            "raw_prompt": raw_prompt[:500],
            "optimised_prompt": optimised_prompt[:500],
            "clip_score_raw": round(clip_raw, 4),
            "clip_score_optimised": round(clip_optimised, 4),
            "clip_delta": round(clip_optimised - clip_raw, 4),
            "generation_time_ms": generation_time_ms,
            "seed": seed,
        }
        append_jsonl(self.inference_log, record)