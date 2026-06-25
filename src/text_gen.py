"""Text generation pipeline — structured output, CoT, hallucination detection."""

import json
import logging
import os
import time
from typing import Optional

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from src.utils import append_jsonl

logger = logging.getLogger("genai_system.text_gen")


# ---------------------------------------------------------------------------
# Pydantic schema for structured LLM output (two-step CoT)
# ---------------------------------------------------------------------------

class TextGenResponse(BaseModel):
    """Two-step Chain-of-Thought structured output."""
    reasoning: str = Field(description="Step-by-step chain of thought")
    answer: str = Field(description="Final extracted answer")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TextGenPipeline:
    """Wraps LLM calls with Instructor structured output, CoT extraction,
    and NLI-based hallucination detection."""

    def __init__(self, config: dict):
        self.config = config
        llm = config["llm"]
        self.model = llm["model"]
        self.temperature = llm.get("temperature", 0)
        self.seed = llm.get("seed", 42)
        self.max_tokens = llm.get("max_tokens", 1024)
        self.inference_log = config.get("logging", {}).get(
            "inference_log", "logs/inference_log.jsonl"
        )

        # Instructor-wrapped OpenAI-compatible client (Groq)
        self.raw_client = OpenAI(
            base_url=llm["base_url"],
            api_key=os.getenv(llm["api_key_env"], ""),
        )
        self.client = instructor.from_openai(self.raw_client, mode=instructor.Mode.JSON)

        # Lazy-loaded NLI model for hallucination detection
        self._nli_model = None
        self._nli_tokenizer = None
        self.hall_threshold = config.get("eval", {}).get(
            "hallucination_threshold", 0.5
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
        sample_id: str = "",
        reference: Optional[str] = None,
    ) -> dict:
        """Run full text-gen pipeline: LLM → structured output → hallucination check.

        Returns dict matching the /v1/text response schema.
        """
        start = time.perf_counter()
        schema_valid = True

        # Retry with exponential backoff for rate limits / transient errors
        max_retries = 3
        structured = None
        raw_output = ""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait = 2 ** attempt
                    logger.info("Retry %d/%d after %ds…", attempt + 1, max_retries, wait)
                    time.sleep(wait)

                resp: TextGenResponse = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    seed=self.seed,
                    max_tokens=self.max_tokens,
                    response_model=TextGenResponse,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                structured = resp.model_dump()
                raw_output = json.dumps(structured)
                break
            except Exception as exc:
                logger.error("LLM call attempt %d failed: %s", attempt + 1, exc)
                if attempt == max_retries - 1:
                    structured = {"reasoning": "", "answer": str(exc)}
                    raw_output = str(exc)
                    schema_valid = False

        latency_ms = int((time.perf_counter() - start) * 1000)

        # Hallucination detection (only when reference exists)
        hallucination_flagged = False
        if reference and schema_valid:
            hallucination_flagged = self.detect_hallucination(
                structured.get("answer", ""), reference
            )

        tokens_used = self._estimate_tokens(system_prompt, user_prompt, raw_output)

        result = {
            "output": structured.get("answer", ""),
            "reasoning": structured.get("reasoning", ""),
            "schema_valid": schema_valid,
            "hallucination_flagged": hallucination_flagged,
            "safety_triggered": False,
            "latency_ms": latency_ms,
            "prompt_version": prompt_version,
            "tokens_used": tokens_used,
        }

        # Per-sample logging
        self._log_inference(sample_id, user_prompt, prompt_version,
                            raw_output, structured, hallucination_flagged,
                            latency_ms, tokens_used)
        return result

    # ------------------------------------------------------------------
    # Hallucination detection via NLI
    # ------------------------------------------------------------------

    def detect_hallucination(self, generated: str, reference: str) -> bool:
        """Return True if generated text contradicts the reference."""
        if not generated or not reference:
            return False
        try:
            self._ensure_nli_loaded()
            import torch

            inputs = self._nli_tokenizer(
                reference, generated,
                return_tensors="pt", truncation=True, max_length=512,
            )
            inputs = {k: v.to(self._nli_device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._nli_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)

            # Use auto-detected label index for contradiction
            contradiction = probs[0][self._contradiction_idx].item()
            return contradiction > self.hall_threshold
        except Exception as exc:
            logger.warning("Hallucination check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_nli_loaded(self):
        if self._nli_model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        name = self.config.get("eval", {}).get(
            "hallucination_model", "cross-encoder/nli-deberta-v3-small"
        )
        self._nli_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading NLI model: %s (device=%s)", name, self._nli_device)
        self._nli_tokenizer = AutoTokenizer.from_pretrained(name)
        self._nli_model = AutoModelForSequenceClassification.from_pretrained(name)
        self._nli_model.to(self._nli_device)
        self._nli_model.eval()

        # Auto-detect label order from model config
        label2id = getattr(self._nli_model.config, "label2id", {})
        logger.info("NLI label mapping: %s", label2id)

        # Find contradiction index — different models use different orders
        self._contradiction_idx = 0  # default for cross-encoder models
        for label, idx in label2id.items():
            if "contradiction" in label.lower():
                self._contradiction_idx = idx
                break
        logger.info("Using contradiction index: %d", self._contradiction_idx)

    @staticmethod
    def _estimate_tokens(*texts: str) -> int:
        total_words = sum(len(t.split()) for t in texts)
        return int(total_words * 1.3)

    def _log_inference(self, sample_id, prompt, version, raw, structured,
                       hall, latency, tokens):
        record = {
            "id": sample_id,
            "prompt": prompt[:500],
            "prompt_version": version,
            "raw_output": raw[:1000],
            "structured_output": structured,
            "hallucination_flagged": hall,
            "latency_ms": latency,
            "tokens_used": tokens,
        }
        append_jsonl(self.inference_log, record)

    @property
    def status(self) -> str:
        """Quick health check — is the LLM client configured?"""
        try:
            if self.raw_client and self.model:
                return "ok"
            return "error"
        except Exception:
            return "error"