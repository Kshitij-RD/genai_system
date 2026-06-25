"""Inference wrappers — orchestrate safety → generation → response for text and image."""

import logging
import time
from pathlib import Path
from typing import Optional

from src.prompts import PromptManager
from src.safety import SafetyFilter, SafetyResult
from src.text_gen import TextGenPipeline
from src.image_gen import ImageGenPipeline
from src.multimodal import MultimodalPipeline

logger = logging.getLogger("genai_system.infer")

# Template selection heuristics based on category
_CATEGORY_TO_TEMPLATE = {
    "factual": "explain_v1",
    "creative": "creative_v1",
    "reasoning": "qa_v1",
}


def infer_text(
    prompt: str,
    category: str = "factual",
    sample_id: str = "",
    reference: Optional[str] = None,
    *,
    pipeline: TextGenPipeline,
    prompt_mgr: PromptManager,
    safety: SafetyFilter,
    k_shot: int = 3,
) -> dict:
    """Full text inference: safety check → prompt render → LLM → response.

    Returns dict with all /v1/text response fields.
    """
    start = time.perf_counter()

    # 1. Input safety check
    safety_result = safety.check_input(prompt, request_id=sample_id)
    if not safety_result.safe:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return _safety_response(safety_result, latency_ms)

    # 2. Select template and render prompt
    template = _CATEGORY_TO_TEMPLATE.get(category, "qa_v1")
    if template not in prompt_mgr.templates:
        template = list(prompt_mgr.templates.keys())[0]

    system_prompt, user_prompt, version = prompt_mgr.render(
        template, {"input_text": prompt}, k_shot=k_shot
    )

    # 3. Generate via LLM pipeline
    result = pipeline.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_version=version,
        sample_id=sample_id,
        reference=reference,
    )

    # 4. Output safety check
    output_safety = safety.check_output(result["output"], request_id=sample_id)
    if not output_safety.safe:
        return _safety_response(output_safety, result.get("latency_ms", 0))

    # Propagate combined safety check duration so eval.py latency_overhead is accurate
    result["safety_check_ms"] = (
        safety_result.check_duration_ms + output_safety.check_duration_ms
    )
    return result


def infer_image(
    prompt: str,
    seed: int = 42,
    sample_id: str = "",
    *,
    pipeline: ImageGenPipeline,
    safety: SafetyFilter,
) -> dict:
    """Full image inference: input safety → generate → output safety → response.

    Mirrors the safety flow in app.py /v1/image but as a testable wrapper
    that eval.py can call to get accurate safety metrics for image samples.

    Returns dict with image response fields + safety_triggered + safety_check_ms.
    """
    start = time.perf_counter()

    # 1. Input safety check (injection + harmful topics + toxicity)
    input_safety = safety.check_input(prompt, request_id=sample_id)
    if not input_safety.safe:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "image_url": "",
            "optimised_prompt": "",
            "clip_score": 0.0,
            "safety_triggered": True,
            "safety_category": input_safety.category,
            "safety_blocked_at": input_safety.blocked_at,
            "safety_check_ms": input_safety.check_duration_ms,
            "generation_time_ms": latency_ms,
            "seed": seed,
        }

    # 2. Generate image
    result = pipeline.generate(prompt=prompt, seed=seed, sample_id=sample_id)

    if result.get("error"):
        result["safety_triggered"] = False
        result["safety_check_ms"] = input_safety.check_duration_ms
        return result

    # 3. Output image safety check (CLIP zero-shot)
    image_path = pipeline.output_dir / f"{sample_id}.png"
    output_safety = SafetyResult(safe=True, check_duration_ms=0)

    if image_path.exists():
        output_safety = safety.check_image(str(image_path), request_id=sample_id)
        if not output_safety.safe:
            # Clean up unsafe image from both directories
            _cleanup_unsafe_image(sample_id, pipeline)
            return {
                "image_url": "",
                "optimised_prompt": result.get("optimised_prompt", ""),
                "clip_score": 0.0,
                "safety_triggered": True,
                "safety_category": output_safety.category,
                "safety_blocked_at": output_safety.blocked_at,
                "safety_check_ms": (
                    input_safety.check_duration_ms + output_safety.check_duration_ms
                ),
                "generation_time_ms": result.get("generation_time_ms", 0),
                "seed": seed,
            }

    # Propagate combined safety check duration
    result["safety_triggered"] = False
    result["safety_check_ms"] = (
        input_safety.check_duration_ms + output_safety.check_duration_ms
    )
    return result


def _cleanup_unsafe_image(sample_id: str, pipeline: ImageGenPipeline) -> None:
    """Remove unsafe image files from all locations."""
    paths = [
        pipeline.output_dir / f"{sample_id}.png",
        pipeline.output_dir / f"{sample_id}.json",
        pipeline.static_dir / f"{sample_id}.png",
    ]
    for p in paths:
        try:
            if p.exists():
                p.unlink()
                logger.info("Deleted unsafe file: %s", p)
        except OSError as exc:
            logger.warning("Failed to delete %s: %s", p, exc)


def _safety_response(result: SafetyResult, latency_ms: int = 0) -> dict:
    """Build a safety-blocked response dict."""
    return {
        "output": "",
        "reasoning": "",
        "schema_valid": True,
        "hallucination_flagged": False,
        "safety_triggered": True,
        "safety_category": result.category,
        "safety_blocked_at": result.blocked_at,
        "safety_check_ms": result.check_duration_ms,
        "latency_ms": latency_ms,
        "prompt_version": "",
        "tokens_used": 0,
    }


def infer_multimodal(
    image_path: str,
    question: str,
    sample_id: str = "",
    return_grounding: bool = True,
    *,
    pipeline: MultimodalPipeline,
    safety: SafetyFilter,
) -> dict:
    """Full multimodal inference: safety check → VLM → output safety → response.

    Returns dict with all /v1/multimodal response fields.
    """
    start = time.perf_counter()

    # 1. Input safety check on the question text
    safety_result = safety.check_input(question, request_id=sample_id)
    if not safety_result.safe:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return _mm_safety_response(safety_result, latency_ms)

    # 2. Call VLM pipeline
    result = pipeline.answer(
        image_path=image_path,
        question=question,
        return_grounding=return_grounding,
        sample_id=sample_id,
    )

    if result.get("error"):
        return result

    # 3. Output safety check on the answer text
    output_safety = safety.check_output(result.get("answer", ""), request_id=sample_id)
    if not output_safety.safe:
        return _mm_safety_response(output_safety, result.get("latency_ms", 0))

    # Propagate safety check duration
    result["safety_check_ms"] = (
        safety_result.check_duration_ms + output_safety.check_duration_ms
    )
    return result


def _mm_safety_response(result: SafetyResult, latency_ms: int = 0) -> dict:
    """Build a safety-blocked response dict for multimodal."""
    return {
        "answer": "",
        "confidence": 0.0,
        "grounding": "",
        "safety_triggered": True,
        "safety_category": result.category,
        "safety_blocked_at": result.blocked_at,
        "safety_check_ms": result.check_duration_ms,
        "latency_ms": latency_ms,
    }