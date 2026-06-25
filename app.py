"""FastAPI application — /v1/text, /v1/image, /v1/multimodal and /v1/health endpoints."""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.prompts import PromptManager
from src.safety import SafetyFilter
from src.text_gen import TextGenPipeline
from src.image_gen import ImageGenPipeline
from src.multimodal import MultimodalPipeline
from src.infer import infer_text, infer_multimodal
from src.utils import load_config, setup_logging, seed_everything

logger = logging.getLogger("genai_system.app")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TextRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    category: str = Field(default="factual", pattern="^(factual|creative|reasoning)$")


class TextResponse(BaseModel):
    output: str
    reasoning: str
    schema_valid: bool
    hallucination_flagged: bool
    safety_triggered: bool
    latency_ms: int
    prompt_version: str
    tokens_used: int


class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    seed: int = Field(default=42)


class ImageResponse(BaseModel):
    image_url: str
    optimised_prompt: str
    clip_score: float
    safety_triggered: bool
    generation_time_ms: int
    seed: int


class MultimodalResponse(BaseModel):
    answer: str
    confidence: float
    grounding: str
    safety_triggered: bool
    latency_ms: int


class HealthResponse(BaseModel):
    text_model: str
    image_model: str
    multimodal_model: str
    safety_filter: str


# ---------------------------------------------------------------------------
# Lifespan — preload all models once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging("INFO")
    logger.info("Starting up…")

    try:
        config = load_config()
        seed_everything(config.get("seed", 42))

        app.state.config = config
        app.state.prompt_mgr = PromptManager(
            prompts_dir=config["prompts"]["dir"],
            few_shot_dir=config["prompts"].get("few_shot_dir", "data/few_shot/"),
            encoder_model=config["prompts"]["encoder_model"],
            context_window=config["prompts"]["context_window"],
        )
        app.state.safety = SafetyFilter(config)
        app.state.pipeline = TextGenPipeline(config)
        app.state.image_pipeline = ImageGenPipeline(config)
        app.state.multimodal_pipeline = MultimodalPipeline(config)
        logger.info("All models loaded successfully")
    except Exception as exc:
        logger.critical("Startup failed: %s", exc)
        raise RuntimeError(f"Model loading failed: {exc}")

    yield
    logger.info("Shutting down…")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GenAI System — Text & Image Generation",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files for serving generated images
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/text", response_model=TextResponse, status_code=200)
def generate_text(req: TextRequest):
    """Generate structured text with CoT reasoning and safety checks."""
    request_id = str(uuid.uuid4())[:8]

    result = infer_text(
        prompt=req.prompt,
        category=req.category,
        sample_id=request_id,
        pipeline=app.state.pipeline,
        prompt_mgr=app.state.prompt_mgr,
        safety=app.state.safety,
    )

    if result.get("safety_triggered"):
        raise HTTPException(
            status_code=451,
            detail={
                "safety_triggered": True,
                "category": result.get("safety_category", "unknown"),
                "blocked_at": result.get("safety_blocked_at", "input"),
            },
        )

    return TextResponse(
        output=result["output"],
        reasoning=result.get("reasoning", ""),
        schema_valid=result["schema_valid"],
        hallucination_flagged=result["hallucination_flagged"],
        safety_triggered=False,
        latency_ms=result["latency_ms"],
        prompt_version=result["prompt_version"],
        tokens_used=result["tokens_used"],
    )


@app.get("/v1/health", response_model=HealthResponse)
def health():
    """Report status of loaded models."""
    return HealthResponse(
        text_model=app.state.pipeline.status,
        image_model=app.state.image_pipeline.status,
        multimodal_model=app.state.multimodal_pipeline.status,
        safety_filter=app.state.safety.status,
    )


@app.post("/v1/image", response_model=ImageResponse, status_code=200)
def generate_image(req: ImageRequest):
    """Generate an image with prompt optimisation and CLIP scoring."""
    request_id = f"img_{uuid.uuid4().hex[:5]}"

    # Input safety check
    safety_result = app.state.safety.check_input(req.prompt, request_id=request_id)
    if not safety_result.safe:
        raise HTTPException(
            status_code=451,
            detail={
                "safety_triggered": True,
                "category": safety_result.category,
                "blocked_at": safety_result.blocked_at,
            },
        )

    result = app.state.image_pipeline.generate(
        prompt=req.prompt,
        seed=req.seed,
        sample_id=request_id,
    )

    if result.get("error"):
        raise HTTPException(status_code=503, detail=result["error"])

    # Output image safety check (CLIP zero-shot)
    # NOTE: generate() already saved to disk — must clean up if unsafe
    if result.get("image_url"):
        image_path = f"static/generated/{request_id}.png"
        img_safety = app.state.safety.check_image(image_path, request_id=request_id)
        if not img_safety.safe:
            # Delete leaked files from both directories before returning 451
            _cleanup_unsafe_image(request_id, app.state.image_pipeline)
            raise HTTPException(
                status_code=451,
                detail={
                    "safety_triggered": True,
                    "category": img_safety.category,
                    "blocked_at": img_safety.blocked_at,
                },
            )

    return ImageResponse(
        image_url=result["image_url"],
        optimised_prompt=result["optimised_prompt"],
        clip_score=result["clip_score"],
        safety_triggered=False,
        generation_time_ms=result["generation_time_ms"],
        seed=result["seed"],
    )


def _cleanup_unsafe_image(request_id: str, pipeline) -> None:
    """Remove unsafe image from all locations to prevent filesystem leakage."""
    from pathlib import Path

    paths = [
        pipeline.output_dir / f"{request_id}.png",
        pipeline.output_dir / f"{request_id}.json",
        pipeline.static_dir / f"{request_id}.png",
    ]
    for p in paths:
        try:
            if Path(p).exists():
                Path(p).unlink()
                logger.info("Deleted unsafe file: %s", p)
        except OSError as exc:
            logger.warning("Failed to delete %s: %s", p, exc)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001)


# ---------------------------------------------------------------------------
# Multimodal endpoint — multipart/form-data
# ---------------------------------------------------------------------------

@app.post("/v1/multimodal", response_model=MultimodalResponse, status_code=200)
async def multimodal_vqa(
    image: UploadFile = File(...),
    question: str = Form(...),
    return_grounding: bool = Form(True),
):
    """Visual question answering with grounded response and confidence scoring.

    Accepts multipart/form-data with image file + question text.
    """
    import tempfile
    import shutil
    from pathlib import Path

    request_id = f"mm_{uuid.uuid4().hex[:5]}"

    # Save uploaded image to temp file
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="static/generated/") as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        result = infer_multimodal(
            image_path=tmp_path,
            question=question,
            sample_id=request_id,
            return_grounding=return_grounding,
            pipeline=app.state.multimodal_pipeline,
            safety=app.state.safety,
        )

        if result.get("safety_triggered"):
            raise HTTPException(
                status_code=451,
                detail={
                    "safety_triggered": True,
                    "category": result.get("safety_category", "unknown"),
                    "blocked_at": result.get("safety_blocked_at", "input"),
                },
            )

        if result.get("error"):
            raise HTTPException(status_code=503, detail=result["error"])

        return MultimodalResponse(
            answer=result["answer"],
            confidence=result["confidence"],
            grounding=result["grounding"],
            safety_triggered=False,
            latency_ms=result["latency_ms"],
        )
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass