"""Content safety filter — gates ALL inputs and outputs.
 
Checks (in order):
  1. Prompt injection detection   (input only)
  2. Toxicity / hate / violence   (input + output)
  3. PII leakage                  (output only)
  4. Image safety via CLIP        (image output only)
 
Every violation is logged to logs/safety_log.jsonl.
check_input() / check_output() / check_image() each return a SafetyResult
that includes check_duration_ms — time spent INSIDE the safety check only,
NOT full pipeline latency. eval.py reads this for latency_overhead_ms.
"""
 
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
 
from src.utils import append_jsonl, utc_now_iso
 
logger = logging.getLogger("genai_system.safety")
 
# ---------------------------------------------------------------------------
# Injection patterns
# ---------------------------------------------------------------------------
 
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(your|all)\s+(previous\s+)?(instructions?|rules?|guidelines?)", re.I),
    re.compile(r"(you are|act as|pretend (you are|to be))\s+(DAN|an evil|a jailbroken|an unrestricted)", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"bypass\s+(your\s+)?(safety|content|filter)", re.I),
    re.compile(r"(no\s+restrictions?|without\s+restrictions?|remove\s+all\s+restrictions?)", re.I),
    re.compile(r"from\s+now\s+on\s+you\s+(will|must|should)\s+(ignore|disregard)", re.I),
]

# ---------------------------------------------------------------------------
# Harmful topic patterns — catches dangerous requests not covered by injection
# ---------------------------------------------------------------------------

_HARMFUL_PATTERNS = [
    re.compile(r"\b(make|build|create|synthesize|manufacture|produce)\b.{0,50}\b(bomb|explosive|meth|drug|poison|weapon|grenade|fentanyl)\b", re.I),
    re.compile(r"\b(how\s+to|instructions?\s+for|steps?\s+(to|for)|guide\s+(to|for))\b.{0,50}\b(murder|kill|attack|hack|steal|phish|stalk|traffic)\b", re.I),
    re.compile(r"\b(terrorist|terrorism|suicide|self[\s-]harm|white\s+supremac|child\s+(sex|porn|abuse|exploit))\b", re.I),
    re.compile(r"\b(steal|fake|forge|spoof).{0,40}\b(identity|credential|password|credit\s+card|ssn|social\s+security)\b", re.I),
    re.compile(r"\b(commit|plan|carry\s+out|execute).{0,40}\b(murder|killing|attack|assault|robbery)\b", re.I),
]
 
# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
 
@dataclass
class SafetyResult:
    safe:              bool
    category:          str = "safe"
    blocked_at:        str = ""
    detail:            str = ""
    check_duration_ms: int = 0   # time spent in safety check only
 
 
class SafetyException(Exception):
    def __init__(self, result: SafetyResult):
        self.result = result
        super().__init__(f"Safety violation: {result.category} at {result.blocked_at}")
 
 
# ---------------------------------------------------------------------------
# Main filter
# ---------------------------------------------------------------------------
 
class SafetyFilter:
    """Unified safety gate for text (input + output) and images (output)."""
 
    def __init__(self, config: dict):
        self.cfg         = config["safety"]
        self.cfg.setdefault("image_enabled", True)
        self.cfg.setdefault("image_unsafe_labels", ["safe image", "nsfw content", "violent image", "hateful image"])
        self.cfg.setdefault("image_threshold", 0.25)
        self.log_path    = self.cfg.get("log_path", "logs/safety_log.jsonl")
        self._analyzer   = None
        self._classifier = None
        self._clip_model = None
        self._clip_prep  = None
        self._clip_tok   = None
        logger.info(
            "SafetyFilter init (pii=%s injection=%s toxicity=%s image=%s)",
            self.cfg["pii_enabled"], self.cfg["injection_enabled"],
            self.cfg["toxicity_enabled"], self.cfg["image_enabled"],
        )
 
    # ── Public API ──────────────────────────────────────────────────────────
 
    def check_input(self, text: str, request_id: str = "") -> SafetyResult:
        """Check user input BEFORE any model call. Returns SafetyResult with check_duration_ms."""
        t0 = time.perf_counter()
 
        if os.getenv("UNSAFE_MODE") == "true":
            return SafetyResult(safe=True, check_duration_ms=0)
 
        if self.cfg["injection_enabled"]:
            result = self._check_injection(text)
            if not result.safe:
                result.check_duration_ms = _ms(t0)
                self._log(request_id, text, result)
                return result

            result = self._check_harmful_topics(text)
            if not result.safe:
                result.check_duration_ms = _ms(t0)
                self._log(request_id, text, result)
                return result

        if self.cfg["toxicity_enabled"]:
            result = self._check_toxicity(text, blocked_at="input")
            if not result.safe:
                result.check_duration_ms = _ms(t0)
                self._log(request_id, text, result)
                return result
 
        return SafetyResult(safe=True, check_duration_ms=_ms(t0))
 
    def check_output(self, text: str, request_id: str = "") -> SafetyResult:
        """Check model text output BEFORE returning to caller."""
        t0 = time.perf_counter()
 
        if os.getenv("UNSAFE_MODE") == "true":
            return SafetyResult(safe=True, check_duration_ms=0)
 
        if self.cfg["toxicity_enabled"]:
            result = self._check_toxicity(text, blocked_at="output")
            if not result.safe:
                result.check_duration_ms = _ms(t0)
                self._log(request_id, text, result)
                return result
 
        if self.cfg["pii_enabled"]:
            result = self._check_pii(text)
            if not result.safe:
                result.check_duration_ms = _ms(t0)
                self._log(request_id, text, result)
                return result
 
        return SafetyResult(safe=True, check_duration_ms=_ms(t0))
 
    def check_image(self, image_path: str, request_id: str = "") -> SafetyResult:
        """Check generated image via CLIP zero-shot classification."""
        t0 = time.perf_counter()
 
        if not self.cfg["image_enabled"] or os.getenv("UNSAFE_MODE") == "true":
            return SafetyResult(safe=True, check_duration_ms=0)
 
        result = self._check_image_clip(image_path)
        result.check_duration_ms = _ms(t0)
        if not result.safe:
            self._log(request_id, image_path, result)
        return result
 
    @property
    def status(self) -> str:
        return "ok"
 
    # ── Internal checkers ────────────────────────────────────────────────────
 
    def _check_injection(self, text: str) -> SafetyResult:
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(text):
                return SafetyResult(
                    safe=False, category="prompt_injection", blocked_at="input",
                    detail=f"Matched: {pattern.pattern[:60]}",
                )
        return SafetyResult(safe=True)

    def _check_harmful_topics(self, text: str) -> SafetyResult:
        for pattern in _HARMFUL_PATTERNS:
            if pattern.search(text):
                return SafetyResult(
                    safe=False, category="harmful_content", blocked_at="input",
                    detail=f"Matched: {pattern.pattern[:60]}",
                )
        return SafetyResult(safe=True)

    def _check_toxicity(self, text: str, blocked_at: str) -> SafetyResult:
        if os.getenv("OPENAI_API_KEY"):
            result = self._openai_moderation(text, blocked_at)
            if result is not None:
                return result
 
        try:
            if not hasattr(self, "_toxic_pipe"):
                from transformers import pipeline as hf_pipeline
                self._toxic_pipe = hf_pipeline(
                    "text-classification",
                    model="unitary/toxic-bert",
                    truncation=True, max_length=512,
                )
            out = self._toxic_pipe(text)[0]
            if out["label"] == "toxic" and out["score"] > 0.7:
                return SafetyResult(
                    safe=False, category="toxicity", blocked_at=blocked_at,
                    detail=f"toxic-bert score={out['score']:.3f}",
                )
        except Exception as exc:
            logger.warning("Toxicity check failed: %s", exc)
 
        return SafetyResult(safe=True)
 
    def _openai_moderation(self, text: str, blocked_at: str):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp   = client.moderations.create(input=text)
            res    = resp.results[0]
            if res.flagged:
                cats = [k for k, v in res.categories.__dict__.items() if v]
                return SafetyResult(
                    safe=False, category=cats[0] if cats else "flagged",
                    blocked_at=blocked_at, detail=f"OpenAI flagged: {cats}",
                )
        except Exception as exc:
            logger.warning("OpenAI moderation unavailable: %s", exc)
        return None
 
    def _check_pii(self, text: str) -> SafetyResult:
        try:
            ignored_entities = set(self.cfg.get("pii_ignored_entities", []))
            score_threshold = float(self.cfg.get("pii_score_threshold", 0.7))

            results = self._get_pii_analyzer().analyze(text=text, language="en")
            # Filter out ignored entity types and low-confidence detections
            serious = [
                r for r in results
                if r.entity_type not in ignored_entities
                and r.score >= score_threshold
            ]
            if serious:
                types = [f"{r.entity_type}({r.score:.2f})" for r in serious]
                return SafetyResult(
                    safe=False, category="pii_leakage", blocked_at="output",
                    detail=f"PII detected: {types}",
                )
        except Exception as exc:
            logger.warning("PII check failed: %s", exc)
        return SafetyResult(safe=True)
 
    def _check_image_clip(self, image_path: str) -> SafetyResult:
        if not Path(image_path).exists():
            logger.error("CLIP check skipped — file not found: %s", image_path)
            return SafetyResult(safe=True)
 
        try:
            import torch
            from PIL import Image as PILImage
 
            model, prep, tok = self._get_clip()
            labels    = self.cfg["image_unsafe_labels"]
            threshold = float(self.cfg.get("image_threshold", 0.25))
 
            img    = prep(PILImage.open(image_path).convert("RGB")).unsqueeze(0)
            tokens = tok(labels)
 
            with torch.no_grad():
                i_feat = model.encode_image(img)
                t_feat = model.encode_text(tokens)
                i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
                t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
                scores = (i_feat @ t_feat.T).squeeze(0).tolist()
 
            for label, score in zip(labels[1:], scores[1:]):
                if score > threshold:
                    return SafetyResult(
                        safe=False,
                        category=label.replace(" ", "_"),
                        blocked_at="image_output",
                        detail=f"CLIP score={score:.3f} > threshold={threshold}",
                    )
        except Exception as exc:
            logger.warning("Image CLIP check failed: %s", exc)
 
        return SafetyResult(safe=True)
 
    # ── Lazy loaders ─────────────────────────────────────────────────────────
 
    def _get_pii_analyzer(self):
        if self._analyzer is None:
            from presidio_analyzer import AnalyzerEngine
            self._analyzer = AnalyzerEngine()
        return self._analyzer
 
    def _get_clip(self):
        if self._clip_model is None:
            import open_clip
            model, _, prep = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            model.eval()
            self._clip_model = model
            self._clip_prep  = prep
            self._clip_tok   = open_clip.get_tokenizer("ViT-B-32")
        return self._clip_model, self._clip_prep, self._clip_tok
 
    # ── Logging ──────────────────────────────────────────────────────────────
 
    def _log(self, request_id: str, content: str, result: SafetyResult) -> None:
        record = {
            "ts":                utc_now_iso(),
            "request_id":        request_id,
            "category":          result.category,
            "blocked_at":        result.blocked_at,
            "detail":            result.detail,
            "check_duration_ms": result.check_duration_ms,
            "content_hash":      hashlib.sha256(content.encode()).hexdigest()[:16],
        }
        append_jsonl(self.log_path, record)
        logger.warning(
            "Safety block | id=%s cat=%s at=%s duration=%dms",
            request_id, result.category, result.blocked_at, result.check_duration_ms,
        )
 
 
# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
 
def _ms(t0: float) -> int:
    """Elapsed milliseconds since t0."""
    return int((time.perf_counter() - t0) * 1000)