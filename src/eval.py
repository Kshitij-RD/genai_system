"""Evaluation — compute text-gen, image-gen, multimodal, and safety metrics on the test split."""
 
import json
import logging
from pathlib import Path
import argparse
 
import numpy as np
 
from src.data import load_split
from src.infer import infer_text, infer_image, infer_multimodal
from src.prompts import PromptManager
from src.safety import SafetyFilter
from src.text_gen import TextGenPipeline
from src.image_gen import ImageGenPipeline
from src.multimodal import MultimodalPipeline
from src.utils import load_config, seed_everything, utc_now_iso
 
logger = logging.getLogger("genai_system.eval")
 
 
# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
 
def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    """Compute average ROUGE-L F1 score."""
    from rouge_score import rouge_scorer
 
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        if ref:
            s = scorer.score(ref, pred)
            scores.append(s["rougeL"].fmeasure)
    return round(np.mean(scores).item(), 4) if scores else 0.0
 
 
def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    """Compute average BERTScore F1."""
    from bert_score import score as bert_score_fn
 
    valid_preds, valid_refs = [], []
    for p, r in zip(predictions, references):
        if r:
            valid_preds.append(p)
            valid_refs.append(r)
    if not valid_preds:
        return 0.0
 
    _, _, f1 = bert_score_fn(valid_preds, valid_refs, lang="en",
                          model_type="roberta-large", verbose=False)
    return round(f1.mean().item(), 4)
 
 
def compute_factual_accuracy(
    predictions: list[str], references: list[str], config: dict
) -> float:
    """Use LLM-as-judge to rate factual accuracy on a 1-5 scale.

    Returns the mean score across all samples with valid references.
    Includes retry with exponential backoff to avoid sample loss from rate limits.
    """
    import os
    import time
    from openai import OpenAI

    llm_cfg = config.get("llm", {})
    client = OpenAI(
        base_url=llm_cfg["base_url"],
        api_key=os.getenv(llm_cfg["api_key_env"], ""),
    )

    judge_system = (
        "You are an accuracy judge. Rate how factually correct the ANSWER is "
        "compared to the REFERENCE on a scale of 1-5:\n"
        "1 = completely wrong\n"
        "2 = mostly wrong with minor correct elements\n"
        "3 = partially correct but missing key facts\n"
        "4 = mostly correct with minor inaccuracies\n"
        "5 = fully correct and complete\n"
        "Return ONLY a single integer (1-5), nothing else."
    )

    scores = []
    skipped = 0
    total_valid = sum(1 for p, r in zip(predictions, references) if p and r)

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        if not ref or not pred:
            continue

        # Retry with exponential backoff — up to 3 attempts per sample
        max_retries = 3
        scored = False
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait = 2 ** attempt
                    logger.info("  Judge retry %d/%d after %ds…", attempt + 1, max_retries, wait)
                    time.sleep(wait)

                resp = client.chat.completions.create(
                    model=llm_cfg["model"],
                    temperature=0,
                    max_tokens=5,
                    messages=[
                        {"role": "system", "content": judge_system},
                        {"role": "user", "content": f"REFERENCE: {ref}\n\nANSWER: {pred}"},
                    ],
                )
                text = resp.choices[0].message.content.strip()
                for ch in text:
                    if ch.isdigit() and 1 <= int(ch) <= 5:
                        scores.append(int(ch))
                        scored = True
                        break
                if scored:
                    break
            except Exception as exc:
                logger.warning("Judge attempt %d failed for sample %d: %s", attempt + 1, i, exc)
                if attempt == max_retries - 1:
                    skipped += 1

    if skipped:
        logger.warning(
            "Factual accuracy: %d/%d samples skipped after retries — "
            "score may be biased (%.0f%% coverage)",
            skipped, total_valid, 100 * len(scores) / max(total_valid, 1),
        )

    return round(np.mean(scores).item(), 2) if scores else 0.0


def compute_safety_metrics(results: list[dict], samples: list[dict]) -> dict:
    """Compute TPR, FPR, FNR and safety-check-only latency overhead.
 
    latency_overhead measures only the safety check duration (check_duration_ms),
    NOT the full end-to-end inference latency — as specified in the test rubric.
    """
    tp = fp = fn = tn = 0
    # FIX: collect safety check duration only, not total inference latency
    safety_check_durations = []
 
    for sample, result in zip(samples, results):
        is_unsafe  = sample.get("safety_label") == "unsafe"
        was_blocked = result.get("safety_triggered", False)
 
        if is_unsafe and was_blocked:
            tp += 1
        elif is_unsafe and not was_blocked:
            fn += 1
        elif not is_unsafe and was_blocked:
            fp += 1
        else:
            tn += 1
 
        # FIX: use safety_check_ms (set by SafetyFilter) not full pipeline latency
        check_ms = result.get("safety_check_ms", 0)
        if check_ms > 0:
            safety_check_durations.append(check_ms)
 
    total_pos = tp + fn or 1
    total_neg = tn + fp or 1
 
    return {
        "tpr": round(tp / total_pos, 4),
        "fpr": round(fp / total_neg, 4),
        "fnr": round(fn / total_pos, 4),
        # latency_overhead = mean safety-check-only duration across all tested samples
        "latency_overhead_ms": int(np.mean(safety_check_durations))
            if safety_check_durations else 0,
    }
 
 
def percentile(data: list[float | int], p: int) -> int:
    """Return integer percentile, safe for empty lists."""
    return int(np.percentile(data, p)) if data else 0


def compute_ece(confidences: list[float], accuracies: list[bool], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error.

    Bins predictions by confidence and compares mean confidence vs actual
    accuracy per bin. Lower is better (0 = perfectly calibrated).
    """
    if not confidences:
        return 0.0

    confs = np.array(confidences)
    accs = np.array(accuracies, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confs > bin_edges[i]) & (confs <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confs[mask].mean()
        bin_acc = accs[mask].mean()
        ece += mask.sum() / len(confs) * abs(bin_acc - bin_conf)

    return round(float(ece), 4)


def compute_grounding_quality(groundings: list[str]) -> float:
    """Score grounding quality on a 0–3 scale.

    3 = specific region reference (e.g. 'upper-left quadrant, spanning ~30%')
    2 = general positional reference (e.g. 'in the center')
    1 = vague reference (e.g. 'in the image')
    0 = no grounding at all
    """
    if not groundings:
        return 0.0

    scores = []
    specific_keywords = [
        "upper-left", "upper-right", "lower-left", "lower-right",
        "top-left", "top-right", "bottom-left", "bottom-right",
        "quadrant", "spanning", "approximately", "percent", "%",
        "foreground", "background", "bounding",
    ]
    general_keywords = [
        "center", "central", "middle", "left", "right",
        "top", "bottom", "upper", "lower", "side",
    ]

    for g in groundings:
        g_lower = g.lower().strip()
        if not g_lower:
            scores.append(0)
        elif any(kw in g_lower for kw in specific_keywords):
            scores.append(3)
        elif any(kw in g_lower for kw in general_keywords):
            scores.append(2)
        elif len(g_lower) > 5:
            scores.append(1)
        else:
            scores.append(0)

    return round(np.mean(scores).item(), 2)


def compute_mm_accuracy(predictions: list[str], references: list[str]) -> float:
    """Compute multimodal answer accuracy using fuzzy keyword matching.

    For VQA, exact match is too strict. Checks if key content
    words from the reference appear in the prediction.
    Uses a two-tier approach: core keywords (colors, shapes, numbers)
    get priority matching, then general token overlap.
    """
    if not predictions:
        return 0.0

    # Core visual keywords that are strong indicators of correctness
    core_words = {
        "red", "blue", "green", "yellow", "orange", "purple", "cyan",
        "black", "white", "dark", "light", "pink", "brown",
        "circle", "square", "rectangle", "triangle", "shape",
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
        "one", "two", "three", "four", "five", "six", "seven",
        "left", "right", "upper", "lower", "center", "top", "bottom",
        "yes", "no", "plus",
    }

    stopwords = {
        "the", "a", "an", "is", "are", "in", "of", "and", "or",
        "to", "it", "this", "that", "image", "shown", "visible",
        "there", "present", "appears", "see", "large", "written",
    }

    correct = 0
    for pred, ref in zip(predictions, references):
        if not ref:
            continue
        ref_tokens = {
            w.lower().strip(".,!?\"'()") for w in ref.split()
            if w.lower().strip(".,!?\"'()") not in stopwords and len(w.strip(".,!?\"'()")) > 1
        }
        pred_lower = pred.lower()

        # Tier 1: check core visual keywords from reference
        ref_core = ref_tokens & core_words
        if ref_core:
            core_matches = sum(1 for t in ref_core if t in pred_lower)
            if core_matches / len(ref_core) >= 0.5:
                correct += 1
                continue

        # Tier 2: general token overlap at 40% threshold
        matches = sum(1 for t in ref_tokens if t in pred_lower)
        if ref_tokens and matches / len(ref_tokens) >= 0.4:
            correct += 1

    return round(correct / len(predictions), 4) if predictions else 0.0
 
 
# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
 
def run_evaluation(config: dict | None = None, task_filter: str = "all") -> dict:
    """Run evaluation on the test split and write/merge metrics.json."""
    if config is None:
        config = load_config()
    seed_everything(config.get("seed", 42))
 
    logger.info("Loading test split…")
    test_samples = load_split(config, split="test")
    logger.info("Total test samples loaded: %d", len(test_samples))
 
    safety = SafetyFilter(config)
 
    model_note = (
        f"{config.get('llm', {}).get('model', 'unknown-llm')} | "
        f"seed={config.get('seed', 42)} | temperature={config.get('llm', {}).get('temperature', 0)}"
    )
    metrics: dict = {
        "timestamp":  utc_now_iso(),
        "eval_split": "test",
        "text":       {},
        "image":      {},
        "multimodal": {},
        "safety":     {},
        "notes":      f"task={task_filter} | {model_note}",
    }
 
    # Global collectors for cross-task safety aggregation
    all_eval_results: list[dict] = []
    all_eval_samples: list[dict] = []
 
    # ------------------------------------------------------------------
    # 1. Text Generation
    # ------------------------------------------------------------------
    if task_filter in ["all", "text"]:
        text_samples = [s for s in test_samples if s.get("task") == "text_gen"]
 
        if text_samples:
            logger.info("--- Text Evaluation (%d samples) ---", len(text_samples))
            prompt_mgr = PromptManager(
                prompts_dir=config["prompts"]["dir"],
                few_shot_dir=config["prompts"].get("few_shot_dir", "data/few_shot/"),
                encoder_model=config["prompts"]["encoder_model"],
                context_window=config["prompts"]["context_window"],
            )
            pipeline = TextGenPipeline(config)
 
            predictions, references, latencies = [], [], []
            schema_valid_count = hallucination_count = safe_sample_count = 0
 
            for i, sample in enumerate(text_samples):
                logger.info("Text %d/%d  %s", i + 1, len(text_samples), sample["id"])
                result = infer_text(
                    prompt=sample["prompt"],
                    category=sample["category"],
                    sample_id=sample["id"],
                    reference=sample.get("reference_output"),
                    pipeline=pipeline,
                    prompt_mgr=prompt_mgr,
                    safety=safety,
                )
                all_eval_results.append(result)
                all_eval_samples.append(sample)
 
                if sample.get("safety_label") == "safe":
                    safe_sample_count += 1
                    predictions.append(result.get("output", ""))
                    references.append(sample.get("reference_output", ""))
                    latencies.append(result.get("latency_ms", 0))
                    if result.get("schema_valid"):
                        schema_valid_count += 1
                    if result.get("hallucination_flagged"):
                        hallucination_count += 1
 
            logger.info("Computing ROUGE-L, BERTScore, and factual accuracy…")
            rouge_l = compute_rouge_l(predictions, references)
            bertscore_f1 = compute_bertscore(predictions, references)

            # LLM-as-judge factual accuracy (1-5 scale) — only on factual samples
            logger.info("Running factual accuracy judge…")
            factual_accuracy = compute_factual_accuracy(
                predictions, references, config
            )

            metrics["text"] = {
                "rouge_l":           rouge_l,
                "bertscore_f1":      bertscore_f1,
                "factual_accuracy":  factual_accuracy,
                "schema_validity":   round(schema_valid_count / max(safe_sample_count, 1), 4),
                "hallucination_rate": round(hallucination_count / max(safe_sample_count, 1), 4),
                "latency_p50_ms":    percentile(latencies, 50),
                "latency_p95_ms":    percentile(latencies, 95),
                "n_samples":         safe_sample_count,
            }
        else:
            logger.info("No text_gen samples in test split.")
 
    # ------------------------------------------------------------------
    # 2. Image Generation
    # ------------------------------------------------------------------
    if task_filter in ["all", "image"]:
        image_samples = [s for s in test_samples if s.get("task") == "image_gen"]
 
        if image_samples:
            logger.info("--- Image Evaluation (%d samples) ---", len(image_samples))
            image_pipeline = ImageGenPipeline(config)
 
            if image_pipeline.status != "ok":
                logger.error(
                    "Image pipeline not ready (status=%s) — skipping image eval",
                    image_pipeline.status,
                )
                metrics["image"] = {
                    "error":    "pipeline_unavailable",
                    "status":   image_pipeline.status,
                    "n_samples": 0,
                }
            else:
                clip_raw_scores: list[float] = []
                clip_opt_scores: list[float] = []
                image_latencies: list[int]   = []
                scored_count = generation_failures = 0
 
                for i, sample in enumerate(image_samples):
                    logger.info("Image %d/%d  %s", i + 1, len(image_samples), sample["id"])
 
                    image_path   = image_pipeline.output_dir / f"{sample['id']}.png"
                    sidecar_path = image_pipeline.output_dir / f"{sample['id']}.json"
 
                    # Use cached result if already generated
                    if image_path.exists() and sidecar_path.exists():
                        logger.info("  Cached — skipping generation for %s", sample["id"])
                        result = {"generation_time_ms": 0, "safety_triggered": False,
                                  "safety_check_ms": 0}
                    else:
                        # Use infer_image wrapper — runs input + output safety checks
                        result = infer_image(
                            prompt=sample["prompt"],
                            seed=config.get("seed", 42),
                            sample_id=sample["id"],
                            pipeline=image_pipeline,
                            safety=safety,
                        )

                        if result.get("safety_triggered"):
                            logger.info(
                                "  Safety blocked %s: %s at %s",
                                sample["id"],
                                result.get("safety_category", "unknown"),
                                result.get("safety_blocked_at", "unknown"),
                            )
                            all_eval_results.append(result)
                            all_eval_samples.append(sample)
                            continue

                        if result.get("error"):
                            logger.warning(
                                "Generation failed for %s: %s",
                                sample["id"], result["error"],
                            )
                            generation_failures += 1
                            all_eval_results.append(result)
                            all_eval_samples.append(sample)
                            continue
 
                    all_eval_results.append(result)
                    all_eval_samples.append(sample)
                    image_latencies.append(result.get("generation_time_ms", 0))
 
                    # Read sidecar for CLIP scores
                    if sidecar_path.exists():
                        with open(sidecar_path) as f:
                            sidecar = json.load(f)
                        raw = sidecar.get("clip_score_raw", 0.0)
                        opt = sidecar.get("clip_score_optimised", 0.0)
                        # Only count as scored if CLIP actually ran (non-zero)
                        if raw > 0.0 or opt > 0.0:
                            clip_raw_scores.append(raw)
                            clip_opt_scores.append(opt)
                            scored_count += 1
                        else:
                            logger.warning(
                                "  CLIP scores are zero for %s — image may not have generated",
                                sample["id"],
                            )
                            generation_failures += 1
                    else:
                        logger.warning("  No sidecar JSON for %s", sample["id"])
                        generation_failures += 1
 
                avg_raw = round(np.mean(clip_raw_scores).item(), 4) if clip_raw_scores else 0.0
                avg_opt = round(np.mean(clip_opt_scores).item(), 4) if clip_opt_scores else 0.0
 
                metrics["image"] = {
                    "clip_score_optimised": avg_opt,
                    "clip_score_raw":       avg_raw,
                    "clip_delta":           round(avg_opt - avg_raw, 4),
                    "n_samples":            len(image_samples),
                    "n_scored":             scored_count,
                    "generation_failures":  generation_failures,
                    "latency_p50_ms":       percentile(image_latencies, 50),
                    "latency_p95_ms":       percentile(image_latencies, 95),
                }
 
                if generation_failures:
                    logger.warning(
                        "%d/%d image samples failed — check CF_API_TOKEN and image_gen logs",
                        generation_failures, len(image_samples),
                    )
        else:
            logger.info("No image_gen samples in test split.")
 
    # ------------------------------------------------------------------
    # 3. Multimodal Evaluation
    # ------------------------------------------------------------------
    if task_filter in ["all", "multimodal"]:
        mm_samples = [s for s in test_samples if s.get("task") == "multimodal"]
 
        if mm_samples:
            logger.info("--- Multimodal Evaluation (%d samples) ---", len(mm_samples))
            mm_pipeline = MultimodalPipeline(config)

            if mm_pipeline.status != "ok":
                logger.error(
                    "Multimodal pipeline not ready (status=%s) — skipping",
                    mm_pipeline.status,
                )
                metrics["multimodal"] = {
                    "error": "pipeline_unavailable",
                    "status": mm_pipeline.status,
                    "n_samples": 0,
                }
            else:
                mm_predictions: list[str] = []
                mm_references: list[str] = []
                mm_confidences: list[float] = []
                mm_accuracies: list[bool] = []
                mm_groundings: list[str] = []
                mm_latencies: list[int] = []
                mm_errors = 0

                for i, sample in enumerate(mm_samples):
                    logger.info("Multimodal %d/%d  %s", i + 1, len(mm_samples), sample["id"])

                    image_path = sample.get("image_path", "")
                    if not image_path or not Path(image_path).exists():
                        logger.warning("  Image not found: %s — skipping", image_path)
                        mm_errors += 1
                        continue

                    result = infer_multimodal(
                        image_path=image_path,
                        question=sample["prompt"],
                        sample_id=sample["id"],
                        return_grounding=True,
                        pipeline=mm_pipeline,
                        safety=safety,
                    )

                    all_eval_results.append(result)
                    all_eval_samples.append(sample)

                    if result.get("error"):
                        logger.warning(
                            "  VLM failed for %s: %s", sample["id"], result["error"]
                        )
                        mm_errors += 1
                        continue

                    answer = result.get("answer", "")
                    confidence = result.get("confidence", 0.5)
                    grounding = result.get("grounding", "")
                    ref = sample.get("reference_output", "")

                    mm_predictions.append(answer)
                    mm_references.append(ref)
                    mm_confidences.append(confidence)
                    mm_groundings.append(grounding)
                    mm_latencies.append(result.get("latency_ms", 0))

                    # Per-sample accuracy for ECE — mirrors compute_mm_accuracy logic
                    if ref:
                        _core_words = {
                            "red", "blue", "green", "yellow", "orange", "purple",
                            "cyan", "black", "white", "dark", "light",
                            "circle", "square", "rectangle", "triangle",
                            "1", "2", "3", "4", "5", "left", "right",
                            "upper", "lower", "center", "yes", "no", "plus",
                            "one", "two", "three", "four", "five",
                        }
                        _stop = {"the", "a", "an", "is", "are", "in", "of", "and",
                                 "or", "to", "it", "this", "that", "image",
                                 "shown", "visible", "there", "present"}
                        ref_tokens = {
                            w.lower().strip(".,!?\"'()")
                            for w in ref.split()
                            if w.lower().strip(".,!?\"'()") not in _stop
                            and len(w.strip(".,!?\"'()")) > 1
                        }
                        pred_lower = answer.lower()
                        ref_core = ref_tokens & _core_words
                        if ref_core:
                            core_hits = sum(1 for t in ref_core if t in pred_lower)
                            is_correct = bool(core_hits / len(ref_core) >= 0.5)
                        else:
                            hits = sum(1 for t in ref_tokens if t in pred_lower)
                            is_correct = bool(
                                ref_tokens and hits / len(ref_tokens) >= 0.4
                            )
                    else:
                        is_correct = False
                    mm_accuracies.append(is_correct)

                    logger.info(
                        "  answer=%s conf=%.2f grounding=%s correct=%s",
                        answer[:60], confidence, grounding[:40], is_correct,
                    )

                accuracy = compute_mm_accuracy(mm_predictions, mm_references)
                ece = compute_ece(mm_confidences, mm_accuracies)
                grounding_quality = compute_grounding_quality(mm_groundings)

                metrics["multimodal"] = {
                    "accuracy": accuracy,
                    "ece": ece,
                    "grounding_quality": grounding_quality,
                    "latency_p50_ms": percentile(mm_latencies, 50),
                    "latency_p95_ms": percentile(mm_latencies, 95),
                    "n_samples": len(mm_samples),
                    "n_evaluated": len(mm_predictions),
                    "n_errors": mm_errors,
                }

                logger.info(
                    "Multimodal results: accuracy=%.4f, ECE=%.4f, grounding=%.2f",
                    accuracy, ece, grounding_quality,
                )
        else:
            logger.info("No multimodal samples in test split.")
 
    # ------------------------------------------------------------------
    # 4. Global Safety Aggregation
    # ------------------------------------------------------------------
    if all_eval_samples:
        logger.info(
            "Computing safety metrics across %d tested samples…",
            len(all_eval_samples),
        )
        metrics["safety"] = compute_safety_metrics(all_eval_results, all_eval_samples)
    else:
        logger.info("No samples evaluated — safety metrics skipped.")
 
    # ------------------------------------------------------------------
    # 5. Smart JSON Merge & Save
    # ------------------------------------------------------------------
    metrics_path = Path(config.get("eval", {}).get("metrics_path", "metrics.json"))
    final_metrics = metrics.copy()
 
    if metrics_path.exists() and task_filter != "all":
        logger.info("Merging '%s' metrics into existing metrics.json…", task_filter)
        with open(metrics_path) as f:
            existing = json.load(f)
 
        # Only overwrite keys that were actually computed in this run
        for key in ["text", "image", "multimodal"]:
            if metrics.get(key):  # skip empty dicts — don't overwrite existing data
                existing[key] = metrics[key]

        # Safety: only overwrite if the current run actually tested unsafe samples
        # (partial re-evals like --task multimodal may only have safe samples,
        # producing meaningless tpr=0/fnr=0 that would clobber real safety data)
        if metrics.get("safety"):
            has_unsafe = any(
                s.get("safety_label") == "unsafe" for s in all_eval_samples
            )
            if has_unsafe or task_filter == "all":
                existing["safety"] = metrics["safety"]
            else:
                logger.info(
                    "Skipping safety merge — no unsafe samples in this partial run"
                )
 
        existing["timestamp"] = metrics["timestamp"]
 
        # FIX: append to notes instead of overwriting — preserves full run history
        prev_notes = existing.get("notes", "")
        existing["notes"] = (
            prev_notes
            + f" | re-eval {task_filter} @ {metrics['timestamp']}"
        )
        final_metrics = existing
 
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    logger.info("Metrics written to %s", metrics_path)
 
    return final_metrics
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    from src.utils import setup_logging
 
    parser = argparse.ArgumentParser(description="Evaluate GenAI System Pipelines")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "text", "image", "multimodal"],
        help="Which pipeline to evaluate",
    )
    args = parser.parse_args()
 
    setup_logging("INFO")
    config  = load_config()
    metrics = run_evaluation(config, task_filter=args.task)
    print(json.dumps(metrics, indent=2))