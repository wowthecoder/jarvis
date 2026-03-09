"""Quasi-exact match scoring for the GAIA benchmark."""
from typing import Optional
from ..utils.normalize import normalize_answer


def score_answer(predicted: Optional[str], ground_truth: Optional[str]) -> bool:
    """Return True if the predicted answer matches the ground truth.

    Uses GAIA's quasi-exact match: normalize both sides, then compare.
    """
    if predicted is None or ground_truth is None:
        return False
    return normalize_answer(str(predicted)) == normalize_answer(str(ground_truth))


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy metrics from a list of result dicts.

    Each dict must have: task_id, predicted, ground_truth, level, correct.
    Returns overall accuracy plus per-level breakdown.
    """
    if not results:
        return {"accuracy": 0.0, "correct": 0, "total": 0, "by_level": {}}

    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))

    by_level: dict[int, dict] = {}
    for r in results:
        lvl = r.get("level", 0)
        if lvl not in by_level:
            by_level[lvl] = {"correct": 0, "total": 0}
        by_level[lvl]["total"] += 1
        if r.get("correct"):
            by_level[lvl]["correct"] += 1

    level_accuracy = {
        lvl: {
            "accuracy": d["correct"] / d["total"] if d["total"] else 0.0,
            "correct": d["correct"],
            "total": d["total"],
        }
        for lvl, d in sorted(by_level.items())
    }

    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "by_level": level_accuracy,
    }
