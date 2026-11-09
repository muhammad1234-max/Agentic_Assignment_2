"""
Evaluation metrics for the PEFT experiment.
Provides JSON validity checking, tool accuracy, full-match accuracy,
per-tool precision/recall/F1, and confusion matrix generation.
"""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical tool list
# ---------------------------------------------------------------------------
VALID_TOOLS = [
    "check_order_status",
    "initiate_refund",
    "track_shipment",
    "escalate_to_human",
    "no_tool",
]


# ===================================================================
# Core metrics
# ===================================================================

def json_validity(raw_output: str) -> bool:
    """Return True if *raw_output* is valid JSON (via json.loads)."""
    try:
        json.loads(raw_output.strip())
        return True
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def parse_json_output(raw_output: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse raw model output into a dict.

    Tries the full string first, then looks for the first ``{...}`` block.
    Returns None on failure.
    """
    text = raw_output.strip()
    # Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # Fallback: extract first JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return None


def tool_accuracy(predicted_tool: Optional[str], gold_tool: str) -> bool:
    """Exact string match between predicted tool name and gold tool name."""
    if predicted_tool is None:
        return False
    return predicted_tool.strip().lower() == gold_tool.strip().lower()


def full_match(
    predicted_json: Optional[Dict[str, Any]],
    gold_json: Dict[str, Any],
) -> bool:
    """Check if predicted JSON matches gold JSON (tool + all arguments).

    Key order is ignored; value comparison is case-insensitive for strings.
    """
    if predicted_json is None:
        return False
    # Tool must match
    pred_tool = str(predicted_json.get("tool", "")).strip().lower()
    gold_tool = str(gold_json.get("tool", "")).strip().lower()
    if pred_tool != gold_tool:
        return False
    # Arguments must match
    pred_args = predicted_json.get("arguments", {})
    gold_args = gold_json.get("arguments", {})
    if not isinstance(pred_args, dict) or not isinstance(gold_args, dict):
        return pred_args == gold_args
    if set(pred_args.keys()) != set(gold_args.keys()):
        return False
    for key in gold_args:
        pv = pred_args[key]
        gv = gold_args[key]
        # Case-insensitive comparison for strings
        if isinstance(pv, str) and isinstance(gv, str):
            if pv.strip().lower() != gv.strip().lower():
                return False
        else:
            if str(pv) != str(gv):
                return False
    return True


# ===================================================================
# Aggregate metrics
# ===================================================================

def compute_all_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute aggregate metrics over a list of prediction dicts.

    Each dict is expected to have keys:
        json_valid (bool), tool_correct (bool), full_match (bool)

    Returns dict with json_validity_pct, tool_accuracy_pct, full_match_pct.
    """
    n = len(predictions)
    if n == 0:
        return {
            "json_validity_pct": 0.0,
            "tool_accuracy_pct": 0.0,
            "full_match_pct": 0.0,
            "num_examples": 0,
        }
    jv = sum(1 for p in predictions if p.get("json_valid", False))
    ta = sum(1 for p in predictions if p.get("tool_correct", False))
    fm = sum(1 for p in predictions if p.get("full_match", False))
    return {
        "json_validity_pct": round(100.0 * jv / n, 2),
        "tool_accuracy_pct": round(100.0 * ta / n, 2),
        "full_match_pct": round(100.0 * fm / n, 2),
        "num_examples": n,
    }


def per_tool_metrics(
    predictions: List[Dict[str, Any]],
    tool_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, F1 per tool class.

    Uses ``gold_tool`` and ``predicted_tool`` fields in each prediction dict.
    """
    if tool_names is None:
        tool_names = VALID_TOOLS

    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)

    for pred in predictions:
        gold = str(pred.get("gold_tool", "")).strip().lower()
        predicted = str(pred.get("predicted_tool", "")).strip().lower()
        if predicted == gold:
            tp[gold] += 1
        else:
            fp[predicted] += 1
            fn[gold] += 1

    results: Dict[str, Dict[str, float]] = {}
    for tool in tool_names:
        t = tool.lower()
        precision = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0.0
        recall = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        results[tool] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp[t],
            "fp": fp[t],
            "fn": fn[t],
        }
    return results


def confusion_matrix(
    predictions: List[Dict[str, Any]],
    tool_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Build a confusion matrix for tool predictions.

    Returns (matrix, labels) where matrix[i][j] = count of gold=i predicted=j.
    """
    if tool_names is None:
        tool_names = VALID_TOOLS

    label_to_idx = {t.lower(): i for i, t in enumerate(tool_names)}
    n_labels = len(tool_names)
    matrix = np.zeros((n_labels, n_labels), dtype=int)

    for pred in predictions:
        gold = str(pred.get("gold_tool", "")).strip().lower()
        predicted = str(pred.get("predicted_tool", "")).strip().lower()
        gi = label_to_idx.get(gold, -1)
        pi = label_to_idx.get(predicted, -1)
        if gi >= 0 and pi >= 0:
            matrix[gi][pi] += 1
        elif gi >= 0:
            # predicted tool not in known labels – count as "other"
            pass  # ignored in matrix; captured in per_tool_metrics fp

    return matrix, tool_names


def categorize_error(
    raw_output: str,
    parsed: Optional[Dict[str, Any]],
    gold_json: Dict[str, Any],
) -> str:
    """Categorize a prediction error.

    Categories:
        correct       – prediction matches gold
        broken_json   – output is not valid JSON
        wrong_tool    – valid JSON but wrong tool selected
        made_up_argument – correct tool but wrong/extra/missing arguments
        refused_to_call – model refused / returned no_tool when tool was needed
        other         – anything else
    """
    if parsed is None:
        return "broken_json"

    gold_tool = str(gold_json.get("tool", "")).strip().lower()
    pred_tool = str(parsed.get("tool", "")).strip().lower()

    # Check for refusal
    if pred_tool == "no_tool" and gold_tool != "no_tool":
        return "refused_to_call"

    # Wrong tool
    if pred_tool != gold_tool:
        return "wrong_tool"

    # Correct tool – check arguments
    gold_args = gold_json.get("arguments", {})
    pred_args = parsed.get("arguments", {})
    if not isinstance(pred_args, dict):
        return "made_up_argument"
    if set(pred_args.keys()) != set(gold_args.keys()):
        return "made_up_argument"
    for key in gold_args:
        pv = pred_args.get(key)
        gv = gold_args[key]
        if isinstance(pv, str) and isinstance(gv, str):
            if pv.strip().lower() != gv.strip().lower():
                return "made_up_argument"
        else:
            if str(pv) != str(gv):
                return "made_up_argument"

    return "correct"
