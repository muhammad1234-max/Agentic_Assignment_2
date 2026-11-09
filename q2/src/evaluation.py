"""
Evaluation orchestrator.
Calls inference, computes all metrics, saves predictions CSV and metrics JSON.
"""

import csv
import json
import logging
import os
from typing import Any, Dict, List

from src.inference import run_inference
from src.metrics import compute_all_metrics, confusion_matrix, per_tool_metrics

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    tokenizer,
    test_data: List[Dict[str, Any]],
    model_name: str,
    device,
    save_dir: str,
) -> Dict[str, Any]:
    """Run full evaluation pipeline for a single model/adapter.

    1. Runs inference (with progress printing).
    2. Computes aggregate + per-tool metrics.
    3. Saves predictions.csv and metrics.json to *save_dir*.
    4. Returns the combined metrics dict.
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION: {model_name}")
    print(f"{'=' * 60}")
    print(f"  Test examples: {len(test_data)}")

    # ---- Inference ----
    predictions = run_inference(model, tokenizer, test_data, device)

    # ---- Aggregate metrics ----
    agg = compute_all_metrics(predictions)
    per_tool = per_tool_metrics(predictions)
    cm, labels = confusion_matrix(predictions)

    avg_latency = (
        sum(p["latency_seconds"] for p in predictions) / len(predictions)
        if predictions
        else 0.0
    )

    metrics = {
        "model_name": model_name,
        **agg,
        "avg_latency_seconds": round(avg_latency, 4),
        "per_tool": per_tool,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": labels,
    }

    # ---- Save predictions CSV ----
    csv_path = os.path.join(save_dir, "predictions.csv")
    fieldnames = [
        "example_id",
        "customer_message",
        "gold_tool",
        "gold_args",
        "raw_output",
        "parsed_output",
        "json_valid",
        "predicted_tool",
        "predicted_args",
        "tool_correct",
        "full_match",
        "error_category",
        "latency_seconds",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predictions)
    print(f"  Predictions saved → {csv_path}")

    # ---- Save metrics JSON ----
    json_path = os.path.join(save_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved    → {json_path}")

    # ---- Print summary ----
    print(f"\n  ┌──────────────────────────────────────┐")
    print(f"  │  JSON validity : {agg['json_validity_pct']:6.2f}%            │")
    print(f"  │  Tool accuracy : {agg['tool_accuracy_pct']:6.2f}%            │")
    print(f"  │  Full match    : {agg['full_match_pct']:6.2f}%            │")
    print(f"  │  Avg latency   : {avg_latency:.4f} s            │")
    print(f"  └──────────────────────────────────────┘")

    return metrics
