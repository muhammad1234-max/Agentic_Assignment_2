"""
Failure Analysis Module for Section 4 (Question 6).
Extracts and categorizes errors from the best model's actual test predictions.
Selects exactly 5 illustrative failures (or all if fewer than 5) and generates
failure_analysis.csv and detailed diagnostic reports.
"""

import csv
import json
import logging
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

ERROR_CATEGORIES = [
    "wrong_tool",
    "made_up_argument",
    "broken_json",
    "refused_to_call",
    "other",
]


def load_predictions(predictions_csv_path: str) -> List[Dict]:
    """Load predictions from CSV file."""
    if not os.path.exists(predictions_csv_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_csv_path}")

    predictions = []
    with open(predictions_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append(dict(row))
    return predictions


def extract_failures(predictions: List[Dict]) -> List[Dict]:
    """Filter out all predictions that were not full matches."""
    failures = []
    for row in predictions:
        is_full_match = str(row.get("full_match", "")).lower() in ("true", "1")
        if not is_full_match:
            failures.append(row)
    return failures


def diagnose_failure(row: Dict) -> Dict:
    """Generate a precise diagnostic explanation for an incorrect prediction."""
    customer_msg = row.get("customer_message", "")
    gold_tool = row.get("gold_tool", "")
    gold_args = row.get("gold_args", "{}")
    raw_output = row.get("raw_output", "")
    pred_tool = row.get("predicted_tool", "")
    pred_args = row.get("predicted_args", "{}")
    json_valid = str(row.get("json_valid", "")).lower() in ("true", "1")
    tool_correct = str(row.get("tool_correct", "")).lower() in ("true", "1")

    # Determine error category & explanation
    if not json_valid:
        category = "broken_json"
        explanation = (
            "Model failed strict JSON parsing (json.loads). It likely included conversational "
            "prose, Markdown code fences, or unescaped characters outside raw JSON."
        )
    elif pred_tool == "no_tool" and gold_tool != "no_tool":
        category = "refused_to_call"
        explanation = (
            f"Customer request clearly warranted tool '{gold_tool}', but the model returned "
            "'no_tool', refusing or overlooking the operational action."
        )
    elif not tool_correct:
        category = "wrong_tool"
        explanation = (
            f"Model hallucinated or misclassified the action intent. Selected '{pred_tool}' "
            f"instead of expected tool '{gold_tool}'."
        )
    else:
        # Tool is correct, but arguments mismatched
        try:
            g_dict = json.loads(gold_args) if isinstance(gold_args, str) else gold_args
            p_dict = json.loads(pred_args) if isinstance(pred_args, str) else pred_args
        except Exception:
            g_dict, p_dict = {}, {}

        missing_keys = set(g_dict.keys()) - set(p_dict.keys())
        extra_keys = set(p_dict.keys()) - set(g_dict.keys())

        if extra_keys:
            category = "made_up_argument"
            explanation = (
                f"Model invented extraneous argument keys {list(extra_keys)} not allowed by schema."
            )
        elif missing_keys:
            category = "made_up_argument"
            explanation = (
                f"Model omitted required schema keys {list(missing_keys)}."
            )
        else:
            mismatched_vals = {k: (g_dict[k], p_dict[k]) for k in g_dict if g_dict[k] != p_dict.get(k)}
            category = "made_up_argument"
            explanation = (
                f"Argument value discrepancy: {mismatched_vals}. Model may have altered order_id or reason string."
            )

    return {
        "example_id": row.get("example_id", ""),
        "customer_message": customer_msg,
        "gold_tool": gold_tool,
        "gold_arguments": gold_args,
        "raw_model_output": raw_output,
        "predicted_tool": pred_tool,
        "predicted_arguments": pred_args,
        "error_category": category,
        "explanation": explanation,
    }


def analyze_best_model_failures(
    predictions_csv_path: str = "results/lora/predictions.csv",
    output_csv_path: str = "results/failure_analysis.csv",
    output_json_path: str = "results/failures/failure_summary.json",
    num_selected: int = 5,
) -> List[Dict]:
    """Analyze failures, pick 5 diverse examples across categories, and save summary."""
    predictions = load_predictions(predictions_csv_path)
    failures = extract_failures(predictions)

    print(f"\n{'=' * 60}")
    print(f"  FAILURE ANALYSIS: Found {len(failures)} errors out of {len(predictions)} test examples")
    print(f"{'=' * 60}")

    diagnosed = [diagnose_failure(f) for f in failures]

    # Group by category for diverse selection
    by_category = {}
    for d in diagnosed:
        by_category.setdefault(d["error_category"], []).append(d)

    category_counts = {cat: len(by_category.get(cat, [])) for cat in ERROR_CATEGORIES}
    print("  Error distribution across categories:")
    for cat, count in category_counts.items():
        print(f"    - {cat:<20}: {count}")

    # Select up to num_selected diverse examples
    selected_failures = []
    # Pick 1 from each category first if available
    for cat in ERROR_CATEGORIES:
        if cat in by_category and len(by_category[cat]) > 0:
            selected_failures.append(by_category[cat][0])
            if len(selected_failures) >= num_selected:
                break

    # Fill remaining from any category
    for d in diagnosed:
        if len(selected_failures) >= num_selected:
            break
        if d not in selected_failures:
            selected_failures.append(d)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    fieldnames = [
        "example_id",
        "customer_message",
        "gold_tool",
        "gold_arguments",
        "raw_model_output",
        "predicted_tool",
        "predicted_arguments",
        "error_category",
        "explanation",
    ]
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_failures)

    # Save JSON summary
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    summary = {
        "total_test_examples": len(predictions),
        "total_failures": len(failures),
        "error_category_distribution": category_counts,
        "selected_failures_for_report": selected_failures,
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  ✓ Saved 5 illustrative failures to: {output_csv_path}")
    print(f"  ✓ Saved failure summary to: {output_json_path}\n")

    return selected_failures


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, default="results/lora/predictions.csv")
    args = parser.parse_args()
    analyze_best_model_failures(args.predictions)

