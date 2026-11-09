"""
Baseline evaluation – no fine-tuning.
Loads the base Qwen2.5-1.5B-Instruct model and evaluates it on the test set.
"""

import json
import logging
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow running as script from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
RESULTS_DIR = "results/baseline"
CHECKPOINT_MARKER = os.path.join(RESULTS_DIR, ".done")
SEED = 42
NUM_TEST = 200
DRIVE_RESULTS_DIR = "/content/drive/MyDrive/peft_experiment/results/baseline"


def load_test_data():
    """Load the dataset and return the 200-example test split."""
    test_file = "data/test/test.json"
    if os.path.exists(test_file):
        with open(test_file, "r", encoding="utf-8") as f:
            raw_test = json.load(f)
        return [{
            "customer_message": ex.get("user_message") or ex.get("customer_message") or ex.get("query", ""),
            "tool": ex.get("tool_name") or ex.get("tool", "no_tool"),
            "arguments": ex.get("arguments", {}),
        } for ex in raw_test]

    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    ds = ds.shuffle(seed=SEED).select(range(2000))
    test_ds = ds.select(range(1800, 2000))

    test_examples = []
    for row in test_ds:
        # Parse the expected output
        answer_text = row.get("answers", "")
        try:
            answers = json.loads(answer_text) if isinstance(answer_text, str) else answer_text
        except (json.JSONDecodeError, TypeError):
            answers = []

        # Map to our tool schema
        if isinstance(answers, list) and len(answers) > 0:
            first = answers[0]
            tool_name = first.get("name", "no_tool")
            tool_args = first.get("arguments", {})
        elif isinstance(answers, dict):
            tool_name = answers.get("name", "no_tool")
            tool_args = answers.get("arguments", {})
        else:
            tool_name = "no_tool"
            tool_args = {}

        # Map tool names to our canonical set
        tool_name = _map_tool_name(tool_name)

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except (json.JSONDecodeError, TypeError):
                tool_args = {}

        test_examples.append({
            "customer_message": row.get("query", ""),
            "tool": tool_name,
            "arguments": tool_args,
        })
    return test_examples


def _map_tool_name(name: str) -> str:
    """Map dataset tool names to our canonical 5 tool names."""
    name_lower = name.lower().strip()
    canonical = {
        "check_order_status": "check_order_status",
        "initiate_refund": "initiate_refund",
        "track_shipment": "track_shipment",
        "escalate_to_human": "escalate_to_human",
        "no_tool": "no_tool",
    }
    return canonical.get(name_lower, name)  # keep original if not in map


def main():
    print("=" * 60)
    print("  STAGE: Baseline Evaluation (No Fine-Tuning)")
    print("=" * 60)

    # ---- Checkpoint: skip if already done ----
    if os.path.exists(CHECKPOINT_MARKER):
        print(f"  ✓ Already completed (marker: {CHECKPOINT_MARKER}). Skipping.")
        with open(os.path.join(RESULTS_DIR, "metrics.json"), "r") as f:
            return json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Load test data ----
    print("\n  Loading test data …")
    test_data = load_test_data()
    print(f"  Loaded {len(test_data)} test examples")

    # ---- Load model ----
    print(f"\n  Loading model: {MODEL_ID}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Model loaded on {device}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Reset memory stats ----
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---- Evaluate ----
    t0 = time.perf_counter()
    metrics = evaluate_model(model, tokenizer, test_data, "baseline", device, RESULTS_DIR)
    t1 = time.perf_counter()

    # Record additional info
    metrics["total_params"] = sum(p.numel() for p in model.parameters())
    metrics["trainable_params"] = 0  # No training
    metrics["training_time_seconds"] = 0.0
    metrics["evaluation_time_seconds"] = round(t1 - t0, 2)
    if torch.cuda.is_available():
        metrics["peak_gpu_memory_mb"] = round(
            torch.cuda.max_memory_allocated() / 1024 / 1024, 2
        )
    else:
        metrics["peak_gpu_memory_mb"] = 0.0

    # Save updated metrics
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Completion marker ----
    with open(CHECKPOINT_MARKER, "w") as f:
        f.write("done")

    # ---- Copy to Google Drive (Colab) ----
    try:
        os.makedirs(DRIVE_RESULTS_DIR, exist_ok=True)
        import shutil
        for fname in os.listdir(RESULTS_DIR):
            src = os.path.join(RESULTS_DIR, fname)
            dst = os.path.join(DRIVE_RESULTS_DIR, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        print(f"  Results backed up to Google Drive: {DRIVE_RESULTS_DIR}")
    except Exception as e:
        print(f"  ⚠ Could not back up to Drive: {e}")

    print(f"\n  Baseline evaluation complete in {t1 - t0:.1f}s")
    return metrics


if __name__ == "__main__":
    main()
