"""
Rank sweep script for Section 3.
Takes the winning fine-tuning method from the 5-way comparison and trains it at:
r = 4, r = 8, r = 16, r = 32.
Evaluates each on the 200 held-out test examples, produces rank_sweep_results.csv,
and generates publication-quality plots.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import evaluate_model
from src.inference import SYSTEM_PROMPT, build_training_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
RANKS = [4, 8, 16, 32]
EPOCHS = 3
LR = 2e-4
WARMUP_RATIO = 0.03
BATCH_SIZE = 4
GRAD_ACCUM = 4
SEQ_LEN = 2048
SEED = 42
DRIVE_BASE = "/content/drive/MyDrive/peft_experiment"


def load_data():
    """Load and split the dataset (1800 train / 200 test)."""
    train_file = "data/train/train.json"
    test_file = "data/test/test.json"
    if os.path.exists(train_file) and os.path.exists(test_file):
        with open(train_file, "r", encoding="utf-8") as f:
            raw_train = json.load(f)
        with open(test_file, "r", encoding="utf-8") as f:
            raw_test = json.load(f)
        train_rows = [{
            "customer_message": ex.get("user_message") or ex.get("customer_message") or ex.get("query", ""),
            "tool": ex.get("tool_name") or ex.get("tool", "no_tool"),
            "arguments": ex.get("arguments", {}),
        } for ex in raw_train]
        test_rows = [{
            "customer_message": ex.get("user_message") or ex.get("customer_message") or ex.get("query", ""),
            "tool": ex.get("tool_name") or ex.get("tool", "no_tool"),
            "arguments": ex.get("arguments", {}),
        } for ex in raw_test]
        return train_rows, test_rows

    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    ds = ds.shuffle(seed=SEED).select(range(2000))
    train_ds = ds.select(range(1800))
    test_ds = ds.select(range(1800, 2000))
    train_rows = [_parse_example(row) for row in train_ds]
    test_rows = [_parse_example(row) for row in test_ds]
    return train_rows, test_rows


def _parse_example(row):
    """Parse a dataset row into customer_message, tool, arguments."""
    if "customer_message" in row and "tool" in row:
        return row
    answer_text = row.get("answers", "")
    try:
        answers = json.loads(answer_text) if isinstance(answer_text, str) else answer_text
    except (json.JSONDecodeError, TypeError):
        answers = []

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

    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except (json.JSONDecodeError, TypeError):
            tool_args = {}

    return {
        "customer_message": row.get("query", ""),
        "tool": tool_name,
        "arguments": tool_args,
    }


def prepare_train_dataset(train_ds):
    texts = []
    for parsed in train_ds:
        json_output = json.dumps({
            "tool": parsed["tool"],
            "arguments": parsed["arguments"],
        })
        text = build_training_text(parsed["customer_message"], json_output)
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def train_single_rank(r: int, method: str = "lora"):
    """Train and evaluate the selected method at rank r."""
    adapter_dir = f"adapters/rank_sweep/r_{r}"
    results_dir = f"results/rank_sweep/r_{r}"
    marker = os.path.join(results_dir, ".done")

    os.makedirs(adapter_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if os.path.exists(marker):
        print(f"  ✓ Rank r={r} already completed. Loading results.")
        with open(os.path.join(results_dir, "metrics.json")) as f:
            return json.load(f)

    # Check if r=8 can be copied from existing run
    if r == 8 and method.lower() == "lora" and os.path.exists("results/lora/.done"):
        print("  Reusing verified r=8 LoRA result from Stage 4...")
        with open("results/lora/metrics.json") as f:
            m = json.load(f)
        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump(m, f, indent=2)
        with open(marker, "w") as f:
            f.write("done\n")
        return m

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"  RANK SWEEP: {method.upper()} at r = {r}")
    print(f"{'=' * 60}")

    train_rows, test_rows = load_data()
    train_dataset = prepare_train_dataset(train_rows)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=r * 2,  # maintain 2x alpha ratio
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    trainable_params, all_params = model.get_nb_trainable_parameters()
    pct = 100 * trainable_params / all_params
    print(f"  Trainable params for r={r}: {trainable_params:,} ({pct:.4f}%)")

    training_args = TrainingArguments(
        output_dir=f"training_output/rank_sweep/r_{r}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        fp16=(torch_dtype == torch.float16),
        bf16=(torch_dtype == torch.bfloat16),
        logging_steps=10,
        save_strategy="epoch",
        seed=SEED,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=SEQ_LEN,
        tokenizer=tokenizer,
        args=training_args,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    trainer.train()
    elapsed_seconds = time.perf_counter() - t0
    elapsed_minutes = elapsed_seconds / 60.0

    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_data=test_rows,
        model_name=f"{method.upper()} (r={r})",
        device=device,
        save_dir=results_dir,
    )

    metrics.update({
        "method": method,
        "rank": r,
        "trainable_params": trainable_params,
        "total_params": all_params,
        "trainable_pct": round(pct, 4),
        "peak_gpu_allocated_gb": round(peak_allocated, 3),
        "peak_gpu_reserved_gb": round(peak_reserved, 3),
        "training_time_seconds": round(elapsed_seconds, 2),
        "training_time_minutes": round(elapsed_minutes, 2),
    })

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(marker, "w") as f:
        f.write("done\n")

    return metrics


def run_rank_sweep(method: str = "lora"):
    """Execute rank sweep across all ranks and generate tables & plots."""
    sweep_results = []
    for r in RANKS:
        res = train_single_rank(r, method=method)
        sweep_results.append({
            "Rank": r,
            "Method": method.upper(),
            "Trainable Parameters": res.get("trainable_params", 0),
            "% of Base Model": f"{res.get('trainable_pct', 0):.4f}%",
            "Peak GPU Memory (GB)": res.get("peak_gpu_allocated_gb", 0.0),
            "Training Time (Min)": res.get("training_time_minutes", 0.0),
            "Valid JSON / 200": f"{res.get('valid_json_count', 0)}/200 ({res.get('json_validity_pct', 0):.1f}%)",
            "Tool Accuracy / 200": f"{res.get('tool_accuracy_count', 0)}/200 ({res.get('tool_accuracy_pct', 0):.1f}%)",
            "Full Match / 200": f"{res.get('full_match_count', 0)}/200 ({res.get('full_match_pct', 0):.1f}%)",
            "Tool Accuracy (%)": res.get("tool_accuracy_pct", 0.0),
            "Full Match (%)": res.get("full_match_pct", 0.0),
        })

    df = pd.DataFrame(sweep_results)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    csv_path = "results/rank_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Rank sweep table saved to {csv_path}")

    # Generate Publication-Quality Plots
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Tool Accuracy vs Rank
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ranks = [row["Rank"] for row in sweep_results]
    accs = [row["Tool Accuracy (%)"] for row in sweep_results]

    ax.plot(ranks, accs, marker="o", markersize=8, linewidth=2.5, color="#1f77b4", label="Tool Accuracy")
    for r, acc in zip(ranks, accs):
        ax.annotate(f"{acc:.1f}%", (r, acc), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold")

    ax.set_title("Tool Accuracy vs LoRA Rank", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Adapter Rank (r)", fontsize=12)
    ax.set_ylabel("Tool Accuracy (%)", fontsize=12)
    ax.set_xticks(ranks)
    ax.set_ylim([max(0, min(accs) - 10), 105])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plot1_path = "results/plots/tool_accuracy_vs_rank.png"
    plt.savefig(plot1_path)
    plt.savefig("reports/figures/tool_accuracy_vs_rank.png")
    plt.close()
    print(f"  ✓ Plot saved: {plot1_path}")

    # Plot 2: Full-Match Accuracy vs Rank
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    full_matches = [row["Full Match (%)"] for row in sweep_results]

    ax.plot(ranks, full_matches, marker="s", markersize=8, linewidth=2.5, color="#2ca02c", label="Full-Match Accuracy")
    for r, fm in zip(ranks, full_matches):
        ax.annotate(f"{fm:.1f}%", (r, fm), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold")

    ax.set_title("Full-Match Accuracy vs LoRA Rank", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Adapter Rank (r)", fontsize=12)
    ax.set_ylabel("Full-Match Accuracy (%)", fontsize=12)
    ax.set_xticks(ranks)
    ax.set_ylim([max(0, min(full_matches) - 10), 105])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plot2_path = "results/plots/full_match_accuracy_vs_rank.png"
    plt.savefig(plot2_path)
    plt.savefig("reports/figures/full_match_accuracy_vs_rank.png")
    plt.close()
    print(f"  ✓ Plot saved: {plot2_path}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="lora", help="Method to rank sweep (lora, qlora, dora)")
    args = parser.parse_args()
    run_rank_sweep(args.method)


if __name__ == "__main__":
    main()

