"""
LoRA fine-tuning script.
Applies LoRA (r=8) to Qwen2.5-1.5B-Instruct, trains for 3 epochs,
evaluates on the test set, and saves adapter + results.
"""

import json
import logging
import os
import shutil
import sys
import time

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
ADAPTER_DIR = "adapters/lora"
RESULTS_DIR = "results/lora"
CHECKPOINT_MARKER = os.path.join(RESULTS_DIR, ".done")
TRAINING_OUTPUT_DIR = "training_output/lora"
SEED = 42
DRIVE_BASE = "/content/drive/MyDrive/peft_experiment"

# LoRA hyperparameters
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training hyperparameters
EPOCHS = 3
LR = 2e-4
WARMUP_RATIO = 0.03
BATCH_SIZE = 4
GRAD_ACCUM = 4
SEQ_LEN = 2048


def load_data():
    """Load and split the dataset (1800 train / 200 test).
    Checks data/train/train.json and data/test/test.json first,
    falls back to HuggingFace datasets if not yet preprocessed locally.
    """
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
    """Convert raw dataset rows into formatted training texts."""
    texts = []
    for row in train_ds:
        parsed = _parse_example(row)
        json_output = json.dumps({
            "tool": parsed["tool"],
            "arguments": parsed["arguments"],
        })
        text = build_training_text(parsed["customer_message"], json_output)
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def prepare_test_examples(test_ds):
    """Convert raw dataset rows into test example dicts."""
    return [_parse_example(row) for row in test_ds]


def backup_to_drive(local_dir: str, drive_subdir: str):
    """Copy local directory contents to Google Drive (best-effort)."""
    drive_dir = os.path.join(DRIVE_BASE, drive_subdir)
    try:
        os.makedirs(drive_dir, exist_ok=True)
        for fname in os.listdir(local_dir):
            src = os.path.join(local_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(drive_dir, fname))
        print(f"  Backed up to Drive: {drive_dir}")
    except Exception as e:
        print(f"  ⚠ Drive backup failed: {e}")


def main():
    print("=" * 60)
    print("  STAGE: LoRA Training (r=8)")
    print("=" * 60)

    # ---- Checkpoint: skip if already done ----
    if os.path.exists(CHECKPOINT_MARKER):
        print(f"  ✓ Already completed (marker: {CHECKPOINT_MARKER}). Skipping.")
        with open(os.path.join(RESULTS_DIR, "metrics.json"), "r") as f:
            return json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    # ---- Load data ----
    print("\n  Loading dataset …")
    train_ds_raw, test_ds_raw = load_data()
    train_dataset = prepare_train_dataset(train_ds_raw)
    test_examples = prepare_test_examples(test_ds_raw)
    print(f"  Train: {len(train_dataset)},  Test: {len(test_examples)}")

    # ---- Load base model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"\n  Loading base model: {MODEL_ID}  (dtype={compute_dtype})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # ---- Apply LoRA ----
    print("\n  Applying LoRA config …")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=TRAINING_OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=SEED,
        optim="adamw_torch",
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        max_seq_length=SEQ_LEN,
    )

    # ---- Train ----
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("\n  Starting training …")
    t0 = time.perf_counter()

    # Resume from checkpoint if available
    resume_ckpt = None
    if os.path.isdir(TRAINING_OUTPUT_DIR):
        ckpts = [
            d for d in os.listdir(TRAINING_OUTPUT_DIR)
            if d.startswith("checkpoint-")
        ]
        if ckpts:
            resume_ckpt = os.path.join(
                TRAINING_OUTPUT_DIR,
                sorted(ckpts, key=lambda x: int(x.split("-")[1]))[-1],
            )
            print(f"  Resuming from checkpoint: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)
    t1 = time.perf_counter()
    training_time = t1 - t0

    peak_mem = 0.0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"\n  Training complete in {training_time:.1f}s")
    print(f"  Peak GPU memory: {peak_mem:.1f} MB")

    # ---- Save adapter ----
    print(f"\n  Saving adapter → {ADAPTER_DIR}")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    backup_to_drive(ADAPTER_DIR, "adapters/lora")

    # ---- Evaluate ----
    metrics = evaluate_model(
        model, tokenizer, test_examples, "lora_r8", device, RESULTS_DIR
    )
    metrics["trainable_params"] = trainable_params
    metrics["total_params"] = total_params
    metrics["training_time_seconds"] = round(training_time, 2)
    metrics["peak_gpu_memory_mb"] = round(peak_mem, 2)

    # Save updated metrics
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Completion marker ----
    with open(CHECKPOINT_MARKER, "w") as f:
        f.write("done")

    backup_to_drive(RESULTS_DIR, "results/lora")
    print(f"\n  LoRA experiment complete.")
    return metrics


if __name__ == "__main__":
    main()
