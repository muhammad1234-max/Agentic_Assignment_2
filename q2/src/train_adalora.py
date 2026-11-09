"""
AdaLoRA (Adaptive Low-Rank Adaptation) fine-tuning script.
Dynamically allocates rank budget across attention and MLP projections based on
singular value importance scores: init_r=12, target_r=8.
Trains for 3 epochs, records peak memory and time, evaluates on test set.
"""

import json
import logging
import os
import shutil
import sys
import time

import torch
from datasets import Dataset, load_dataset
from peft import AdaLoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
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
ADAPTER_DIR = "adapters/adalora"
RESULTS_DIR = "results/adalora"
CHECKPOINT_MARKER = os.path.join(RESULTS_DIR, ".done")
TRAINING_OUTPUT_DIR = "training_output/adalora"
SEED = 42
DRIVE_BASE = "/content/drive/MyDrive/peft_experiment"

# AdaLoRA hyperparameters
INIT_R = 12
TARGET_R = 8
ADALORA_ALPHA = 16
ADALORA_DROPOUT = 0.05
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
    """Convert dataset rows into formatted training texts."""
    texts = []
    for parsed in train_ds:
        json_output = json.dumps({
            "tool": parsed["tool"],
            "arguments": parsed["arguments"],
        })
        text = build_training_text(parsed["customer_message"], json_output)
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def backup_to_drive(local_dir: str, drive_subdir: str):
    """Copy local directory contents to Google Drive if available."""
    drive_dir = os.path.join(DRIVE_BASE, drive_subdir)
    try:
        os.makedirs(drive_dir, exist_ok=True)
        for fname in os.listdir(local_dir):
            src = os.path.join(local_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(drive_dir, fname))
        print(f"  Backed up to Drive: {drive_dir}")
    except Exception as e:
        print(f"  Note: Drive backup not active or skipped ({e})")


class AdaLoraRankAllocatorCallback(TrainerCallback):
    """Callback to step AdaLoRA's rank allocator during training."""
    def __init__(self, model):
        self.model = model

    def on_step_end(self, args, state, control, **kwargs):
        if hasattr(self.model, "update_and_allocate"):
            self.model.update_and_allocate(state.global_step)


def inspect_adalora_ranks(model):
    """Inspect allocated ranks across layers and modules."""
    rank_distribution = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_E") and hasattr(module, "ranknum"):
            try:
                active_r = int((module.lora_E > 0).sum().item()) if hasattr(module.lora_E, "sum") else int(module.ranknum.item())
                rank_distribution[name] = active_r
            except Exception:
                pass
    return rank_distribution


def main():
    print("=" * 60)
    print("  STAGE: AdaLoRA Training (init_r=12, target_r=8)")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    # Checkpoint check
    if os.path.exists(CHECKPOINT_MARKER):
        print(f"  ✓ Found checkpoint marker: {CHECKPOINT_MARKER}")
        print("  AdaLoRA training and evaluation already complete. Skipping.")
        metrics_file = os.path.join(RESULTS_DIR, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                return json.load(f)
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # 1. Load data
    train_rows, test_rows = load_data()
    train_dataset = prepare_train_dataset(train_rows)
    print(f"  Loaded {len(train_rows)} train, {len(test_rows)} test examples.")

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Model Loading
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    print(f"  Loading base model with dtype: {torch_dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # Calculate total steps for 1800 examples, batch=4, grad_accum=4 -> approx 336 steps
    total_steps = (len(train_rows) // (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS
    tinit = max(10, int(total_steps * 0.15))
    tfinal = max(tinit + 10, int(total_steps * 0.75))
    delta_t = 10

    # 4. Apply AdaLoRA Config
    peft_config = AdaLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        init_r=INIT_R,
        target_r=TARGET_R,
        lora_alpha=ADALORA_ALPHA,
        lora_dropout=ADALORA_DROPOUT,
        target_modules=TARGET_MODULES,
        tinit=tinit,
        tfinal=tfinal,
        deltaT=delta_t,
        beta1=0.85,
        beta2=0.85,
        orth_reg_weight=0.5,
    )
    model = get_peft_model(model, peft_config)

    trainable_params, all_params = model.get_nb_trainable_parameters()
    pct = 100 * trainable_params / all_params
    print(f"  Trainable parameters : {trainable_params:,} / {all_params:,} ({pct:.4f}%)")
    print(f"  AdaLoRA schedule: tinit={tinit}, tfinal={tfinal}, deltaT={delta_t}")

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=TRAINING_OUTPUT_DIR,
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
        callbacks=[AdaLoraRankAllocatorCallback(model)],
    )

    # 6. Execute training & time it
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("  Starting AdaLoRA training...")
    t0 = time.perf_counter()
    trainer.train()
    elapsed_seconds = time.perf_counter() - t0
    elapsed_minutes = elapsed_seconds / 60.0

    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0

    print(f"  Training finished in {elapsed_minutes:.2f} min ({elapsed_seconds:.1f} s)")
    print(f"  Peak GPU allocated: {peak_allocated:.3f} GB, reserved: {peak_reserved:.3f} GB")

    # 7. Check allocated ranks
    rank_dist = inspect_adalora_ranks(model)
    if rank_dist:
        print(f"  Sample allocated module ranks (first 5): {dict(list(rank_dist.items())[:5])}")

    # 8. Save adapter
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"  Adapter saved to: {ADAPTER_DIR}")

    # 9. Evaluate on test set
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_data=test_rows,
        model_name="AdaLoRA (init=12, target=8)",
        device=device,
        save_dir=RESULTS_DIR,
    )

    # Append training-specific stats
    metrics.update({
        "method": "AdaLoRA",
        "init_r": INIT_R,
        "target_r": TARGET_R,
        "trainable_params": trainable_params,
        "total_params": all_params,
        "trainable_pct": round(pct, 4),
        "peak_gpu_allocated_gb": round(peak_allocated, 3),
        "peak_gpu_reserved_gb": round(peak_reserved, 3),
        "training_time_seconds": round(elapsed_seconds, 2),
        "training_time_minutes": round(elapsed_minutes, 2),
        "rank_schedule": {"tinit": tinit, "tfinal": tfinal, "deltaT": delta_t},
    })

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Write completion marker
    with open(CHECKPOINT_MARKER, "w") as f:
        f.write("done\n")

    backup_to_drive(ADAPTER_DIR, "adapters/adalora")
    backup_to_drive(RESULTS_DIR, "results/adalora")

    return metrics


if __name__ == "__main__":
    main()

