"""
Model Loader — PEFT Experiment (23K-0068)
Load Qwen2.5-1.5B-Instruct in various configurations for PEFT training.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
    """Load and configure the tokenizer with proper padding."""
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    # Ensure pad token is set (Qwen uses eos as pad by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token = eos_token")

    logger.info(f"Vocab size: {tokenizer.vocab_size}, pad_token: '{tokenizer.pad_token}'")
    return tokenizer


def load_base_model(
    model_name: str = MODEL_NAME,
    precision: str = "bf16",
    device_map: str = "auto",
) -> AutoModelForCausalLM:
    """
    Load the base model in full/half precision (for LoRA, DoRA, AdaLoRA).

    Args:
        model_name: HuggingFace model identifier.
        precision: 'bf16', 'fp16', or 'fp32'.
        device_map: Device placement strategy.
    """
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if precision not in dtype_map:
        raise ValueError(f"Unsupported precision '{precision}'. Choose from: {list(dtype_map.keys())}")

    logger.info(f"Loading base model: {model_name} (precision={precision})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_map[precision],
        device_map=device_map,
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    param_info = get_parameter_count(model)
    logger.info(f"Model loaded — {param_info['total_params_M']:.1f}M params, {param_info['model_size_MB']:.0f} MB")
    return model


def load_quantized_model(
    model_name: str = MODEL_NAME,
    device_map: str = "auto",
) -> AutoModelForCausalLM:
    """
    Load the model with 4-bit NF4 quantization + double quantization (for QLoRA).
    """
    logger.info(f"Loading quantized model: {model_name} (NF4 + double quant)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    param_info = get_parameter_count(model)
    logger.info(f"Quantized model loaded — {param_info['total_params_M']:.1f}M params")
    return model


def get_parameter_count(model) -> dict:
    """
    Count total, trainable, and frozen parameters.

    Returns:
        Dict with param counts and percentages.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    # Estimate model size in MB (rough: fp16 = 2 bytes, fp32 = 4 bytes)
    param_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )

    info = {
        "total_params": total,
        "total_params_M": total / 1e6,
        "trainable_params": trainable,
        "trainable_params_M": trainable / 1e6,
        "frozen_params": frozen,
        "trainable_pct": (trainable / total * 100) if total > 0 else 0,
        "model_size_MB": param_bytes / (1024 ** 2),
    }
    return info


def print_parameter_summary(model, method_name: str = "") -> dict:
    """Print a formatted parameter summary table."""
    info = get_parameter_count(model)
    header = f"Parameter Summary" + (f" — {method_name}" if method_name else "")
    print(f"\n{'=' * 50}")
    print(f"  {header}")
    print(f"{'=' * 50}")
    print(f"  Total params:     {info['total_params']:>14,}")
    print(f"  Trainable params: {info['trainable_params']:>14,}")
    print(f"  Frozen params:    {info['frozen_params']:>14,}")
    print(f"  Trainable %:      {info['trainable_pct']:>13.4f}%")
    print(f"  Model size:       {info['model_size_MB']:>11.1f} MB")
    print(f"{'=' * 50}\n")
    return info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tokenizer = load_tokenizer()
    model = load_base_model(precision="bf16")
    print_parameter_summary(model, "Base Qwen2.5-1.5B-Instruct")
