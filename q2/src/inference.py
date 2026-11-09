"""
Inference engine for the PEFT experiment.
Runs model inference on test examples, records latency, and returns
structured prediction dicts.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.metrics import (
    categorize_error,
    full_match,
    json_validity,
    parse_json_output,
    tool_accuracy,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (shared across all scripts)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an e-commerce customer support assistant. When a customer request "
    "requires action, respond with a JSON object containing \"tool\" and "
    "\"arguments\". Valid tools: check_order_status, initiate_refund, "
    "track_shipment, escalate_to_human. If no tool is needed, respond with "
    '{\"tool\": \"no_tool\", \"arguments\": {}}. Respond ONLY with the JSON '
    "object, no other text."
)


def build_prompt(customer_message: str) -> str:
    """Construct the ChatML-style prompt for inference."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{customer_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_training_text(customer_message: str, json_output: str) -> str:
    """Construct the full ChatML conversation for training (with answer)."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{customer_message}<|im_end|>\n"
        f"<|im_start|>assistant\n{json_output}<|im_end|>"
    )


def _warm_up(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    n: int = 3,
) -> None:
    """Run *n* dummy forward passes so CUDA is warmed up before timing."""
    dummy_prompt = build_prompt("Hello, this is a warm-up message.")
    inputs = tokenizer(
        dummy_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)
    for _ in range(n):
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=None,
                top_p=None,
            )


@torch.no_grad()
def run_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_examples: List[Dict[str, Any]],
    device: torch.device,
    max_new_tokens: int = 256,
) -> List[Dict[str, Any]]:
    """Run inference on *test_examples* and return structured predictions.

    Each element of *test_examples* must have:
        - customer_message (str)
        - tool (str): gold tool name
        - arguments (dict): gold arguments

    Returns a list of prediction dicts with all required fields.
    """
    model.eval()

    # Warm-up
    logger.info("Warming up with 3 dummy inferences …")
    _warm_up(model, tokenizer, device)

    predictions: List[Dict[str, Any]] = []
    total = len(test_examples)

    for idx, example in enumerate(test_examples):
        if (idx + 1) % 25 == 0 or idx == 0 or (idx + 1) == total:
            logger.info(f"Inference  {idx + 1}/{total}")
            print(f"  Inference  {idx + 1}/{total} …")

        customer_message = example["customer_message"]
        gold_tool = example["tool"]
        gold_args = example.get("arguments", {})
        if isinstance(gold_args, str):
            try:
                gold_args = json.loads(gold_args)
            except (json.JSONDecodeError, TypeError):
                gold_args = {}

        gold_json = {"tool": gold_tool, "arguments": gold_args}

        prompt = build_prompt(customer_message)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        # ----- timed generation -----
        t0 = time.perf_counter()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        t1 = time.perf_counter()
        latency = t1 - t0

        # Decode only the generated tokens (strip the prompt)
        generated_ids = output_ids[0, prompt_len:]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Parse and evaluate
        is_valid_json = json_validity(raw_output)
        parsed = parse_json_output(raw_output)

        predicted_tool: Optional[str] = None
        predicted_args: Optional[Dict] = None
        if parsed is not None and isinstance(parsed, dict):
            predicted_tool = parsed.get("tool")
            predicted_args = parsed.get("arguments", {})

        is_tool_correct = tool_accuracy(predicted_tool, gold_tool)
        is_full_match = full_match(parsed, gold_json)
        error_cat = categorize_error(raw_output, parsed, gold_json)

        predictions.append(
            {
                "example_id": idx,
                "customer_message": customer_message,
                "gold_tool": gold_tool,
                "gold_args": json.dumps(gold_args) if isinstance(gold_args, dict) else str(gold_args),
                "raw_output": raw_output,
                "parsed_output": json.dumps(parsed) if parsed is not None else "",
                "json_valid": is_valid_json,
                "predicted_tool": predicted_tool or "",
                "predicted_args": json.dumps(predicted_args) if predicted_args is not None else "",
                "tool_correct": is_tool_correct,
                "full_match": is_full_match,
                "error_category": error_cat,
                "latency_seconds": round(latency, 4),
            }
        )

    avg_latency = (
        sum(p["latency_seconds"] for p in predictions) / len(predictions)
        if predictions
        else 0.0
    )
    logger.info(f"Inference complete. Avg latency: {avg_latency:.4f}s")
    print(f"\n  Average inference latency: {avg_latency:.4f} s")

    return predictions
