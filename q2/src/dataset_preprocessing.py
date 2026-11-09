"""
Dataset Preprocessing — PEFT Experiment (23K-0068)
THE CORE PIPELINE: Parse Glaive conversations, map to our 4 e-commerce tools,
validate against schemas, sample 2000 examples, split train/test.
"""

import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Tool Mapping Configuration
# ─────────────────────────────────────────────────────────────

# Keywords used to map Glaive's arbitrary function names to our 4 permitted tools.
# We match against the function name AND its arguments/context.
TOOL_KEYWORD_MAP = {
    "check_order_status": {
        "function_name_patterns": [
            r"order.?status", r"check.?order", r"get.?order", r"lookup.?order",
            r"order.?info", r"order.?detail", r"order.?lookup", r"find.?order",
            r"view.?order", r"order.?check", r"retrieve.?order",
        ],
        "context_keywords": [
            "order status", "check order", "order information", "order details",
            "what is the status", "where is my order", "order number",
            "look up my order", "find my order", "order inquiry",
        ],
    },
    "initiate_refund": {
        "function_name_patterns": [
            r"refund", r"return", r"money.?back", r"cancel.?order",
            r"request.?refund", r"process.?refund", r"initiate.?refund",
            r"create.?refund", r"submit.?return", r"process.?return",
        ],
        "context_keywords": [
            "refund", "return", "money back", "cancel order", "want my money",
            "return the item", "return the product", "get a refund",
            "damaged", "defective", "wrong item", "not what i ordered",
        ],
    },
    "track_shipment": {
        "function_name_patterns": [
            r"track", r"ship", r"deliver", r"package.?status",
            r"shipping.?status", r"track.?package", r"track.?order",
            r"track.?shipment", r"delivery.?status", r"package.?location",
            r"transit", r"carrier", r"tracking.?number",
        ],
        "context_keywords": [
            "track", "shipping", "delivery", "shipment", "where is my package",
            "shipping status", "delivery status", "tracking number",
            "in transit", "out for delivery", "package location",
            "estimated delivery", "when will it arrive",
        ],
    },
    "escalate_to_human": {
        "function_name_patterns": [
            r"escalat", r"human", r"agent", r"transfer", r"complaint",
            r"support.?agent", r"live.?agent", r"customer.?service",
            r"speak.?to", r"talk.?to", r"connect.?to", r"supervisor",
        ],
        "context_keywords": [
            "escalate", "human agent", "speak to someone", "complaint",
            "supervisor", "manager", "not satisfied", "unresolved",
            "complex issue", "need help", "talk to a person",
            "transfer me", "real person", "live support",
        ],
    },
}

# Patterns in user messages that indicate no function call is needed
NO_TOOL_INDICATORS = [
    r"^(hi|hello|hey|good morning|good evening|thanks|thank you|bye|goodbye)",
    r"^what (do you|can you) (do|help)",
    r"^(who are you|what are you)",
    r"how are you",
    r"^(yes|no|ok|okay|sure|alright|fine)$",
]

# ─────────────────────────────────────────────────────────────
# Schema Loading & Validation
# ─────────────────────────────────────────────────────────────


def load_tool_schemas(schema_path: str = "configs/tool_schemas.json") -> dict:
    """Load tool schemas from JSON config."""
    with open(schema_path, "r") as f:
        data = json.load(f)
    schemas = {tool["name"]: tool for tool in data["tools"]}
    logger.info(f"Loaded schemas for {len(schemas)} tools")
    return schemas


def validate_against_schema(tool_name: str, arguments: dict, schemas: dict) -> tuple[bool, str]:
    """
    Validate a tool call's arguments against its schema.

    Returns:
        (is_valid, error_message)
    """
    if tool_name not in schemas:
        return False, f"Unknown tool: {tool_name}"

    schema = schemas[tool_name]

    # Check required arguments
    for arg_name, arg_spec in schema["required_arguments"].items():
        if arg_name not in arguments:
            return False, f"Missing required argument: {arg_name}"
        expected_type = arg_spec["type"]
        if expected_type == "string" and not isinstance(arguments[arg_name], str):
            return False, f"Argument '{arg_name}' must be string, got {type(arguments[arg_name]).__name__}"

    # Check no completely unknown arguments (allow optional ones)
    all_known = set(schema["required_arguments"].keys()) | set(schema["optional_arguments"].keys())
    for arg_name in arguments:
        if arg_name not in all_known:
            return False, f"Unknown argument: {arg_name}"

    return True, ""


# ─────────────────────────────────────────────────────────────
# Glaive Conversation Parsing
# ─────────────────────────────────────────────────────────────


def parse_glaive_conversation(chat_text: str) -> list[dict]:
    """
    Parse a Glaive 'chat' column into structured turns.

    Glaive format typically looks like:
      SYSTEM: You are a helpful assistant...
      USER: I want to check my order status for order #12345
      ASSISTANT: <functioncall> {"name": "get_order_status", "arguments": {"order_id": "12345"}}
      FUNCTION RESPONSE: {"status": "shipped"}
      ASSISTANT: Your order has been shipped!

    Returns list of turn dicts with keys: role, content, function_call (if present).
    """
    turns = []
    if not chat_text or not isinstance(chat_text, str):
        return turns

    # Split by role prefixes
    role_pattern = re.compile(
        r"(SYSTEM|USER|ASSISTANT|FUNCTION RESPONSE)\s*:\s*",
        re.IGNORECASE
    )
    parts = role_pattern.split(chat_text)

    # parts alternates: [preamble, role, content, role, content, ...]
    i = 1  # skip any preamble
    while i < len(parts) - 1:
        role = parts[i].strip().upper()
        content = parts[i + 1].strip()
        i += 2

        turn = {"role": role, "content": content}

        # Check for function calls in ASSISTANT turns
        if role == "ASSISTANT" and "<functioncall>" in content.lower():
            func_call = extract_function_call(content)
            if func_call:
                turn["function_call"] = func_call

        turns.append(turn)

    return turns


def extract_function_call(assistant_content: str) -> Optional[dict]:
    """
    Extract function name and arguments from an ASSISTANT turn containing <functioncall>.

    Handles formats like:
      <functioncall> {"name": "func_name", "arguments": {"key": "value"}}
      <functioncall> {"name": "func_name", "arguments": "{\"key\": \"value\"}"}
    """
    # Find the JSON blob after <functioncall>
    match = re.search(r"<functioncall>\s*(\{.*\})", assistant_content, re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    json_str = match.group(1).strip()

    # Try to parse the JSON
    try:
        call_data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try fixing common issues: trailing text, unescaped chars
        json_str = re.sub(r"\}\s*$", "}", json_str)
        # Find the balanced braces
        json_str = extract_balanced_json(json_str)
        if not json_str:
            return None
        try:
            call_data = json.loads(json_str)
        except json.JSONDecodeError:
            return None

    func_name = call_data.get("name", "")
    arguments = call_data.get("arguments", {})

    # Arguments are sometimes double-encoded as a JSON string
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            # Try to salvage key-value pairs
            arguments = parse_malformed_arguments(arguments)

    if not isinstance(arguments, dict):
        arguments = {}

    return {"name": func_name, "arguments": arguments}


def extract_balanced_json(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from text."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
    return None


def parse_malformed_arguments(arg_str: str) -> dict:
    """Best-effort parse of malformed argument strings."""
    result = {}
    # Try common patterns like key=value or "key": "value"
    pairs = re.findall(r'"?(\w+)"?\s*[:=]\s*"([^"]*)"', arg_str)
    for key, val in pairs:
        result[key] = val
    return result


# ─────────────────────────────────────────────────────────────
# Function Name → Tool Mapping
# ─────────────────────────────────────────────────────────────


def map_function_to_tool(func_name: str, arguments: dict, user_message: str = "") -> Optional[str]:
    """
    Map a Glaive function name to one of our 4 permitted tools.

    Strategy (in priority order):
      1. Direct regex match on the original function name
      2. Keyword match on the user's message context
      3. Argument-based heuristic (e.g., has 'order_id' + 'reason' → refund)
      4. Return None if unmappable
    """
    func_lower = func_name.lower().replace("-", "_").replace(" ", "_")
    user_lower = user_message.lower() if user_message else ""

    # --- Strategy 1: Match function name against known patterns ---
    scores = {}
    for tool_name, config in TOOL_KEYWORD_MAP.items():
        score = 0
        for pattern in config["function_name_patterns"]:
            if re.search(pattern, func_lower):
                score += 10  # strong signal from function name
                break
        scores[tool_name] = score

    # --- Strategy 2: Match user message context ---
    for tool_name, config in TOOL_KEYWORD_MAP.items():
        for keyword in config["context_keywords"]:
            if keyword.lower() in user_lower:
                scores[tool_name] = scores.get(tool_name, 0) + 2

    # --- Strategy 3: Argument-based heuristics ---
    arg_keys = set(k.lower() for k in arguments.keys())
    arg_vals = " ".join(str(v).lower() for v in arguments.values())

    if "reason" in arg_keys and ("order" in arg_vals or "order_id" in arg_keys):
        scores["initiate_refund"] = scores.get("initiate_refund", 0) + 5
    if "tracking" in arg_vals or "tracking_number" in arg_keys:
        scores["track_shipment"] = scores.get("track_shipment", 0) + 5
    if "issue_summary" in arg_keys or "complaint" in arg_keys or "issue" in arg_keys:
        scores["escalate_to_human"] = scores.get("escalate_to_human", 0) + 5
    if "order_id" in arg_keys and len(arg_keys) == 1:
        # Single order_id argument — could be status check or tracking
        # Differentiate by user context
        if scores.get("track_shipment", 0) > scores.get("check_order_status", 0):
            pass  # tracking already has a higher score
        else:
            scores["check_order_status"] = scores.get("check_order_status", 0) + 3

    # Pick the highest-scoring tool (must have score > 0)
    if not scores or max(scores.values()) == 0:
        return None

    best_tool = max(scores, key=scores.get)
    if scores[best_tool] > 0:
        return best_tool

    return None


def check_no_tool(chat_text: str, turns: list[dict]) -> bool:
    """
    Determine if a conversation has no function call (classify as no_tool).

    Returns True if:
      - No ASSISTANT turn contains a <functioncall> tag
      - OR user message matches no-tool indicators (greetings, general questions)
    """
    # Check for any function call in the conversation
    has_function_call = any(
        "<functioncall>" in turn.get("content", "").lower()
        for turn in turns
        if turn.get("role") == "ASSISTANT"
    )
    if not has_function_call:
        return True

    return False


# ─────────────────────────────────────────────────────────────
# Argument Normalization
# ─────────────────────────────────────────────────────────────


def normalize_arguments(tool_name: str, raw_args: dict) -> dict:
    """
    Normalize and clean arguments to match our schema expectations.

    - Standardize order_id format
    - Map argument name variants to canonical names
    - Clean string values
    """
    normalized = {}

    # Common argument name aliases
    ORDER_ID_ALIASES = {"order_id", "orderid", "order_number", "ordernumber",
                        "order_no", "orderno", "id", "order"}
    REASON_ALIASES = {"reason", "refund_reason", "return_reason", "cancel_reason", "description"}
    ISSUE_ALIASES = {"issue_summary", "issue", "summary", "complaint", "description", "problem", "message"}
    CUSTOMER_ID_ALIASES = {"customer_id", "customerid", "customer_number", "user_id", "userid"}

    for key, value in raw_args.items():
        key_lower = key.lower().strip()
        value = str(value).strip() if value is not None else ""

        if tool_name == "check_order_status":
            if key_lower in ORDER_ID_ALIASES:
                normalized["order_id"] = normalize_order_id(value)

        elif tool_name == "initiate_refund":
            if key_lower in ORDER_ID_ALIASES:
                normalized["order_id"] = normalize_order_id(value)
            elif key_lower in REASON_ALIASES:
                normalized["reason"] = value if value else "Customer requested refund"

        elif tool_name == "track_shipment":
            if key_lower in ORDER_ID_ALIASES:
                normalized["order_id"] = normalize_order_id(value)

        elif tool_name == "escalate_to_human":
            if key_lower in ISSUE_ALIASES:
                normalized["issue_summary"] = value if value else "Customer issue requires human attention"
            elif key_lower in CUSTOMER_ID_ALIASES:
                normalized["customer_id"] = value

    return normalized


def normalize_order_id(raw_id: str) -> str:
    """Standardize order ID to ORD-XXXXX format."""
    raw_id = raw_id.strip().upper()

    # Already in ORD-XXXXX format
    if re.match(r"^ORD-\w+$", raw_id):
        return raw_id

    # Extract digits
    digits = re.findall(r"\d+", raw_id)
    if digits:
        return f"ORD-{digits[0]}"

    # If alphanumeric, prefix with ORD-
    cleaned = re.sub(r"[^A-Z0-9]", "", raw_id)
    if cleaned:
        return f"ORD-{cleaned}"

    return f"ORD-{abs(hash(raw_id)) % 100000:05d}"


# ─────────────────────────────────────────────────────────────
# Synthesize Missing Arguments
# ─────────────────────────────────────────────────────────────


def synthesize_missing_args(tool_name: str, existing_args: dict, user_message: str) -> dict:
    """
    If mapping succeeded but arguments are incomplete, try to extract from user message.
    """
    args = dict(existing_args)

    if tool_name in ("check_order_status", "initiate_refund", "track_shipment"):
        if "order_id" not in args:
            # Try to extract order ID from user message
            order_match = re.search(
                r"(?:order\s*(?:#|number|no\.?|id)?[:.\s]*)([A-Z0-9-]+\d+)",
                user_message, re.IGNORECASE
            )
            if order_match:
                args["order_id"] = normalize_order_id(order_match.group(1))

    if tool_name == "initiate_refund" and "reason" not in args:
        # Use the user message as the reason if nothing else available
        if len(user_message) > 10:
            args["reason"] = user_message[:200].strip()

    if tool_name == "escalate_to_human" and "issue_summary" not in args:
        if len(user_message) > 5:
            args["issue_summary"] = user_message[:200].strip()

    return args


# ─────────────────────────────────────────────────────────────
# Main Processing Pipeline
# ─────────────────────────────────────────────────────────────


def get_first_user_message(turns: list[dict]) -> str:
    """Extract the first USER message from parsed turns."""
    for turn in turns:
        if turn.get("role") == "USER":
            return turn["content"]
    return ""


def process_single_example(chat_text: str, schemas: dict, idx: int) -> Optional[dict]:
    """
    Process a single Glaive conversation into a structured training example.

    Returns None if the example cannot be mapped to a valid tool.
    """
    turns = parse_glaive_conversation(chat_text)
    if not turns:
        return None

    user_message = get_first_user_message(turns)
    if not user_message:
        return None

    # Check if this is a no-tool conversation
    if check_no_tool(chat_text, turns):
        return {
            "id": f"glaive_{idx}",
            "user_message": user_message.strip(),
            "tool_name": "no_tool",
            "arguments": {},
            "original_function": None,
            "source_index": idx,
        }

    # Find the first function call
    func_call = None
    for turn in turns:
        if "function_call" in turn:
            func_call = turn["function_call"]
            break

    if not func_call:
        # Has functioncall tag but couldn't parse it
        return None

    original_func_name = func_call["name"]
    raw_arguments = func_call["arguments"]

    # Map to our permitted tools
    mapped_tool = map_function_to_tool(original_func_name, raw_arguments, user_message)
    if not mapped_tool:
        return None  # unmappable

    # Normalize arguments
    normalized_args = normalize_arguments(mapped_tool, raw_arguments)

    # Synthesize missing arguments from context
    normalized_args = synthesize_missing_args(mapped_tool, normalized_args, user_message)

    # Validate against schema
    is_valid, error = validate_against_schema(mapped_tool, normalized_args, schemas)
    if not is_valid:
        return None

    return {
        "id": f"glaive_{idx}",
        "user_message": user_message.strip(),
        "tool_name": mapped_tool,
        "arguments": normalized_args,
        "original_function": original_func_name,
        "source_index": idx,
    }


def process_dataset(
    raw_path: str = "data/raw/glaive_function_calling_v2.jsonl",
    schema_path: str = "configs/tool_schemas.json",
    total_samples: int = 2000,
    train_size: int = 1800,
    test_size: int = 200,
    seed: int = 42,
    train_dir: str = "data/train/",
    test_dir: str = "data/test/",
    manifest_path: str = "data/dataset_manifest.json",
) -> dict:
    """
    Full preprocessing pipeline.

    1. Load raw JSONL
    2. Parse & map each conversation
    3. Sample exactly `total_samples` valid examples
    4. Split into train/test
    5. Save to disk
    6. Create provenance manifest
    """
    import random
    random.seed(seed)

    schemas = load_tool_schemas(schema_path)

    # Load raw data
    logger.info(f"Loading raw data from {raw_path}")
    raw_examples = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_examples.append(json.loads(line))
    logger.info(f"Loaded {len(raw_examples)} raw examples")

    # Process all examples
    valid_examples = []
    unmapped_count = 0
    parse_error_count = 0
    unmapped_functions = Counter()

    for idx, example in enumerate(raw_examples):
        chat_text = example.get("chat", example.get("conversations", ""))

        # Handle list-format conversations (some HF datasets use this)
        if isinstance(chat_text, list):
            chat_text = "\n".join(
                f"{turn.get('role', 'USER').upper()}: {turn.get('content', '')}"
                for turn in chat_text
            )

        if not chat_text:
            parse_error_count += 1
            continue

        result = process_single_example(chat_text, schemas, idx)

        if result is None:
            unmapped_count += 1
            # Log which original function couldn't be mapped
            func_match = re.search(r'"name"\s*:\s*"([^"]+)"', chat_text)
            if func_match:
                unmapped_functions[func_match.group(1)] += 1
        else:
            valid_examples.append(result)

        if (idx + 1) % 20000 == 0:
            logger.info(
                f"  Processed {idx + 1}/{len(raw_examples)} — "
                f"valid: {len(valid_examples)}, unmapped: {unmapped_count}"
            )

    logger.info(f"Processing complete:")
    logger.info(f"  Total raw: {len(raw_examples)}")
    logger.info(f"  Valid mapped: {len(valid_examples)}")
    logger.info(f"  Unmapped/excluded: {unmapped_count}")
    logger.info(f"  Parse errors: {parse_error_count}")

    if len(valid_examples) < total_samples:
        logger.warning(
            f"Only {len(valid_examples)} valid examples available "
            f"(need {total_samples}). Using all of them."
        )
        total_samples = len(valid_examples)
        train_size = int(total_samples * 0.9)
        test_size = total_samples - train_size

    # Log top unmapped functions
    if unmapped_functions:
        logger.info("Top 15 unmapped original functions:")
        for func, count in unmapped_functions.most_common(15):
            logger.info(f"    {func}: {count}")

    # Tool distribution before sampling
    pre_sample_dist = Counter(ex["tool_name"] for ex in valid_examples)
    logger.info(f"Tool distribution (before sampling): {dict(pre_sample_dist)}")

    # Stratified sampling: maintain tool distribution proportions
    by_tool = {}
    for ex in valid_examples:
        by_tool.setdefault(ex["tool_name"], []).append(ex)

    # Calculate proportional allocation
    sampled = []
    remaining_quota = total_samples
    tool_quotas = {}

    for tool_name, examples in sorted(by_tool.items()):
        proportion = len(examples) / len(valid_examples)
        quota = max(1, int(proportion * total_samples))
        tool_quotas[tool_name] = min(quota, len(examples))
        remaining_quota -= tool_quotas[tool_name]

    # Distribute any remaining quota to the largest classes
    if remaining_quota > 0:
        for tool_name in sorted(by_tool.keys(), key=lambda t: len(by_tool[t]), reverse=True):
            can_add = len(by_tool[tool_name]) - tool_quotas[tool_name]
            add = min(remaining_quota, can_add)
            tool_quotas[tool_name] += add
            remaining_quota -= add
            if remaining_quota <= 0:
                break

    for tool_name, examples in by_tool.items():
        random.shuffle(examples)
        quota = tool_quotas.get(tool_name, 0)
        sampled.extend(examples[:quota])

    random.shuffle(sampled)
    sampled = sampled[:total_samples]

    logger.info(f"Sampled {len(sampled)} examples")
    post_sample_dist = Counter(ex["tool_name"] for ex in sampled)
    logger.info(f"Tool distribution (after sampling): {dict(post_sample_dist)}")

    # Split train/test with stratification
    train_data = []
    test_data = []

    by_tool_sampled = {}
    for ex in sampled:
        by_tool_sampled.setdefault(ex["tool_name"], []).append(ex)

    for tool_name, examples in by_tool_sampled.items():
        random.shuffle(examples)
        test_count = max(1, int(len(examples) * (test_size / total_samples)))
        test_data.extend(examples[:test_count])
        train_data.extend(examples[test_count:])

    # Trim/pad to exact sizes
    random.shuffle(train_data)
    random.shuffle(test_data)
    train_data = train_data[:train_size]
    test_data = test_data[:test_size]

    logger.info(f"Split: {len(train_data)} train, {len(test_data)} test")

    # Save train data
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    train_path = os.path.join(train_dir, "train.json")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved train data to {train_path}")

    # Save test data
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    test_path = os.path.join(test_dir, "test.json")
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved test data to {test_path}")

    # Compute data hash for reproducibility
    train_hash = hashlib.sha256(json.dumps(train_data, sort_keys=True).encode()).hexdigest()[:16]
    test_hash = hashlib.sha256(json.dumps(test_data, sort_keys=True).encode()).hexdigest()[:16]

    # Create manifest
    manifest = {
        "project": "PEFT Comparison Experiment — AI4015 Agentic AI",
        "student_id": "23K-0068",
        "created_at": datetime.now().isoformat(),
        "source_dataset": "glaiveai/glaive-function-calling-v2",
        "raw_data_path": raw_path,
        "seed": seed,
        "processing": {
            "total_raw_examples": len(raw_examples),
            "valid_after_mapping": len(valid_examples),
            "unmapped_excluded": unmapped_count,
            "parse_errors": parse_error_count,
            "sampled": len(sampled),
        },
        "splits": {
            "train": {
                "path": train_path,
                "count": len(train_data),
                "sha256_prefix": train_hash,
            },
            "test": {
                "path": test_path,
                "count": len(test_data),
                "sha256_prefix": test_hash,
            },
        },
        "tool_distribution": {
            "train": dict(Counter(ex["tool_name"] for ex in train_data)),
            "test": dict(Counter(ex["tool_name"] for ex in test_data)),
            "total": dict(post_sample_dist),
        },
        "permitted_tools": [
            "check_order_status", "initiate_refund",
            "track_shipment", "escalate_to_human", "no_tool"
        ],
        "top_unmapped_functions": dict(unmapped_functions.most_common(20)),
    }

    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved to {manifest_path}")

    return manifest


def main():
    manifest = process_dataset()
    logger.info("✅ Dataset preprocessing complete!")
    logger.info(f"   Train: {manifest['splits']['train']['count']} examples")
    logger.info(f"   Test:  {manifest['splits']['test']['count']} examples")
    logger.info(f"   Tool distribution: {manifest['tool_distribution']['total']}")
    return manifest


if __name__ == "__main__":
    main()
