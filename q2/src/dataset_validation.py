"""
Dataset Validation — PEFT Experiment (23K-0068)
Validates the processed train/test datasets for correctness, schema compliance,
distribution balance, and no test leakage.
"""

import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_tool_schemas(schema_path: str = "configs/tool_schemas.json") -> dict:
    """Load tool schemas from JSON config."""
    with open(schema_path, "r") as f:
        data = json.load(f)
    return {tool["name"]: tool for tool in data["tools"]}


def validate_example_structure(example: dict, schemas: dict) -> list[str]:
    """
    Validate a single example's JSON structure and schema compliance.
    Returns a list of error messages (empty = valid).
    """
    errors = []

    # Required fields
    required_fields = {"id", "user_message", "tool_name", "arguments"}
    missing = required_fields - set(example.keys())
    if missing:
        errors.append(f"Missing fields: {missing}")
        return errors  # can't validate further

    # Type checks
    if not isinstance(example["id"], str):
        errors.append(f"'id' must be string, got {type(example['id']).__name__}")
    if not isinstance(example["user_message"], str):
        errors.append(f"'user_message' must be string, got {type(example['user_message']).__name__}")
    if not isinstance(example["tool_name"], str):
        errors.append(f"'tool_name' must be string, got {type(example['tool_name']).__name__}")
    if not isinstance(example["arguments"], dict):
        errors.append(f"'arguments' must be dict, got {type(example['arguments']).__name__}")

    # Tool name must be one of the permitted tools
    valid_tools = set(schemas.keys())
    if example["tool_name"] not in valid_tools:
        errors.append(f"Invalid tool_name: '{example['tool_name']}' not in {valid_tools}")
        return errors

    # Schema validation for arguments
    tool_name = example["tool_name"]
    schema = schemas[tool_name]
    args = example["arguments"]

    if tool_name == "no_tool":
        if args:
            errors.append(f"no_tool should have empty arguments, got: {args}")
        return errors

    # Check required arguments
    for arg_name in schema["required_arguments"]:
        if arg_name not in args:
            errors.append(f"Missing required argument '{arg_name}' for {tool_name}")
        elif not isinstance(args[arg_name], str):
            errors.append(f"Argument '{arg_name}' must be string for {tool_name}")
        elif not args[arg_name].strip():
            errors.append(f"Argument '{arg_name}' is empty for {tool_name}")

    # Check no unknown arguments
    all_known = set(schema["required_arguments"].keys()) | set(schema["optional_arguments"].keys())
    unknown = set(args.keys()) - all_known
    if unknown:
        errors.append(f"Unknown arguments for {tool_name}: {unknown}")

    return errors


def check_test_leakage(train_data: list[dict], test_data: list[dict]) -> dict:
    """Check for data leakage between train and test sets."""
    # Check by ID
    train_ids = {ex["id"] for ex in train_data}
    test_ids = {ex["id"] for ex in test_data}
    id_overlap = train_ids & test_ids

    # Check by user message content (exact match)
    train_messages = {ex["user_message"] for ex in train_data}
    test_messages = {ex["user_message"] for ex in test_data}
    message_overlap = train_messages & test_messages

    # Check by content hash (user_message + tool_name + arguments)
    def content_hash(ex):
        content = f"{ex['user_message']}|{ex['tool_name']}|{json.dumps(ex['arguments'], sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()

    train_hashes = {content_hash(ex) for ex in train_data}
    test_hashes = {content_hash(ex) for ex in test_data}
    hash_overlap = train_hashes & test_hashes

    return {
        "id_overlap_count": len(id_overlap),
        "id_overlaps": list(id_overlap)[:10],
        "message_overlap_count": len(message_overlap),
        "message_overlaps": list(message_overlap)[:5],
        "content_hash_overlap_count": len(hash_overlap),
        "has_leakage": len(id_overlap) > 0 or len(hash_overlap) > 0,
    }


def validate_dataset(
    train_path: str = "data/train/train.json",
    test_path: str = "data/test/test.json",
    schema_path: str = "configs/tool_schemas.json",
    expected_train: int = 1800,
    expected_test: int = 200,
) -> dict:
    """
    Run full validation suite on the processed dataset.

    Returns a report dict with pass/fail status and details.
    """
    report = {"checks": {}, "passed": True, "errors": []}

    # Load schemas
    schemas = load_tool_schemas(schema_path)
    logger.info(f"Loaded schemas for {len(schemas)} tools")

    # Load data
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logger.info(f"Loaded {len(train_data)} train, {len(test_data)} test examples")

    # ── Check 1: Counts ──
    count_ok = len(train_data) == expected_train and len(test_data) == expected_test
    report["checks"]["counts"] = {
        "passed": count_ok,
        "train_count": len(train_data),
        "test_count": len(test_data),
        "expected_train": expected_train,
        "expected_test": expected_test,
    }
    if not count_ok:
        report["passed"] = False
        msg = f"Count mismatch: train={len(train_data)} (exp {expected_train}), test={len(test_data)} (exp {expected_test})"
        report["errors"].append(msg)
        logger.warning(f"❌ {msg}")
    else:
        logger.info(f"✅ Counts OK: {len(train_data)} train, {len(test_data)} test")

    # ── Check 2: JSON Structure Validation ──
    all_errors = {"train": [], "test": []}
    for split_name, data in [("train", train_data), ("test", test_data)]:
        for i, example in enumerate(data):
            errs = validate_example_structure(example, schemas)
            if errs:
                all_errors[split_name].append({"index": i, "id": example.get("id"), "errors": errs})

    structure_ok = not any(all_errors[s] for s in all_errors)
    report["checks"]["structure"] = {
        "passed": structure_ok,
        "train_errors": len(all_errors["train"]),
        "test_errors": len(all_errors["test"]),
        "sample_errors": (all_errors["train"][:3] + all_errors["test"][:3]),
    }
    if not structure_ok:
        report["passed"] = False
        msg = f"Structure errors: {len(all_errors['train'])} train, {len(all_errors['test'])} test"
        report["errors"].append(msg)
        logger.warning(f"❌ {msg}")
    else:
        logger.info(f"✅ All examples pass structure/schema validation")

    # ── Check 3: Tool Distribution ──
    train_dist = Counter(ex["tool_name"] for ex in train_data)
    test_dist = Counter(ex["tool_name"] for ex in test_data)
    total_dist = Counter(ex["tool_name"] for ex in train_data + test_data)

    # Check all permitted tools are represented
    missing_tools = set(schemas.keys()) - set(total_dist.keys())
    dist_ok = len(missing_tools) == 0

    report["checks"]["distribution"] = {
        "passed": dist_ok,
        "train_distribution": dict(train_dist),
        "test_distribution": dict(test_dist),
        "total_distribution": dict(total_dist),
        "missing_tools": list(missing_tools),
    }

    if not dist_ok:
        report["passed"] = False
        msg = f"Missing tools in data: {missing_tools}"
        report["errors"].append(msg)
        logger.warning(f"⚠️  {msg}")
    else:
        logger.info(f"✅ All tools represented in data")

    logger.info(f"   Train distribution: {dict(train_dist)}")
    logger.info(f"   Test distribution:  {dict(test_dist)}")

    # ── Check 4: Test Leakage ──
    leakage = check_test_leakage(train_data, test_data)
    report["checks"]["leakage"] = leakage

    if leakage["has_leakage"]:
        report["passed"] = False
        msg = (f"Test leakage detected! IDs: {leakage['id_overlap_count']}, "
               f"content: {leakage['content_hash_overlap_count']}")
        report["errors"].append(msg)
        logger.warning(f"❌ {msg}")
    else:
        logger.info(f"✅ No test leakage detected")
        logger.info(f"   Duplicate user messages: {leakage['message_overlap_count']} "
                     f"(acceptable if different tools)")

    # ── Check 5: Argument Schema Compliance (detailed) ──
    arg_issues = 0
    for split_name, data in [("train", train_data), ("test", test_data)]:
        for ex in data:
            tool = ex["tool_name"]
            args = ex["arguments"]
            if tool == "no_tool":
                continue
            schema = schemas[tool]
            for req_arg in schema["required_arguments"]:
                if req_arg not in args or not str(args[req_arg]).strip():
                    arg_issues += 1

    args_ok = arg_issues == 0
    report["checks"]["argument_compliance"] = {
        "passed": args_ok,
        "missing_required_args": arg_issues,
    }
    if not args_ok:
        report["passed"] = False
        msg = f"Argument compliance issues: {arg_issues} missing required arguments"
        report["errors"].append(msg)
        logger.warning(f"❌ {msg}")
    else:
        logger.info(f"✅ All required arguments present and non-empty")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  DATASET VALIDATION SUMMARY")
    print("=" * 60)
    total_checks = len(report["checks"])
    passed_checks = sum(1 for c in report["checks"].values() if c.get("passed"))
    print(f"  Checks passed: {passed_checks}/{total_checks}")
    print(f"  Overall: {'✅ PASSED' if report['passed'] else '❌ FAILED'}")
    if report["errors"]:
        print(f"\n  Errors:")
        for err in report["errors"]:
            print(f"    - {err}")
    print()

    # Tool distribution table
    print("  Tool Distribution:")
    print(f"  {'Tool':<25} {'Train':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for tool in sorted(schemas.keys()):
        tr = train_dist.get(tool, 0)
        te = test_dist.get(tool, 0)
        tot = total_dist.get(tool, 0)
        print(f"  {tool:<25} {tr:>8} {te:>8} {tot:>8}")
    print(f"  {'TOTAL':<25} {sum(train_dist.values()):>8} {sum(test_dist.values()):>8} {sum(total_dist.values()):>8}")
    print("=" * 60)

    # Save report
    report_path = "results/logs/validation_report.json"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Validation report saved to {report_path}")

    return report


def main():
    report = validate_dataset()
    if not report["passed"]:
        logger.error("Dataset validation FAILED — fix issues before training")
        sys.exit(1)
    else:
        logger.info("✅ Dataset validation passed — ready for training!")
    return report


if __name__ == "__main__":
    main()
