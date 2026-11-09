"""
Dataset Download — PEFT Experiment (23K-0068)
Downloads glaive-function-calling-v2 from HuggingFace and saves raw data locally.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def download_dataset(output_dir: str = "data/raw/", dataset_name: str = "glaiveai/glaive-function-calling-v2") -> str:
    """
    Download the Glaive function-calling dataset from HuggingFace.

    Returns:
        Path to the saved raw data directory.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading dataset: {dataset_name}")
    start = time.time()

    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        logger.info("Retrying with streaming disabled and no cache verification...")
        try:
            dataset = load_dataset(dataset_name, split="train", verification_mode="no_checks")
        except Exception as e2:
            logger.error(f"Retry also failed: {e2}")
            sys.exit(1)

    elapsed = time.time() - start
    logger.info(f"Download complete in {elapsed:.1f}s — {len(dataset)} examples")

    # Save as JSONL for portability
    raw_path = os.path.join(output_dir, "glaive_function_calling_v2.jsonl")
    logger.info(f"Saving raw data to {raw_path}")

    with open(raw_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            if (i + 1) % 10000 == 0:
                logger.info(f"  Written {i + 1}/{len(dataset)} examples...")

    logger.info(f"Saved {len(dataset)} examples to {raw_path}")

    # Save metadata
    meta = {
        "dataset_name": dataset_name,
        "total_examples": len(dataset),
        "columns": list(dataset.column_names),
        "download_time_seconds": round(elapsed, 2),
        "output_path": raw_path,
    }
    meta_path = os.path.join(output_dir, "download_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved to {meta_path}")

    return raw_path


def main():
    raw_path = download_dataset()
    logger.info(f"✅ Raw dataset ready at: {raw_path}")
    return raw_path


if __name__ == "__main__":
    main()
