"""Raw Latency Logger for SaafHawa Agent Latency Experiment.

Maintains an append-only CSV log at results/raw_tool_latency_log.csv conforming
strictly to the assignment specification.
"""

import csv
from datetime import datetime
import os
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

LOG_COLUMNS = [
    "run_id",
    "sample_id",
    "local_date",
    "started_at",
    "completed_at",
    "timezone",
    "tool_name",
    "sample_role",
    "sample_number",
    "attempt_number",
    "success",
    "http_status",
    "elapsed_ms",
    "json_parse_ms",
    "error_category",
    "error_message",
    "request_url",
    "location_id",
    "station_name",
    "pm25_value",
    "pm25_unit",
    "pm25_observed_at",
    "pm10_value",
    "pm10_unit",
    "pm10_observed_at",
    "temperature_c",
    "relative_humidity_pct",
    "wind_speed",
    "wind_speed_unit",
    "heat_index_c",
    "heat_index_f",
    "kb_query_id",
    "kb_result_count",
    "kb_document_ids",
    "extra_metadata",
]


class LatencyLogger:
    """Manages appending and reading records from the raw tool latency CSV."""

    def __init__(self, csv_path: str, timezone_str: str = "Asia/Karachi"):
        self.csv_path = csv_path
        self.timezone_str = timezone_str
        self.tz = ZoneInfo(timezone_str)
        self._ensure_header()

    def _ensure_header(self) -> None:
        """Create directory and write header row if CSV file does not exist."""
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
                writer.writeheader()

    def log_record(self, record: Dict[str, Any]) -> None:
        """Append a single sample observation or failed attempt to the CSV.

        Guarantees that all required columns are present and flushes immediately.
        """
        row = {col: "" for col in LOG_COLUMNS}
        for k, v in record.items():
            if k in row:
                if v is None:
                    row[k] = ""
                elif isinstance(v, bool):
                    row[k] = "True" if v else "False"
                elif isinstance(v, float):
                    row[k] = f"{v:.4f}"
                else:
                    row[k] = str(v)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writerow(row)

    def read_all_records(self) -> List[Dict[str, Any]]:
        """Read all rows from the raw latency log."""
        if not os.path.exists(self.csv_path):
            return []

        records: List[Dict[str, Any]] = []
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
        return records

    def get_summary_counts(self) -> Dict[str, Any]:
        """Compute counts of successful measured samples grouped by local date and tool."""
        records = self.read_all_records()
        by_date: Dict[str, Dict[str, int]] = {}
        total_by_tool: Dict[str, int] = {}
        failed_by_tool: Dict[str, int] = {}

        for r in records:
            tool = r.get("tool_name", "")
            role = r.get("sample_role", "")
            success = str(r.get("success", "")).strip().lower() == "true"
            dt = r.get("local_date", "")

            if role == "measured":
                if success:
                    total_by_tool[tool] = total_by_tool.get(tool, 0) + 1
                    if dt:
                        if dt not in by_date:
                            by_date[dt] = {}
                        by_date[dt][tool] = by_date[dt].get(tool, 0) + 1
                else:
                    failed_by_tool[tool] = failed_by_tool.get(tool, 0) + 1

        distinct_dates = sorted(list(by_date.keys()))
        return {
            "total_records": len(records),
            "distinct_dates": distinct_dates,
            "counts_by_date": by_date,
            "successful_by_tool": total_by_tool,
            "failed_by_tool": failed_by_tool,
        }

