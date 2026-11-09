"""Unit tests for the append-only raw CSV logger."""

import os
import tempfile
import unittest
from src.logger import LOG_COLUMNS, LatencyLogger


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp_dir.name, "test_log.csv")
        self.logger = LatencyLogger(self.csv_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_header_creation(self):
        self.assertTrue(os.path.exists(self.csv_path))
        records = self.logger.read_all_records()
        self.assertEqual(len(records), 0)

    def test_append_records(self):
        rec1 = {
            "run_id": "RUN_01",
            "sample_id": "SMP_01",
            "local_date": "2026-11-09",
            "started_at": "2026-11-09T19:00:00Z",
            "completed_at": "2026-11-09T19:00:01Z",
            "tool_name": "Open-Meteo",
            "sample_role": "measured",
            "success": True,
            "elapsed_ms": 120.5,
        }
        self.logger.log_record(rec1)

        rec2 = {
            "run_id": "RUN_01",
            "sample_id": "SMP_02",
            "local_date": "2026-11-09",
            "started_at": "2026-11-09T19:00:02Z",
            "completed_at": "2026-11-09T19:00:03Z",
            "tool_name": "Open-Meteo",
            "sample_role": "measured",
            "success": False,
            "error_category": "TIMEOUT",
            "elapsed_ms": 5000.0,
        }
        self.logger.log_record(rec2)

        records = self.logger.read_all_records()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["sample_id"], "SMP_01")
        self.assertEqual(records[0]["success"], "True")
        self.assertEqual(records[1]["sample_id"], "SMP_02")
        self.assertEqual(records[1]["success"], "False")
        self.assertEqual(records[1]["error_category"], "TIMEOUT")

    def test_summary_counts(self):
        for i in range(5):
            self.logger.log_record({
                "sample_id": f"SMP_OK_{i}",
                "local_date": "2026-11-09",
                "tool_name": "HeatIndex",
                "sample_role": "measured",
                "success": True,
                "elapsed_ms": 0.05,
            })
        # Add 1 failed record
        self.logger.log_record({
            "sample_id": "SMP_FAIL",
            "local_date": "2026-11-09",
            "tool_name": "HeatIndex",
            "sample_role": "measured",
            "success": False,
            "elapsed_ms": 0.0,
        })
        # Add 1 warmup record
        self.logger.log_record({
            "sample_id": "SMP_WARMUP",
            "local_date": "2026-11-09",
            "tool_name": "HeatIndex",
            "sample_role": "warmup",
            "success": True,
            "elapsed_ms": 0.05,
        })

        summary = self.logger.get_summary_counts()
        self.assertEqual(summary["successful_by_tool"].get("HeatIndex"), 5)
        self.assertEqual(summary["failed_by_tool"].get("HeatIndex"), 1)
        self.assertEqual(summary["distinct_dates"], ["2026-11-09"])


if __name__ == "__main__":
    unittest.main()

