"""Unit tests for the latency analyzer, pooled median derivation, and two-day validation."""

import unittest
from src.analyzer import LatencyAnalyzer


class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        self.config = {
            "student": {
                "student_id": "23K-0068",
                "r_val": 68,
                "d_val": 9,
                "normal_daily_queries_q": 3200,
                "llm_calls_per_query_n": 5,
                "tool_failure_probability_p": 0.13,
                "heatwave_days_per_month_h": 4,
                "monthly_budget_usd_b": 1880,
                "end_to_end_latency_target_sec_t": 8.0,
            },
            "geo": {"timezone": "Asia/Karachi"},
            "model_candidates": [
                {
                    "name": "Mistral Small 4",
                    "provider": "Mistral AI",
                    "reported_ttft_sec": 0.84,
                    "reported_tokens_per_sec": 173.0,
                },
                {
                    "name": "Gemini 4 Argon",
                    "provider": "Google DeepMind",
                    "reported_ttft_sec": 40.21,
                    "reported_tokens_per_sec": 71.0,
                },
            ],
        }
        self.analyzer = LatencyAnalyzer(self.config)

    def test_pooled_median_calculation(self):
        records = [
            {"sample_role": "measured", "success": "True", "elapsed_ms": "100.0", "tool_name": "Open-Meteo"},
            {"sample_role": "measured", "success": "True", "elapsed_ms": "200.0", "tool_name": "Open-Meteo"},
            {"sample_role": "measured", "success": "True", "elapsed_ms": "300.0", "tool_name": "OpenAQ"},
            {"sample_role": "measured", "success": "True", "elapsed_ms": "0.1", "tool_name": "HeatIndex"},
            {"sample_role": "measured", "success": "True", "elapsed_ms": "0.2", "tool_name": "KnowledgeBase"},
            # Warm-up sample should be ignored
            {"sample_role": "warmup", "success": "True", "elapsed_ms": "999.0", "tool_name": "Open-Meteo"},
            # Failed sample should be ignored
            {"sample_role": "measured", "success": "False", "elapsed_ms": "5000.0", "tool_name": "OpenAQ"},
        ]
        # Measured values: [0.1, 0.2, 100.0, 200.0, 300.0] -> Median is 100.0
        med_ms, med_sec, count, all_rep = self.analyzer.compute_pooled_median(records)
        self.assertEqual(count, 5)
        self.assertEqual(med_ms, 100.0)
        self.assertEqual(med_sec, 0.1)
        self.assertTrue(all_rep)

    def test_two_day_validation_rejects_single_day(self):
        records = []
        # Generate 20 samples all on same calendar date 2026-11-09
        for i in range(20):
            records.append({
                "sample_id": f"SMP_M_{i}",
                "sample_role": "measured",
                "success": "True",
                "local_date": "2026-11-09",
                "started_at": "2026-11-09T19:00:00Z",
                "completed_at": "2026-11-09T19:00:01Z",
                "elapsed_ms": "150.0",
                "tool_name": "Open-Meteo",
            })
            records.append({
                "sample_id": f"SMP_AQ_{i}",
                "sample_role": "measured",
                "success": "True",
                "local_date": "2026-11-09",
                "started_at": "2026-11-09T19:00:00Z",
                "completed_at": "2026-11-09T19:00:01Z",
                "elapsed_ms": "250.0",
                "tool_name": "OpenAQ",
            })
            records.append({
                "sample_id": f"SMP_HI_{i}",
                "sample_role": "measured",
                "success": "True",
                "local_date": "2026-11-09",
                "started_at": "2026-11-09T19:00:00Z",
                "completed_at": "2026-11-09T19:00:01Z",
                "elapsed_ms": "0.05",
                "tool_name": "HeatIndex",
            })
            records.append({
                "sample_id": f"SMP_KB_{i}",
                "sample_role": "measured",
                "success": "True",
                "local_date": "2026-11-09",
                "started_at": "2026-11-09T19:00:00Z",
                "completed_at": "2026-11-09T19:00:01Z",
                "elapsed_ms": "0.15",
                "tool_name": "KnowledgeBase",
            })

        val = self.analyzer.validate_dataset(records)
        # Must NOT be complete because all samples are on only 1 date!
        self.assertFalse(val.is_complete)
        self.assertEqual(len(val.distinct_dates), 1)
        self.assertTrue(any("2 distinct dates required" in r for r in val.reasons))

    def test_two_day_validation_passes_when_distinct_dates_and_counts_met(self):
        records = []
        for dt in ["2026-11-09", "2026-11-10"]:
            for i in range(10):
                for tool in ["OpenAQ", "Open-Meteo", "HeatIndex", "KnowledgeBase"]:
                    records.append({
                        "sample_id": f"SMP_{tool}_{dt}_{i}",
                        "sample_role": "measured",
                        "success": "True",
                        "local_date": dt,
                        "started_at": f"{dt}T19:00:00Z",
                        "completed_at": f"{dt}T19:00:01Z",
                        "elapsed_ms": "100.0",
                        "tool_name": tool,
                    })

        val = self.analyzer.validate_dataset(records)
        self.assertTrue(val.is_complete)
        self.assertEqual(val.status_label, "COMPLETE — ALL EXPERIMENT REQUIREMENTS SATISFIED")
        self.assertEqual(len(val.distinct_dates), 2)

    def test_model_latency_calculation(self):
        tool_stats = self.analyzer.compute_tool_statistics([])
        evals = self.analyzer.evaluate_model_candidates(pooled_median_sec=0.15, tool_stats=tool_stats)
        self.assertEqual(len(evals), 2)

        mistral = evals[0]
        # Base LLM: 5 * 0.84 + 2000 / 173 = 4.2 + 11.5607 = 15.7607
        # Retry LLM: 4 * 0.13 * 0.84 + 182 / 173 = 0.4368 + 1.0520 = 1.4888
        # Total LLM: 17.2495 s
        # Tool delay: 4.52 * 0.15 = 0.678 s
        # Total E2E: 17.9275 s -> Fails 8.0 s target
        self.assertFalse(mistral.meets_target)
        self.assertGreater(mistral.expected_e2e_sec, 8.0)

        # But with revised architecture:
        # revised LLM: 0.84 + 300 / 173 = 0.84 + 1.7341 = 2.5741 s
        # Parallel tools: ~0.3 s
        # Total ~ 2.87 s < 8.0 s -> PASS
        self.assertTrue(mistral.revised_meets_target)


if __name__ == "__main__":
    unittest.main()
