"""Statistical Analyzer and Latency Budget Calculator for SaafHawa Agent.

Performs robust statistical evaluation on measured tool latency samples, calculates
the overall pooled median, evaluates the retry budget, and computes end-to-end latency
for the two candidate LLMs (Mistral Small 4 and Gemini 4 Argon).
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ToolStatistics:
    tool_name: str
    sample_count: int
    failed_count: int
    mean_ms: float
    median_ms: float
    std_dev_ms: float
    min_ms: float
    max_ms: float
    p90_ms: float
    p95_ms: float
    mean_sec: float
    median_sec: float
    p95_sec: float


@dataclass
class ModelLatencyEvaluation:
    model_name: str
    provider: str
    ttft_sec: float
    tokens_per_sec: float
    base_llm_sec: float
    retry_llm_sec: float
    total_llm_sec: float
    base_tool_delay_sec: float
    retry_tool_delay_sec: float
    total_tool_delay_sec: float
    expected_e2e_sec: float
    target_sec: float
    meets_target: bool
    slack_sec: float
    revised_architecture_sec: Optional[float] = None
    revised_meets_target: Optional[bool] = None


@dataclass
class ExperimentValidationResult:
    is_complete: bool
    status_label: str
    reasons: List[str]
    distinct_dates: List[str]
    sample_counts_by_tool: Dict[str, int]
    failed_counts_by_tool: Dict[str, int]
    no_duplicates: bool
    no_negative_latencies: bool
    no_missing_timestamps: bool


class LatencyAnalyzer:
    """Analyzes raw timing observations and calculates latency budget."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.student = config.get("student", {})
        self.r_val = self.student.get("r_val", 68)
        self.d_val = self.student.get("d_val", 9)
        self.n_calls = self.student.get("llm_calls_per_query_n", 5)
        self.p_failure = self.student.get("tool_failure_probability_p", 0.13)
        self.target_sec = self.student.get("end_to_end_latency_target_sec_t", 8.0)
        self.candidates = config.get("model_candidates", [])

    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        """Compute the p-th percentile (0 <= p <= 100) using linear interpolation."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        rank = (p / 100.0) * (len(sorted_vals) - 1)
        k = int(rank)
        d = rank - k
        if k + 1 < len(sorted_vals):
            return sorted_vals[k] + d * (sorted_vals[k + 1] - sorted_vals[k])
        return sorted_vals[-1]

    @staticmethod
    def _median(values: List[float]) -> float:
        """Calculate exact median of a list of floats."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        if n % 2 == 1:
            return sorted_vals[mid]
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0

    def compute_tool_statistics(self, records: List[Dict[str, Any]]) -> Dict[str, ToolStatistics]:
        """Compute descriptive statistics for each tool from valid measured samples."""
        latencies_by_tool: Dict[str, List[float]] = {}
        failed_by_tool: Dict[str, int] = {}

        for r in records:
            tool = r.get("tool_name", "")
            role = r.get("sample_role", "")
            success = str(r.get("success", "")).strip().lower() == "true"

            if role != "measured":
                continue

            if tool not in latencies_by_tool:
                latencies_by_tool[tool] = []
                failed_by_tool[tool] = 0

            if success:
                try:
                    val = float(r.get("elapsed_ms", 0.0))
                    if val > 0:
                        latencies_by_tool[tool].append(val)
                except (ValueError, TypeError):
                    pass
            else:
                failed_by_tool[tool] += 1

        stats_map: Dict[str, ToolStatistics] = {}
        for tool, vals in latencies_by_tool.items():
            n = len(vals)
            failed = failed_by_tool.get(tool, 0)
            if n == 0:
                stats_map[tool] = ToolStatistics(
                    tool_name=tool,
                    sample_count=0,
                    failed_count=failed,
                    mean_ms=0.0,
                    median_ms=0.0,
                    std_dev_ms=0.0,
                    min_ms=0.0,
                    max_ms=0.0,
                    p90_ms=0.0,
                    p95_ms=0.0,
                    mean_sec=0.0,
                    median_sec=0.0,
                    p95_sec=0.0,
                )
                continue

            mean_val = sum(vals) / n
            median_val = self._median(vals)
            min_val = min(vals)
            max_val = max(vals)
            p90_val = self._percentile(vals, 90.0)
            p95_val = self._percentile(vals, 95.0)
            variance = sum((x - mean_val) ** 2 for x in vals) / (n - 1) if n > 1 else 0.0
            std_dev = math.sqrt(variance)

            stats_map[tool] = ToolStatistics(
                tool_name=tool,
                sample_count=n,
                failed_count=failed,
                mean_ms=round(mean_val, 3),
                median_ms=round(median_val, 3),
                std_dev_ms=round(std_dev, 3),
                min_ms=round(min_val, 3),
                max_ms=round(max_val, 3),
                p90_ms=round(p90_val, 3),
                p95_ms=round(p95_val, 3),
                mean_sec=round(mean_val / 1000.0, 4),
                median_sec=round(median_val / 1000.0, 4),
                p95_sec=round(p95_val / 1000.0, 4),
            )

        return stats_map

    def compute_pooled_median(self, records: List[Dict[str, Any]]) -> Tuple[float, float, int, bool]:
        """Compute the pooled median of all valid measured observations across all tools.

        Returns (pooled_median_ms, pooled_median_sec, total_observation_count, all_tools_represented).
        """
        all_latencies: List[float] = []
        tools_present: Set[str] = set()
        for r in records:
            if r.get("sample_role") == "measured" and str(r.get("success", "")).strip().lower() == "true":
                try:
                    val = float(r.get("elapsed_ms", 0.0))
                    if val > 0:
                        all_latencies.append(val)
                        tools_present.add(r.get("tool_name", ""))
                except (ValueError, TypeError):
                    pass

        required_tools = {"OpenAQ", "Open-Meteo", "HeatIndex", "KnowledgeBase"}
        all_tools_represented = required_tools.issubset(tools_present)

        if not all_latencies:
            return 0.0, 0.0, 0, False

        med_ms = self._median(all_latencies)
        return round(med_ms, 3), round(med_ms / 1000.0, 4), len(all_latencies), all_tools_represented

    def evaluate_model_candidates(
        self, pooled_median_sec: float, tool_stats: Dict[str, ToolStatistics]
    ) -> List[ModelLatencyEvaluation]:
        """Calculate sequential latency budget and architectural revision for model candidates."""
        # Assignment parameters:
        # Calls 1..4 have 1 retry with probability p = 0.13
        # Expected retries = 4 * 0.13 = 0.52
        # Expected total tool attempts = 4 * 1.13 = 4.52
        p = self.p_failure
        expected_tool_ops = 4.0
        expected_tool_retries = expected_tool_ops * p  # 0.52
        expected_total_tool_attempts = expected_tool_ops * (1.0 + p)  # 4.52

        base_tool_delay_sec = expected_tool_ops * pooled_median_sec
        retry_tool_delay_sec = expected_tool_retries * pooled_median_sec
        total_tool_delay_sec = expected_total_tool_attempts * pooled_median_sec

        # Token output sums:
        # Base: Calls 1..4 output 350, Call 5 outputs 600 -> Total = 2,000
        # Retried outputs: p * 4 * 350 = 0.13 * 1,400 = 182 tokens
        # Total output tokens expected = 2,182
        base_output_tokens = 2000.0
        retry_output_tokens = 4.0 * 350.0 * p  # 182.0
        total_output_tokens = base_output_tokens + retry_output_tokens  # 2182.0

        evaluations: List[ModelLatencyEvaluation] = []
        for cand in self.candidates:
            name = cand["name"]
            provider = cand["provider"]
            ttft = float(cand["reported_ttft_sec"])
            v = float(cand["reported_tokens_per_sec"])

            # Base LLM latency: 5 * TTFT + 2,000 / v
            base_llm_sec = 5.0 * ttft + (base_output_tokens / v)

            # Retry LLM latency: 4 * p * TTFT + 182 / v
            retry_llm_sec = (4.0 * p * ttft) + (retry_output_tokens / v)

            # Total LLM latency = 5.52 * TTFT + 2,182 / v
            total_llm_sec = base_llm_sec + retry_llm_sec

            # Expected End-to-End Latency
            expected_e2e_sec = total_llm_sec + total_tool_delay_sec
            meets_target = expected_e2e_sec <= self.target_sec
            slack = round(self.target_sec - expected_e2e_sec, 3)

            # Architectural modification evaluation:
            # Deterministic router + parallel tool execution + single final synthesis call (300 output cap)
            # Parallel tool delay = max(OpenAQ, Open-Meteo) + max(HeatIndex, KnowledgeBase)
            openaq_med = tool_stats.get("OpenAQ", None)
            meteo_med = tool_stats.get("Open-Meteo", None)
            hi_med = tool_stats.get("HeatIndex", None)
            kb_med = tool_stats.get("KnowledgeBase", None)

            t_aq = openaq_med.median_sec if openaq_med and openaq_med.median_sec > 0 else pooled_median_sec
            t_meteo = meteo_med.median_sec if meteo_med and meteo_med.median_sec > 0 else pooled_median_sec
            t_hi = hi_med.median_sec if hi_med and hi_med.median_sec > 0 else pooled_median_sec
            t_kb = kb_med.median_sec if kb_med and kb_med.median_sec > 0 else pooled_median_sec

            parallel_external_tool_sec = max(t_aq, t_meteo)
            parallel_local_tool_sec = max(t_hi, t_kb)
            total_parallel_tools_sec = parallel_external_tool_sec + parallel_local_tool_sec

            # Revised single LLM call with 300 token cap
            revised_llm_sec = ttft + (300.0 / v)
            revised_e2e_sec = revised_llm_sec + total_parallel_tools_sec
            revised_meets = revised_e2e_sec <= self.target_sec

            evaluations.append(
                ModelLatencyEvaluation(
                    model_name=name,
                    provider=provider,
                    ttft_sec=ttft,
                    tokens_per_sec=v,
                    base_llm_sec=round(base_llm_sec, 4),
                    retry_llm_sec=round(retry_llm_sec, 4),
                    total_llm_sec=round(total_llm_sec, 4),
                    base_tool_delay_sec=round(base_tool_delay_sec, 4),
                    retry_tool_delay_sec=round(retry_tool_delay_sec, 4),
                    total_tool_delay_sec=round(total_tool_delay_sec, 4),
                    expected_e2e_sec=round(expected_e2e_sec, 4),
                    target_sec=self.target_sec,
                    meets_target=meets_target,
                    slack_sec=slack,
                    revised_architecture_sec=round(revised_e2e_sec, 4),
                    revised_meets_target=revised_meets,
                )
            )

        return evaluations

    def validate_dataset(self, records: List[Dict[str, Any]]) -> ExperimentValidationResult:
        """Run all data quality and completion checks on raw CSV records."""
        reasons: List[str] = []
        sample_ids: Set[str] = set()
        has_duplicate_id = False
        has_negative_lat = False
        has_missing_timestamp = False

        dates_by_tool: Dict[str, Set[str]] = {
            "OpenAQ": set(),
            "Open-Meteo": set(),
            "HeatIndex": set(),
            "KnowledgeBase": set(),
        }
        success_counts: Dict[str, int] = {"OpenAQ": 0, "Open-Meteo": 0, "HeatIndex": 0, "KnowledgeBase": 0}
        failed_counts: Dict[str, int] = {"OpenAQ": 0, "Open-Meteo": 0, "HeatIndex": 0, "KnowledgeBase": 0}
        all_measured_dates: Set[str] = set()

        for r in records:
            sid = r.get("sample_id", "")
            if sid:
                if sid in sample_ids:
                    has_duplicate_id = True
                sample_ids.add(sid)

            tool = r.get("tool_name", "")
            role = r.get("sample_role", "")
            success = str(r.get("success", "")).strip().lower() == "true"
            dt = r.get("local_date", "")
            started = r.get("started_at", "")
            completed = r.get("completed_at", "")

            if role == "measured":
                if not started or not completed:
                    has_missing_timestamp = True
                try:
                    el = float(r.get("elapsed_ms", 0.0))
                    if el < 0:
                        has_negative_lat = True
                except (ValueError, TypeError):
                    pass

                if tool in success_counts:
                    if success:
                        success_counts[tool] += 1
                        if dt:
                            dates_by_tool[tool].add(dt)
                            all_measured_dates.add(dt)
                    else:
                        failed_counts[tool] += 1

        distinct_dates = sorted(list(all_measured_dates))

        # Check sample counts
        for t, cnt in success_counts.items():
            if cnt < 20:
                reasons.append(f"{t}: {cnt}/20 required successful measurements collected.")

        # Check distinct dates
        if len(distinct_dates) < 2:
            reasons.append(
                f"Measurements recorded on {len(distinct_dates)} calendar date(s) ({distinct_dates}). At least 2 distinct dates required."
            )

        if has_duplicate_id:
            reasons.append("Duplicate sample_id values detected in raw log.")
        if has_negative_lat:
            reasons.append("Negative latency values detected in raw log.")
        if has_missing_timestamp:
            reasons.append("Missing start/complete timestamps detected for measured samples.")

        # Determine status label
        if not reasons:
            is_complete = True
            status_label = "COMPLETE — ALL EXPERIMENT REQUIREMENTS SATISFIED"
        else:
            is_complete = False
            # Check if Day 1 was achieved
            day1_met = len(distinct_dates) >= 1 and all(
                cnt >= 10 for cnt in success_counts.values()
            )
            if len(distinct_dates) == 1 and day1_met:
                status_label = "PARTIAL — FIRST EXPERIMENT DAY RECORDED"
            elif any(cnt == 0 for cnt in success_counts.values()):
                status_label = "PARTIAL — INSUFFICIENT SAMPLES / BLOCKED COMPONENT"
            else:
                status_label = "PARTIAL — SECOND EXPERIMENT DAY REQUIRED"

        return ExperimentValidationResult(
            is_complete=is_complete,
            status_label=status_label,
            reasons=reasons,
            distinct_dates=distinct_dates,
            sample_counts_by_tool=success_counts,
            failed_counts_by_tool=failed_counts,
            no_duplicates=not has_duplicate_id,
            no_negative_latencies=not has_negative_lat,
            no_missing_timestamps=not has_missing_timestamp,
        )
