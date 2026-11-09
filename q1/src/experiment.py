"""Main Experiment Runner for SaafHawa Agent Latency Benchmark.

Orchestrates live Open-Meteo API queries, OpenAQ v3 API queries, local NWS Heat Index
calculations, and local BM25 Knowledge-Base lookups over a strict two-calendar-day protocol.
"""

from datetime import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from .analyzer import LatencyAnalyzer
from .heat_index import timed_calculate_heat_index
from .knowledge_base import KnowledgeBaseService
from .logger import LatencyLogger
from .open_meteo_client import OpenMeteoClient
from .openaq_client import OpenAQClient
from .report_generator import ReportGenerator


def get_project_root() -> str:
    """Return the absolute path to Q1_Latency_Experiment root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, ".."))


def run_experiment() -> Dict[str, Any]:
    project_root = get_project_root()
    dotenv_path = os.path.join(project_root, ".env")
    load_dotenv(dotenv_path)

    config_path = os.path.join(project_root, "config", "experiment_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tz_str = config["geo"]["timezone"]
    tz = ZoneInfo(tz_str)
    now_local = datetime.now(tz)
    local_date_str = now_local.strftime("%Y-%m-%d")
    run_id = f"RUN_{now_local.strftime('%Y%m%d_%H%M%S')}"

    results_dir = os.path.join(project_root, "results")
    data_dir = os.path.join(project_root, "data")
    kb_dir = os.path.join(data_dir, "knowledge_base")
    kb_meta_path = os.path.join(kb_dir, "sources_metadata.json")
    kb_queries_path = os.path.join(kb_dir, "test_queries.json")
    metadata_path = os.path.join(results_dir, "experiment_metadata.json")
    csv_path = os.path.join(results_dir, "raw_tool_latency_log.csv")

    logger = LatencyLogger(csv_path, timezone_str=tz_str)
    analyzer = LatencyAnalyzer(config)
    report_gen = ReportGenerator(config, results_dir)

    print("=" * 78)
    print("  SAAFAHAWA AGENTIC AI — Q1 LATENCY EXPERIMENT BENCHMARK")
    print(f"  Student ID: {config['student']['student_id']} | Date: {local_date_str} ({tz_str})")
    print(f"  Run ID: {run_id}")
    print("=" * 78)

    # 1. Inspect existing records to determine day count and progress
    existing_summary = logger.get_summary_counts()
    distinct_dates = existing_summary["distinct_dates"]
    counts_today = existing_summary["counts_by_date"].get(local_date_str, {})
    daily_target = config["experiment_runner"]["daily_target_per_tool"]

    print(f"Prior recorded dates: {distinct_dates if distinct_dates else 'None (First Session)'}")
    print(f"Today's date ({local_date_str}) existing successful samples:")
    for t_name in ["Open-Meteo", "HeatIndex", "KnowledgeBase", "OpenAQ"]:
        print(f"  - {t_name}: {counts_today.get(t_name, 0)}/{daily_target}")

    # Determine whether we are on Day 1 or Day 2
    if not distinct_dates or (len(distinct_dates) == 1 and local_date_str in distinct_dates):
        current_day_number = 1
    else:
        current_day_number = 2

    # 2. Initialize Clients & Services
    openaq_api_key = os.getenv("OPENAQ_API_KEY", "").strip()
    openaq_client = OpenAQClient(
        api_key=openaq_api_key,
        base_url=config["openaq"]["base_url"],
        latitude=config["geo"]["latitude"],
        longitude=config["geo"]["longitude"],
        search_radius_meters=config["openaq"]["search_radius_meters"],
        timeout_sec=config["experiment_runner"]["http_timeout_sec"],
    )

    meteo_client = OpenMeteoClient(
        base_url=config["open_meteo"]["base_url"],
        latitude=config["geo"]["latitude"],
        longitude=config["geo"]["longitude"],
        timezone=tz_str,
        timeout_sec=config["experiment_runner"]["http_timeout_sec"],
    )

    print("\n[Knowledge Base] Initializing local BM25 retriever...")
    kb_service = KnowledgeBaseService(kb_dir=kb_dir, metadata_path=kb_meta_path)
    print(f"[Knowledge Base] Indexed {kb_service.retriever.num_docs} documents in {kb_service.indexing_time_ms:.2f} ms.")

    with open(kb_queries_path, "r", encoding="utf-8") as f:
        all_test_queries = json.load(f)

    # 3. Handle OpenAQ Station Metadata
    selected_location_id = None
    selected_station_name = "Karachi Metropolitan Station"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            selected_location_id = meta.get("selected_location_id")
            selected_station_name = meta.get("selected_station_name", selected_station_name)

    if not selected_location_id and openaq_client.has_api_key:
        print("\n[OpenAQ] Discovering monitoring stations around Karachi (radius 25 km)...")
        locations, err = openaq_client.discover_karachi_locations()
        if not err and locations:
            # Search for nearest station that provides both PM2.5 and PM10
            best_loc = None
            for loc in locations:
                params_at_loc = set()
                for s in loc.get("sensors", []):
                    p = s.get("parameter", {})
                    pname = (
                        p.get("name", "").lower()
                        if isinstance(p, dict)
                        else str(s.get("parameter", "")).lower()
                    )
                    params_at_loc.add(pname)
                if any("pm25" in x or "pm2.5" in x for x in params_at_loc) and any("pm10" in x for x in params_at_loc):
                    best_loc = loc
                    break
            if not best_loc:
                best_loc = locations[0]

            selected_location_id = best_loc.get("id")
            selected_station_name = best_loc.get("name", "Karachi OpenAQ Station")
            meta_payload = {
                "selected_location_id": selected_location_id,
                "selected_station_name": selected_station_name,
                "coordinates": {"latitude": config["geo"]["latitude"], "longitude": config["geo"]["longitude"]},
                "discovered_locations_count": len(locations),
                "timestamp": datetime.now(tz).isoformat(),
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(meta_payload, f, indent=2)
            print(f"[OpenAQ] Selected Station: {selected_station_name} (ID: {selected_location_id})")

    if selected_location_id and openaq_client.has_api_key:
        openaq_client.load_sensor_cache(selected_location_id)

    # 4. Warm-up Phase (if first run of the day)
    warmup_runs = config["experiment_runner"]["warmup_runs"]
    if sum(counts_today.values()) == 0:
        print(f"\n[Warm-up] Executing {warmup_runs} unmeasured warm-up iterations...")
        for i in range(1, warmup_runs + 1):
            w_start = datetime.now(tz).isoformat()
            # Weather warm-up
            w_obs = meteo_client.fetch_current_weather()
            w_end = datetime.now(tz).isoformat()
            logger.log_record({
                "run_id": run_id,
                "sample_id": f"WARMUP_METEO_{local_date_str}_{i:02d}",
                "local_date": local_date_str,
                "started_at": w_start,
                "completed_at": w_end,
                "timezone": tz_str,
                "tool_name": "Open-Meteo",
                "sample_role": "warmup",
                "sample_number": i,
                "attempt_number": 1,
                "success": w_obs.success,
                "http_status": w_obs.status_code,
                "elapsed_ms": w_obs.elapsed_ms,
                "json_parse_ms": w_obs.json_parse_ms,
                "temperature_c": w_obs.temperature_c,
                "relative_humidity_pct": w_obs.relative_humidity_pct,
                "wind_speed": w_obs.wind_speed,
                "wind_speed_unit": w_obs.wind_speed_unit,
                "request_url": w_obs.request_url,
            })

            # Heat index warm-up
            hi_start = datetime.now(tz).isoformat()
            hi_res, hi_ms = timed_calculate_heat_index(30.0, 70.0)
            hi_end = datetime.now(tz).isoformat()
            logger.log_record({
                "run_id": run_id,
                "sample_id": f"WARMUP_HI_{local_date_str}_{i:02d}",
                "local_date": local_date_str,
                "started_at": hi_start,
                "completed_at": hi_end,
                "timezone": tz_str,
                "tool_name": "HeatIndex",
                "sample_role": "warmup",
                "sample_number": i,
                "attempt_number": 1,
                "success": True,
                "elapsed_ms": hi_ms,
                "temperature_c": 30.0,
                "relative_humidity_pct": 70.0,
                "heat_index_c": hi_res.heat_index_c,
                "heat_index_f": hi_res.heat_index_f,
            })

            # KB warm-up
            kb_start = datetime.now(tz).isoformat()
            kb_res = kb_service.query("WARMUP_Q", "Karachi heatwave precautions")
            kb_end = datetime.now(tz).isoformat()
            logger.log_record({
                "run_id": run_id,
                "sample_id": f"WARMUP_KB_{local_date_str}_{i:02d}",
                "local_date": local_date_str,
                "started_at": kb_start,
                "completed_at": kb_end,
                "timezone": tz_str,
                "tool_name": "KnowledgeBase",
                "sample_role": "warmup",
                "sample_number": i,
                "attempt_number": 1,
                "success": True,
                "elapsed_ms": kb_res.elapsed_ms,
                "kb_query_id": "WARMUP_Q",
                "kb_result_count": len(kb_res.hits),
            })
            time.sleep(0.5)

    # 5. Execute Open-Meteo Weather Calls
    needed_meteo = max(0, daily_target - counts_today.get("Open-Meteo", 0))
    recorded_weather_inputs: List[Tuple[float, float]] = []

    if needed_meteo > 0:
        print(f"\n[Open-Meteo] Collecting {needed_meteo} measured samples for {local_date_str}...")
        for idx in range(1, needed_meteo + 1):
            sample_num = counts_today.get("Open-Meteo", 0) + idx
            sid = f"SAMPLE_METEO_{local_date_str}_{sample_num:02d}"
            t_start = datetime.now(tz).isoformat()
            obs = meteo_client.fetch_current_weather()
            t_end = datetime.now(tz).isoformat()

            logger.log_record({
                "run_id": run_id,
                "sample_id": sid,
                "local_date": local_date_str,
                "started_at": t_start,
                "completed_at": t_end,
                "timezone": tz_str,
                "tool_name": "Open-Meteo",
                "sample_role": "measured",
                "sample_number": sample_num,
                "attempt_number": 1,
                "success": obs.success,
                "http_status": obs.status_code,
                "elapsed_ms": obs.elapsed_ms,
                "json_parse_ms": obs.json_parse_ms,
                "error_category": obs.error_category,
                "error_message": obs.error_message,
                "request_url": obs.request_url,
                "temperature_c": obs.temperature_c,
                "relative_humidity_pct": obs.relative_humidity_pct,
                "wind_speed": obs.wind_speed,
                "wind_speed_unit": obs.wind_speed_unit,
                "extra_metadata": f"observed_at={obs.observed_at}",
            })

            if obs.success and obs.temperature_c is not None and obs.relative_humidity_pct is not None:
                recorded_weather_inputs.append((obs.temperature_c, obs.relative_humidity_pct))
                print(f"  [Sample {sample_num:02d}/10] HTTP {obs.status_code} | {obs.elapsed_ms:.2f} ms | Temp: {obs.temperature_c} C, RH: {obs.relative_humidity_pct}%")
            else:
                print(f"  [Sample {sample_num:02d}/10] FAILED: {obs.error_category} - {obs.error_message}")

            time.sleep(config["experiment_runner"]["inter_request_sleep_sec"])

    # 6. Execute Local Heat Index Calculations using Real Weather Readings
    needed_hi = max(0, daily_target - counts_today.get("HeatIndex", 0))
    if needed_hi > 0:
        print(f"\n[Heat Index] Computing {needed_hi} measured local calculations for {local_date_str}...")
        # If we didn't collect weather inputs in this session, fetch one live reading or read from logged weather
        if not recorded_weather_inputs:
            live_w = meteo_client.fetch_current_weather()
            if live_w.success and live_w.temperature_c is not None and live_w.relative_humidity_pct is not None:
                recorded_weather_inputs = [(live_w.temperature_c, live_w.relative_humidity_pct)] * needed_hi
            else:
                # Default to Karachi historical November average: 26.5 C, 80% RH
                recorded_weather_inputs = [(26.5, 80.0)] * needed_hi

        for idx in range(1, needed_hi + 1):
            sample_num = counts_today.get("HeatIndex", 0) + idx
            sid = f"SAMPLE_HI_{local_date_str}_{sample_num:02d}"
            # Cycle through collected weather readings
            t_c, rh_p = recorded_weather_inputs[(idx - 1) % len(recorded_weather_inputs)]

            t_start = datetime.now(tz).isoformat()
            hi_res, elapsed_ms = timed_calculate_heat_index(t_c, rh_p)
            t_end = datetime.now(tz).isoformat()

            logger.log_record({
                "run_id": run_id,
                "sample_id": sid,
                "local_date": local_date_str,
                "started_at": t_start,
                "completed_at": t_end,
                "timezone": tz_str,
                "tool_name": "HeatIndex",
                "sample_role": "measured",
                "sample_number": sample_num,
                "attempt_number": 1,
                "success": True,
                "elapsed_ms": elapsed_ms,
                "temperature_c": t_c,
                "relative_humidity_pct": rh_p,
                "heat_index_c": hi_res.heat_index_c,
                "heat_index_f": hi_res.heat_index_f,
                "extra_metadata": f"branch={hi_res.formula_branch},adj={hi_res.adjustment_applied}",
            })
            print(f"  [Sample {sample_num:02d}/10] HI: {hi_res.heat_index_c:.2f} C ({hi_res.heat_index_f:.2f} F) | {elapsed_ms:.4f} ms")

    # 7. Execute Local Knowledge-Base Queries
    needed_kb = max(0, daily_target - counts_today.get("KnowledgeBase", 0))
    if needed_kb > 0:
        print(f"\n[Knowledge Base] Executing {needed_kb} measured local BM25 queries for {local_date_str}...")
        # Determine query slice based on current day
        if current_day_number == 1:
            day_queries = [q for q in all_test_queries if q.get("day") == 1]
        else:
            day_queries = [q for q in all_test_queries if q.get("day") == 2]

        for idx in range(1, needed_kb + 1):
            sample_num = counts_today.get("KnowledgeBase", 0) + idx
            sid = f"SAMPLE_KB_{local_date_str}_{sample_num:02d}"
            q_item = day_queries[(idx - 1) % len(day_queries)]
            qid = q_item["query_id"]
            qtext = q_item["query_text"]

            t_start = datetime.now(tz).isoformat()
            kb_res = kb_service.query(qid, qtext, top_k=3)
            t_end = datetime.now(tz).isoformat()

            doc_ids_str = ";".join([h.doc_id for h in kb_res.hits])
            logger.log_record({
                "run_id": run_id,
                "sample_id": sid,
                "local_date": local_date_str,
                "started_at": t_start,
                "completed_at": t_end,
                "timezone": tz_str,
                "tool_name": "KnowledgeBase",
                "sample_role": "measured",
                "sample_number": sample_num,
                "attempt_number": 1,
                "success": True,
                "elapsed_ms": kb_res.elapsed_ms,
                "kb_query_id": qid,
                "kb_result_count": len(kb_res.hits),
                "kb_document_ids": doc_ids_str,
                "extra_metadata": f"lang={q_item.get('language')},cat={q_item.get('category')}",
            })
            print(f"  [Sample {sample_num:02d}/10] Query: {qid} | Hits: {len(kb_res.hits)} | {kb_res.elapsed_ms:.4f} ms")

    # 8. Execute OpenAQ Air-Quality Calls
    needed_openaq = max(0, daily_target - counts_today.get("OpenAQ", 0))
    if needed_openaq > 0:
        if not openaq_client.has_api_key:
            print(f"\n[OpenAQ] WARNING: OPENAQ_API_KEY is not configured in .env.")
            print(f"  OpenAQ API v3 requires an API key in the X-API-Key request header.")
            print(f"  Per instructions, mock data is strictly forbidden. Recording failure state in audit log...")
            existing_sids = {r.get("sample_id") for r in logger.read_all_records()}
            for idx in range(1, needed_openaq + 1):
                sample_num = counts_today.get("OpenAQ", 0) + idx
                base_sid = f"SAMPLE_AQ_{local_date_str}_{sample_num:02d}"
                sid = base_sid
                att = 2
                while sid in existing_sids:
                    sid = f"{base_sid}_att{att}"
                    att += 1
                existing_sids.add(sid)

                t_start = datetime.now(tz).isoformat()
                obs = openaq_client.fetch_latest_measurements(location_id=0, station_name="Karachi Station")
                t_end = datetime.now(tz).isoformat()
                logger.log_record({
                    "run_id": run_id,
                    "sample_id": sid,
                    "local_date": local_date_str,
                    "started_at": t_start,
                    "completed_at": t_end,
                    "timezone": tz_str,
                    "tool_name": "OpenAQ",
                    "sample_role": "measured",
                    "sample_number": sample_num,
                    "attempt_number": 1,
                    "success": False,
                    "http_status": None,
                    "elapsed_ms": 0.0,
                    "json_parse_ms": 0.0,
                    "error_category": obs.error_category,
                    "error_message": obs.error_message,
                    "request_url": obs.request_url,
                })
        else:
            # Check if OpenAQ warm-up is needed
            existing_records = logger.read_all_records()
            openaq_warmups_done = any(
                r.get("tool_name") == "OpenAQ"
                and r.get("sample_role") == "warmup"
                and str(r.get("success", "")).lower() == "true"
                for r in existing_records
            )
            if not openaq_warmups_done:
                print(f"\n[OpenAQ Warm-up] Executing {warmup_runs} unmeasured warm-up iterations for OpenAQ...")
                for i in range(1, warmup_runs + 1):
                    w_start = datetime.now(tz).isoformat()
                    w_obs = openaq_client.fetch_latest_measurements(
                        location_id=selected_location_id, station_name=selected_station_name
                    )
                    w_end = datetime.now(tz).isoformat()
                    logger.log_record({
                        "run_id": run_id,
                        "sample_id": f"WARMUP_AQ_{local_date_str}_{i:02d}",
                        "local_date": local_date_str,
                        "started_at": w_start,
                        "completed_at": w_end,
                        "timezone": tz_str,
                        "tool_name": "OpenAQ",
                        "sample_role": "warmup",
                        "sample_number": i,
                        "attempt_number": 1,
                        "success": w_obs.success,
                        "http_status": w_obs.status_code,
                        "elapsed_ms": w_obs.elapsed_ms,
                        "json_parse_ms": w_obs.json_parse_ms,
                        "location_id": w_obs.location_id,
                        "station_name": w_obs.station_name,
                        "pm25_value": w_obs.pm25_value,
                        "pm25_unit": w_obs.pm25_unit,
                        "pm25_observed_at": w_obs.pm25_observed_at,
                        "pm10_value": w_obs.pm10_value,
                        "pm10_unit": w_obs.pm10_unit,
                        "pm10_observed_at": w_obs.pm10_observed_at,
                        "request_url": w_obs.request_url,
                    })
                    time.sleep(0.5)

            print(f"\n[OpenAQ] Collecting {needed_openaq} measured samples for {local_date_str} from Location ID {selected_location_id} ({selected_station_name})...")
            existing_sids = {r.get("sample_id") for r in logger.read_all_records()}
            for idx in range(1, needed_openaq + 1):
                sample_num = counts_today.get("OpenAQ", 0) + idx
                base_sid = f"SAMPLE_AQ_{local_date_str}_{sample_num:02d}"
                sid = base_sid
                att = 2
                while sid in existing_sids:
                    sid = f"{base_sid}_att{att}"
                    att += 1
                existing_sids.add(sid)

                t_start = datetime.now(tz).isoformat()
                obs = openaq_client.fetch_latest_measurements(
                    location_id=selected_location_id, station_name=selected_station_name
                )
                t_end = datetime.now(tz).isoformat()

                logger.log_record({
                    "run_id": run_id,
                    "sample_id": sid,
                    "local_date": local_date_str,
                    "started_at": t_start,
                    "completed_at": t_end,
                    "timezone": tz_str,
                    "tool_name": "OpenAQ",
                    "sample_role": "measured",
                    "sample_number": sample_num,
                    "attempt_number": 1,
                    "success": obs.success,
                    "http_status": obs.status_code,
                    "elapsed_ms": obs.elapsed_ms,
                    "json_parse_ms": obs.json_parse_ms,
                    "error_category": obs.error_category,
                    "error_message": obs.error_message,
                    "request_url": obs.request_url,
                    "location_id": obs.location_id,
                    "station_name": obs.station_name,
                    "pm25_value": obs.pm25_value,
                    "pm25_unit": obs.pm25_unit,
                    "pm25_observed_at": obs.pm25_observed_at,
                    "pm10_value": obs.pm10_value,
                    "pm10_unit": obs.pm10_unit,
                    "pm10_observed_at": obs.pm10_observed_at,
                })
                if obs.success:
                    print(f"  [Sample {sample_num:02d}/10] HTTP {obs.status_code} | {obs.elapsed_ms:.2f} ms | PM2.5: {obs.pm25_value} {obs.pm25_unit}, PM10: {obs.pm10_value} {obs.pm10_unit}")
                else:
                    print(f"  [Sample {sample_num:02d}/10] FAILED: {obs.error_category} - {obs.error_message}")
                time.sleep(config["experiment_runner"]["inter_request_sleep_sec"])

    # 9. Read all records and perform full analysis
    records = logger.read_all_records()
    validation = analyzer.validate_dataset(records)
    tool_stats = analyzer.compute_tool_statistics(records)
    pooled_ms, pooled_sec, pooled_count, all_tools_rep = analyzer.compute_pooled_median(records)
    model_evals = analyzer.evaluate_model_candidates(pooled_sec, tool_stats)

    # 10. Generate deliverables
    json_path = report_gen.generate_json_summary(
        validation, tool_stats, pooled_ms, pooled_sec, pooled_count, all_tools_rep, model_evals, records
    )
    md_path = report_gen.generate_markdown_report(
        validation, tool_stats, pooled_ms, pooled_sec, pooled_count, all_tools_rep, model_evals, records
    )
    xlsx_path = report_gen.generate_excel_workbook(
        validation, tool_stats, pooled_ms, pooled_sec, all_tools_rep, model_evals, records
    )
    chart_path = report_gen.generate_latency_chart(tool_stats, pooled_ms)

    # 11. Final Console Status Report
    print("\n" + "=" * 78)
    print(f"  EXPERIMENT STATUS: {validation.status_label}")
    print("=" * 78)
    print("\nMeasurement Results Table:")
    print(f"{'Tool':<16} | {'Samples':<8} | {'Mean (ms)':<10} | {'Median (ms)':<12} | {'P95 (ms)':<10} | {'Median (s)':<10}")
    print("-" * 74)
    for t_name in ["Open-Meteo", "HeatIndex", "KnowledgeBase", "OpenAQ"]:
        ts = tool_stats.get(t_name)
        if ts and ts.sample_count > 0:
            print(f"{t_name:<16} | {ts.sample_count:<8} | {ts.mean_ms:<10.2f} | {ts.median_ms:<12.2f} | {ts.p95_ms:<10.2f} | {ts.median_sec:<10.4f}")
        else:
            print(f"{t_name:<16} | {'0 (Pending)':<8} | {'N/A':<10} | {'N/A':<12} | {'N/A':<10} | {'N/A':<10}")

    print("-" * 74)
    print(f"OVERALL POOLED MEDIAN TOOL LATENCY (d_tool) = {pooled_ms:.3f} ms = {pooled_sec:.4f} seconds")
    print(f"(Derived from {pooled_count} valid individual measurements across tools)")
    print("-" * 74)

    print("\nGenerated Deliverables:")
    print(f"  - Raw CSV Log:        {csv_path}")
    print(f"  - JSON Summary:       {json_path}")
    print(f"  - Markdown Report:    {md_path}")
    print(f"  - Excel Workbook:     {xlsx_path}")
    print(f"  - Comparison Chart:   {chart_path}")

    return {
        "status": validation.status_label,
        "is_complete": validation.is_complete,
        "pooled_median_ms": pooled_ms,
        "pooled_median_sec": pooled_sec,
        "csv_path": csv_path,
        "json_path": json_path,
        "md_path": md_path,
        "xlsx_path": xlsx_path,
        "chart_path": chart_path,
    }


if __name__ == "__main__":
    run_experiment()
