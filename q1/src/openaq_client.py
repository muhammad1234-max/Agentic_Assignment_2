"""OpenAQ API v3 Client for SaafHawa Agent Latency Experiment.

Implements OpenAQ v3 location discovery, sensor parameter inspection, and timed
measurement retrieval for PM2.5 and PM10 in the Karachi metropolitan area.
"""

from dataclasses import dataclass
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import requests


@dataclass
class AirQualityObservation:
    success: bool
    status_code: Optional[int]
    elapsed_ms: float
    json_parse_ms: float
    location_id: Optional[int]
    station_name: Optional[str]
    pm25_value: Optional[float]
    pm25_unit: Optional[str]
    pm25_observed_at: Optional[str]
    pm10_value: Optional[float]
    pm10_unit: Optional[str]
    pm10_observed_at: Optional[str]
    request_url: str
    error_category: Optional[str] = None
    error_message: Optional[str] = None


class OpenAQClient:
    """Client for OpenAQ API v3 with authentication and sensor discovery."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openaq.org/v3",
        latitude: float = 24.8607,
        longitude: float = 67.0011,
        search_radius_meters: int = 25000,
        timeout_sec: float = 15.0,
    ):
        self.api_key = (api_key or "").strip()
        self.base_url = base_url.rstrip("/")
        self.latitude = latitude
        self.longitude = longitude
        self.search_radius_meters = search_radius_meters
        self.timeout_sec = timeout_sec
        self.session = requests.Session()
        self.sensor_param_cache: Dict[int, Dict[str, str]] = {}
        headers = {
            "User-Agent": "SaafHawa-LatencyProbe/1.0 (FAST-NUCES 23K-0068)",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        self.session.headers.update(headers)

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def load_sensor_cache(self, location_id: int) -> Dict[int, Dict[str, str]]:
        """Fetch sensors for a location to map sensorsId to parameter name and units."""
        if not self.has_api_key:
            return {}
        try:
            url = f"{self.base_url}/locations/{location_id}/sensors"
            resp = self.session.get(url, timeout=self.timeout_sec)
            if resp.status_code == 200:
                data = resp.json()
                for s in data.get("results", []):
                    sid = s.get("id")
                    param = s.get("parameter", {})
                    pname = (
                        param.get("name", "").lower()
                        if isinstance(param, dict)
                        else str(s.get("parameter", "")).lower()
                    )
                    punits = (
                        param.get("units", "µg/m³")
                        if isinstance(param, dict)
                        else s.get("unit", "µg/m³")
                    )
                    self.sensor_param_cache[sid] = {"name": pname, "units": punits}
        except Exception:
            pass
        return self.sensor_param_cache

    def discover_karachi_locations(self) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Query /v3/locations to discover active monitoring stations around Karachi."""
        if not self.has_api_key:
            return [], "OPENAQ_API_KEY is not configured in .env"

        url = f"{self.base_url}/locations"
        params = {
            "coordinates": f"{self.latitude},{self.longitude}",
            "radius": self.search_radius_meters,
            "limit": 20,
        }
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout_sec)
            if resp.status_code != 200:
                return [], f"HTTP {resp.status_code}: {resp.text[:200]}"
            data = resp.json()
            results = data.get("results", [])
            return results, None
        except Exception as e:
            return [], str(e)

    def fetch_latest_measurements(
        self, location_id: int, station_name: Optional[str] = None
    ) -> AirQualityObservation:
        """Fetch latest air quality readings for a specific OpenAQ v3 location.

        Measures network request duration and JSON parsing time separately using perf_counter_ns().
        """
        redacted_url = f"{self.base_url}/locations/{location_id}/latest"

        if not self.has_api_key:
            return AirQualityObservation(
                success=False,
                status_code=None,
                elapsed_ms=0.0,
                json_parse_ms=0.0,
                location_id=location_id,
                station_name=station_name,
                pm25_value=None,
                pm25_unit=None,
                pm25_observed_at=None,
                pm10_value=None,
                pm10_unit=None,
                pm10_observed_at=None,
                request_url=redacted_url,
                error_category="MISSING_API_KEY",
                error_message="OpenAQ API key not found in environment (.env). API v3 requires X-API-Key header.",
            )

        start_ns = time.perf_counter_ns()
        try:
            resp = self.session.get(redacted_url, timeout=self.timeout_sec)
            http_done_ns = time.perf_counter_ns()
            total_http_ms = (http_done_ns - start_ns) / 1_000_000.0

            if resp.status_code != 200:
                return AirQualityObservation(
                    success=False,
                    status_code=resp.status_code,
                    elapsed_ms=round(total_http_ms, 3),
                    json_parse_ms=0.0,
                    location_id=location_id,
                    station_name=station_name,
                    pm25_value=None,
                    pm25_unit=None,
                    pm25_observed_at=None,
                    pm10_value=None,
                    pm10_unit=None,
                    pm10_observed_at=None,
                    request_url=redacted_url,
                    error_category="HTTP_ERROR",
                    error_message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )

            raw_text = resp.text
            json_start_ns = time.perf_counter_ns()
            data = json.loads(raw_text)
            json_done_ns = time.perf_counter_ns()

            total_elapsed_ms = (json_done_ns - start_ns) / 1_000_000.0
            json_parse_ms = (json_done_ns - json_start_ns) / 1_000_000.0

            results = data.get("results", [])
            pm25_val = None
            pm25_unit = None
            pm25_time = None
            pm10_val = None
            pm10_unit = None
            pm10_time = None

            for item in results:
                # OpenAQ v3 latest payload structure inspection
                param = item.get("parameter", {})
                param_name = (
                    param.get("name", "").lower()
                    if isinstance(param, dict)
                    else str(item.get("parameter", "")).lower()
                )
                unit = (
                    param.get("units")
                    if isinstance(param, dict)
                    else item.get("unit", "µg/m³")
                )

                # If parameter name is missing in latest payload, look up by sensorsId
                sensors_id = item.get("sensorsId")
                if (not param_name or param_name == "none") and sensors_id in self.sensor_param_cache:
                    cached_info = self.sensor_param_cache[sensors_id]
                    param_name = cached_info.get("name", "").lower()
                    unit = cached_info.get("units", unit)

                val = item.get("value")
                period = item.get("period", {})
                obs_dt = (
                    period.get("datetimeTo", {}).get("utc")
                    if isinstance(period, dict)
                    else item.get("datetime", {}).get("utc")
                )

                if "pm25" in param_name or param_name == "pm2.5":
                    pm25_val = float(val) if val is not None else None
                    pm25_unit = unit or "µg/m³"
                    pm25_time = obs_dt
                elif "pm10" in param_name:
                    pm10_val = float(val) if val is not None else None
                    pm10_unit = unit or "µg/m³"
                    pm10_time = obs_dt

            # Success requires at least valid response; check whether pollutants are populated
            if pm25_val is None and pm10_val is None:
                return AirQualityObservation(
                    success=False,
                    status_code=resp.status_code,
                    elapsed_ms=round(total_elapsed_ms, 3),
                    json_parse_ms=round(json_parse_ms, 3),
                    location_id=location_id,
                    station_name=station_name,
                    pm25_value=None,
                    pm25_unit=None,
                    pm25_observed_at=None,
                    pm10_value=None,
                    pm10_unit=None,
                    pm10_observed_at=None,
                    request_url=redacted_url,
                    error_category="POLLUTANT_NOT_FOUND",
                    error_message=f"No PM2.5 or PM10 sensors returned data for location {location_id}.",
                )

            return AirQualityObservation(
                success=True,
                status_code=resp.status_code,
                elapsed_ms=round(total_elapsed_ms, 3),
                json_parse_ms=round(json_parse_ms, 3),
                location_id=location_id,
                station_name=station_name,
                pm25_value=pm25_val,
                pm25_unit=pm25_unit,
                pm25_observed_at=pm25_time,
                pm10_value=pm10_val,
                pm10_unit=pm10_unit,
                pm10_observed_at=pm10_time,
                request_url=redacted_url,
                error_category=None,
                error_message=None,
            )

        except requests.exceptions.Timeout as e:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            return AirQualityObservation(
                success=False,
                status_code=None,
                elapsed_ms=round(elapsed_ms, 3),
                json_parse_ms=0.0,
                location_id=location_id,
                station_name=station_name,
                pm25_value=None,
                pm25_unit=None,
                pm25_observed_at=None,
                pm10_value=None,
                pm10_unit=None,
                pm10_observed_at=None,
                request_url=redacted_url,
                error_category="TIMEOUT",
                error_message=str(e),
            )
        except requests.exceptions.RequestException as e:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            return AirQualityObservation(
                success=False,
                status_code=getattr(e.response, "status_code", None),
                elapsed_ms=round(elapsed_ms, 3),
                json_parse_ms=0.0,
                location_id=location_id,
                station_name=station_name,
                pm25_value=None,
                pm25_unit=None,
                pm25_observed_at=None,
                pm10_value=None,
                pm10_unit=None,
                pm10_observed_at=None,
                request_url=redacted_url,
                error_category="NETWORK_ERROR",
                error_message=str(e),
            )
