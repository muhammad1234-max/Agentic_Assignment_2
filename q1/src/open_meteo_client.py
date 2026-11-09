"""Open-Meteo Weather API Client for SaafHawa Agent Latency Experiment.

Fetches current weather observations for Karachi (temperature_2m, relative_humidity_2m, wind_speed_10m)
and measures precise network round-trip and JSON deserialization timings.
"""

from dataclasses import dataclass
import json
import time
from typing import Any, Dict, Optional, Tuple
import requests


@dataclass
class WeatherObservation:
    success: bool
    status_code: Optional[int]
    elapsed_ms: float
    json_parse_ms: float
    temperature_c: Optional[float]
    relative_humidity_pct: Optional[float]
    wind_speed: Optional[float]
    wind_speed_unit: Optional[str]
    observed_at: Optional[str]
    request_url: str
    error_category: Optional[str] = None
    error_message: Optional[str] = None


class OpenMeteoClient:
    """Client for querying Open-Meteo Forecast API."""

    def __init__(
        self,
        base_url: str = "https://api.open-meteo.com/v1/forecast",
        latitude: float = 24.8607,
        longitude: float = 67.0011,
        timezone: str = "Asia/Karachi",
        timeout_sec: float = 15.0,
    ):
        self.base_url = base_url
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.timeout_sec = timeout_sec
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SaafHawa-LatencyProbe/1.0 (FAST-NUCES 23K-0068)"})

    def fetch_current_weather(self) -> WeatherObservation:
        """Fetch current weather with microsecond timing and response validation.

        Measures total elapsed time (network + parse) and JSON parse time separately.
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "timezone": self.timezone,
        }

        # Prepare request URL for audit logging
        prepared = requests.Request("GET", self.base_url, params=params).prepare()
        request_url = prepared.url or self.base_url

        start_ns = time.perf_counter_ns()
        try:
            resp = self.session.get(self.base_url, params=params, timeout=self.timeout_sec)
            http_done_ns = time.perf_counter_ns()
            total_http_ms = (http_done_ns - start_ns) / 1_000_000.0

            if resp.status_code != 200:
                return WeatherObservation(
                    success=False,
                    status_code=resp.status_code,
                    elapsed_ms=round(total_http_ms, 3),
                    json_parse_ms=0.0,
                    temperature_c=None,
                    relative_humidity_pct=None,
                    wind_speed=None,
                    wind_speed_unit=None,
                    observed_at=None,
                    request_url=request_url,
                    error_category="HTTP_ERROR",
                    error_message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )

            # Measure JSON deserialization separately
            raw_text = resp.text
            json_start_ns = time.perf_counter_ns()
            data = json.loads(raw_text)
            json_done_ns = time.perf_counter_ns()

            total_elapsed_ms = (json_done_ns - start_ns) / 1_000_000.0
            json_parse_ms = (json_done_ns - json_start_ns) / 1_000_000.0

            current = data.get("current")
            units = data.get("current_units", {})
            if not current or not isinstance(current, dict):
                return WeatherObservation(
                    success=False,
                    status_code=resp.status_code,
                    elapsed_ms=round(total_elapsed_ms, 3),
                    json_parse_ms=round(json_parse_ms, 3),
                    temperature_c=None,
                    relative_humidity_pct=None,
                    wind_speed=None,
                    wind_speed_unit=None,
                    observed_at=None,
                    request_url=request_url,
                    error_category="SCHEMA_VALIDATION_ERROR",
                    error_message="Missing 'current' payload block in response",
                )

            # Validate numeric variables
            temp = current.get("temperature_2m")
            rh = current.get("relative_humidity_2m")
            wind = current.get("wind_speed_10m")
            obs_time = current.get("time")

            if None in (temp, rh, wind):
                return WeatherObservation(
                    success=False,
                    status_code=resp.status_code,
                    elapsed_ms=round(total_elapsed_ms, 3),
                    json_parse_ms=round(json_parse_ms, 3),
                    temperature_c=temp,
                    relative_humidity_pct=rh,
                    wind_speed=wind,
                    wind_speed_unit=units.get("wind_speed_10m", "km/h"),
                    observed_at=obs_time,
                    request_url=request_url,
                    error_category="MISSING_DATA_FIELD",
                    error_message="One or more weather values (temperature, humidity, wind) were null",
                )

            return WeatherObservation(
                success=True,
                status_code=resp.status_code,
                elapsed_ms=round(total_elapsed_ms, 3),
                json_parse_ms=round(json_parse_ms, 3),
                temperature_c=float(temp),
                relative_humidity_pct=float(rh),
                wind_speed=float(wind),
                wind_speed_unit=units.get("wind_speed_10m", "km/h"),
                observed_at=str(obs_time),
                request_url=request_url,
                error_category=None,
                error_message=None,
            )

        except requests.exceptions.Timeout as e:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            return WeatherObservation(
                success=False,
                status_code=None,
                elapsed_ms=round(elapsed_ms, 3),
                json_parse_ms=0.0,
                temperature_c=None,
                relative_humidity_pct=None,
                wind_speed=None,
                wind_speed_unit=None,
                observed_at=None,
                request_url=request_url,
                error_category="TIMEOUT",
                error_message=str(e),
            )
        except requests.exceptions.RequestException as e:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            return WeatherObservation(
                success=False,
                status_code=getattr(e.response, "status_code", None),
                elapsed_ms=round(elapsed_ms, 3),
                json_parse_ms=0.0,
                temperature_c=None,
                relative_humidity_pct=None,
                wind_speed=None,
                wind_speed_unit=None,
                observed_at=None,
                request_url=request_url,
                error_category="NETWORK_ERROR",
                error_message=str(e),
            )
        except json.JSONDecodeError as e:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            return WeatherObservation(
                success=False,
                status_code=resp.status_code if "resp" in locals() else None,
                elapsed_ms=round(elapsed_ms, 3),
                json_parse_ms=0.0,
                temperature_c=None,
                relative_humidity_pct=None,
                wind_speed=None,
                wind_speed_unit=None,
                observed_at=None,
                request_url=request_url,
                error_category="JSON_DECODE_ERROR",
                error_message=str(e),
            )
