"""Local Heat-Index calculation implementation based on National Weather Service (NWS) method.

Reference:
National Oceanic and Atmospheric Administration (NOAA) / NWS
https://www.weather.gov/ctp/heat
https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml

Implements:
1. Steadman lower-temperature refinement.
2. Rothfusz 9-parameter polynomial regression for temperatures >= 80 deg F.
3. Low relative humidity (< 13%) correction for 80 <= T <= 112 deg F.
4. High relative humidity (> 85%) correction for 80 <= T <= 87 deg F.
5. High-resolution monotonic timing using time.perf_counter_ns().
"""

from dataclasses import dataclass
import math
import time
from typing import Optional, Tuple


@dataclass(frozen=True)
class HeatIndexResult:
    temperature_c: float
    temperature_f: float
    relative_humidity_pct: float
    heat_index_c: float
    heat_index_f: float
    formula_branch: str
    adjustment_applied: float


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert degrees Celsius to Fahrenheit."""
    return (celsius * 9.0 / 5.0) + 32.0


def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert degrees Fahrenheit to Celsius."""
    return (fahrenheit - 32.0) * 5.0 / 9.0


def validate_weather_inputs(temperature_c: float, relative_humidity_pct: float) -> None:
    """Validate that temperature and relative humidity fall within physically plausible terrestrial limits."""
    if not isinstance(temperature_c, (int, float)) or not isinstance(relative_humidity_pct, (int, float)):
        raise TypeError("Temperature and relative humidity must be numeric (int or float).")
    if math.isnan(temperature_c) or math.isnan(relative_humidity_pct):
        raise ValueError("Temperature or relative humidity is NaN.")
    if not (-50.0 <= temperature_c <= 65.0):
        raise ValueError(f"Temperature {temperature_c} C is outside plausible terrestrial range (-50 to 65 C).")
    if not (0.0 <= relative_humidity_pct <= 100.0):
        raise ValueError(f"Relative humidity {relative_humidity_pct}% is outside valid range (0 to 100%).")


def calculate_heat_index(temperature_c: float, relative_humidity_pct: float) -> HeatIndexResult:
    """Compute the National Weather Service Heat Index from Celsius temperature and RH.

    Args:
        temperature_c: Ambient temperature in degrees Celsius.
        relative_humidity_pct: Relative humidity as a percentage (0 to 100).

    Returns:
        HeatIndexResult containing Fahrenheit and Celsius heat index values,
        the applied branch, and any humidity adjustment applied.
    """
    validate_weather_inputs(temperature_c, relative_humidity_pct)

    t_f = celsius_to_fahrenheit(temperature_c)
    rh = float(relative_humidity_pct)

    # Simple Steadman equation for lower temperatures
    hi_simple = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094))

    # NWS rule: average simple HI with T; if < 80 deg F, simple equation is sufficient
    if ((hi_simple + t_f) / 2.0) < 80.0:
        hi_f = hi_simple
        branch = "steadman_simple"
        adjustment = 0.0
    else:
        # Full Rothfusz regression
        hi_f = (
            -42.379
            + (2.04901523 * t_f)
            + (10.14333127 * rh)
            - (0.22475541 * t_f * rh)
            - (0.00683783 * (t_f**2))
            - (0.05481717 * (rh**2))
            + (0.00122874 * (t_f**2) * rh)
            + (0.00085282 * t_f * (rh**2))
            - (0.00000199 * (t_f**2) * (rh**2))
        )
        branch = "rothfusz_full"
        adjustment = 0.0

        # Low humidity adjustment: RH < 13% and 80 <= T_F <= 112
        if rh < 13.0 and 80.0 <= t_f <= 112.0:
            radicand = (17.0 - abs(t_f - 95.0)) / 17.0
            if radicand > 0:
                adjustment = -((13.0 - rh) / 4.0) * math.sqrt(radicand)
                hi_f += adjustment
                branch = "rothfusz_low_rh_adjusted"

        # High humidity adjustment: RH > 85% and 80 <= T_F <= 87
        elif rh > 85.0 and 80.0 <= t_f <= 87.0:
            adjustment = ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)
            hi_f += adjustment
            branch = "rothfusz_high_rh_adjusted"

    hi_c = fahrenheit_to_celsius(hi_f)

    return HeatIndexResult(
        temperature_c=round(temperature_c, 2),
        temperature_f=round(t_f, 2),
        relative_humidity_pct=round(rh, 2),
        heat_index_c=round(hi_c, 2),
        heat_index_f=round(hi_f, 2),
        formula_branch=branch,
        adjustment_applied=round(adjustment, 4),
    )


def timed_calculate_heat_index(temperature_c: float, relative_humidity_pct: float) -> Tuple[HeatIndexResult, float]:
    """Execute calculate_heat_index while precisely recording elapsed time in milliseconds.

    Uses time.perf_counter_ns() for monotonic sub-microsecond resolution.
    Returns (result, elapsed_ms).
    """
    start_ns = time.perf_counter_ns()
    result = calculate_heat_index(temperature_c, relative_humidity_pct)
    stop_ns = time.perf_counter_ns()

    elapsed_ms = (stop_ns - start_ns) / 1_000_000.0
    return result, elapsed_ms

