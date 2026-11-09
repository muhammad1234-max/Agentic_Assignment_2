"""Unit tests for the National Weather Service (NWS) Heat Index algorithm."""

import math
import unittest
from src.heat_index import (
    calculate_heat_index,
    celsius_to_fahrenheit,
    fahrenheit_to_celsius,
    timed_calculate_heat_index,
    validate_weather_inputs,
)


class TestHeatIndex(unittest.TestCase):
    def test_temperature_conversions(self):
        self.assertAlmostEqual(celsius_to_fahrenheit(0.0), 32.0, places=4)
        self.assertAlmostEqual(celsius_to_fahrenheit(100.0), 212.0, places=4)
        self.assertAlmostEqual(celsius_to_fahrenheit(25.0), 77.0, places=4)
        self.assertAlmostEqual(fahrenheit_to_celsius(32.0), 0.0, places=4)
        self.assertAlmostEqual(fahrenheit_to_celsius(212.0), 100.0, places=4)

    def test_input_validation(self):
        # Valid inputs
        validate_weather_inputs(25.0, 50.0)
        validate_weather_inputs(-10.0, 0.0)
        validate_weather_inputs(45.0, 100.0)

        # Invalid temperature
        with self.assertRaises(ValueError):
            validate_weather_inputs(70.0, 50.0)
        with self.assertRaises(ValueError):
            validate_weather_inputs(-60.0, 50.0)
        with self.assertRaises(ValueError):
            validate_weather_inputs(float("nan"), 50.0)

        # Invalid humidity
        with self.assertRaises(ValueError):
            validate_weather_inputs(25.0, -5.0)
        with self.assertRaises(ValueError):
            validate_weather_inputs(25.0, 105.0)

        # Non-numeric
        with self.assertRaises(TypeError):
            validate_weather_inputs("25", 50.0)  # type: ignore

    def test_steadman_simple_branch(self):
        # 20 deg C is 68 deg F -> Simple equation branch
        res = calculate_heat_index(20.0, 50.0)
        self.assertEqual(res.formula_branch, "steadman_simple")
        self.assertEqual(res.adjustment_applied, 0.0)
        # At 68 F and 50% RH: HI should be around 68 F
        self.assertAlmostEqual(res.heat_index_f, 67.85, delta=1.5)

    def test_rothfusz_full_regression(self):
        # 35 deg C = 95 deg F, 50% RH -> standard Rothfusz regression
        res = calculate_heat_index(35.0, 50.0)
        self.assertEqual(res.formula_branch, "rothfusz_full")
        # NOAA heat index chart: 95 F at 50% RH gives ~107 F
        self.assertAlmostEqual(res.heat_index_f, 107.0, delta=2.0)
        self.assertTrue(res.heat_index_c > 35.0)

    def test_rothfusz_low_humidity_adjustment(self):
        # 35 deg C = 95 deg F, 10% RH (< 13% RH) -> negative adjustment applied
        res = calculate_heat_index(35.0, 10.0)
        self.assertEqual(res.formula_branch, "rothfusz_low_rh_adjusted")
        self.assertLess(res.adjustment_applied, 0.0)

    def test_rothfusz_high_humidity_adjustment(self):
        # 28 deg C = 82.4 deg F, 90% RH (> 85% RH and 80 <= T <= 87) -> positive adjustment
        res = calculate_heat_index(28.0, 90.0)
        self.assertEqual(res.formula_branch, "rothfusz_high_rh_adjusted")
        self.assertGreater(res.adjustment_applied, 0.0)

    def test_timed_execution(self):
        res, elapsed_ms = timed_calculate_heat_index(32.0, 65.0)
        self.assertGreater(elapsed_ms, 0.0)
        self.assertLess(elapsed_ms, 100.0)  # Pure local math should take < 1 ms typically
        self.assertIsInstance(res.heat_index_c, float)


if __name__ == "__main__":
    unittest.main()
