"""Unit tests for Open-Meteo and OpenAQ response parsing with test fixtures."""

import json
import unittest
from unittest.mock import MagicMock, patch
from src.open_meteo_client import OpenMeteoClient, WeatherObservation
from src.openaq_client import OpenAQClient, AirQualityObservation


class TestApiParsing(unittest.TestCase):
    def setUp(self):
        self.meteo_client = OpenMeteoClient()
        self.openaq_client = OpenAQClient(api_key="mock-api-key")

    @patch("requests.Session.get")
    def test_open_meteo_successful_parse(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = json.dumps({
            "latitude": 24.86,
            "longitude": 67.0,
            "current_units": {"temperature_2m": "°C", "relative_humidity_2m": "%", "wind_speed_10m": "km/h"},
            "current": {
                "time": "2026-11-09T19:00",
                "temperature_2m": 27.4,
                "relative_humidity_2m": 78,
                "wind_speed_10m": 8.5
            }
        })
        mock_get.return_value = mock_resp

        obs = self.meteo_client.fetch_current_weather()
        self.assertTrue(obs.success)
        self.assertEqual(obs.status_code, 200)
        self.assertEqual(obs.temperature_c, 27.4)
        self.assertEqual(obs.relative_humidity_pct, 78.0)
        self.assertEqual(obs.wind_speed, 8.5)
        self.assertEqual(obs.wind_speed_unit, "km/h")
        self.assertEqual(obs.observed_at, "2026-11-09T19:00")
        self.assertGreater(obs.elapsed_ms, 0.0)

    @patch("requests.Session.get")
    def test_open_meteo_missing_field(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = json.dumps({
            "current": {"time": "2026-11-09T19:00", "temperature_2m": None, "relative_humidity_2m": 78}
        })
        mock_get.return_value = mock_resp

        obs = self.meteo_client.fetch_current_weather()
        self.assertFalse(obs.success)
        self.assertEqual(obs.error_category, "MISSING_DATA_FIELD")

    @patch("requests.Session.get")
    def test_open_meteo_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_get.return_value = mock_resp

        obs = self.meteo_client.fetch_current_weather()
        self.assertFalse(obs.success)
        self.assertEqual(obs.status_code, 500)
        self.assertEqual(obs.error_category, "HTTP_ERROR")

    @patch("requests.Session.get")
    def test_openaq_successful_parse(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = json.dumps({
            "results": [
                {
                    "parameter": {"name": "pm25", "units": "µg/m³"},
                    "value": 42.8,
                    "period": {"datetimeTo": {"utc": "2026-11-09T13:00:00Z"}}
                },
                {
                    "parameter": {"name": "pm10", "units": "µg/m³"},
                    "value": 98.4,
                    "period": {"datetimeTo": {"utc": "2026-11-09T13:00:00Z"}}
                }
            ]
        })
        mock_get.return_value = mock_resp

        obs = self.openaq_client.fetch_latest_measurements(location_id=1234, station_name="Karachi US Consulate")
        self.assertTrue(obs.success)
        self.assertEqual(obs.pm25_value, 42.8)
        self.assertEqual(obs.pm10_value, 98.4)
        self.assertEqual(obs.pm25_observed_at, "2026-11-09T13:00:00Z")

    def test_openaq_missing_api_key(self):
        client_no_key = OpenAQClient(api_key="")
        obs = client_no_key.fetch_latest_measurements(location_id=1234)
        self.assertFalse(obs.success)
        self.assertEqual(obs.error_category, "MISSING_API_KEY")


if __name__ == "__main__":
    unittest.main()

