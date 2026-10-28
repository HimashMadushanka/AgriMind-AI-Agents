import requests


def get_weather_forecast(latitude: float, longitude: float) -> dict:
    """Retrieve basic forecast data from Open-Meteo for user-provided coordinates.

    This is contextual weather information, not a crop-disease diagnosis.
    """
    if not (-90 <= float(latitude) <= 90 and -180 <= float(longitude) <= 180):
        raise ValueError("Latitude/longitude are outside valid ranges.")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": float(latitude), "longitude": float(longitude),
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "forecast_days": 3, "timezone": "auto"
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    return {"timezone": data.get("timezone"), "current": data.get("current", {}), "daily": data.get("daily", {})}
