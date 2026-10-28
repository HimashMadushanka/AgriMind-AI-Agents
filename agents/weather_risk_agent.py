from tools.get_weather_forecast import get_weather_forecast


def check_weather(latitude: float | None = None, longitude: float | None = None) -> dict:
    if latitude is None or longitude is None:
        return {"available": False, "message": "Weather was not checked because coordinates were not supplied."}
    try:
        return {"available": True, "forecast": get_weather_forecast(latitude, longitude)}
    except Exception as exc:
        return {"available": False, "message": f"Weather service unavailable: {exc}"}
