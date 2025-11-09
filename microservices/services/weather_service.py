import os
import requests

OPEN_METEO_WX = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_AQI = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Optional fallback (if OPENWEATHERMAP_KEY present)
OWM_WX = "https://api.openweathermap.org/data/2.5/weather"
OWM_AQI = "https://api.openweathermap.org/data/2.5/air_pollution"

WMO_CODE = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _ok(resp) -> bool:
    try:
        return 200 <= resp.status_code < 300
    except Exception:
        return False


def _err(message: str, status: int | None = None, body: str | None = None):
    out = {"error": message}
    details = {}
    if status is not None:
        details["status"] = status
    if body:
        details["body"] = body
    if details:
        out["details"] = details
    return out


def _from_open_meteo(lat: str | float, lon: str | float):
    # Weather
    wx = requests.get(
        OPEN_METEO_WX,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code",
            "timezone": "auto",
        },
        timeout=10,
    )
    if not _ok(wx):
        return _err("Open-Meteo weather error", status=wx.status_code, body=wx.text)
    wj = wx.json() or {}
    cur = (wj.get("current") or {})
    temperature = cur.get("temperature_2m")
    humidity = cur.get("relative_humidity_2m")
    code = cur.get("weather_code")
    condition = WMO_CODE.get(int(code)) if code is not None else ""

    # AQI
    aq = requests.get(
        OPEN_METEO_AQI,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "european_aqi",
            "timezone": "auto",
        },
        timeout=10,
    )
    if not _ok(aq):
        return _err("Open-Meteo air quality error", status=aq.status_code, body=aq.text)
    aj = aq.json() or {}
    aqi = (aj.get("current") or {}).get("european_aqi")

    return {
        "temperature": float(temperature) if temperature is not None else None,
        "humidity": float(humidity) if humidity is not None else None,
        "condition": condition,
        "air_quality_index": float(aqi) if aqi is not None else None,
        "source": "open-meteo",
    }


def _from_openweather(lat: str | float, lon: str | float, key: str):
    wx = requests.get(OWM_WX, params={"lat": lat, "lon": lon, "appid": key, "units": "metric"}, timeout=10)
    if not _ok(wx):
        return _err("OpenWeather weather error", status=wx.status_code, body=wx.text)
    wj = wx.json() or {}
    temperature = (wj.get("main") or {}).get("temp")
    humidity = (wj.get("main") or {}).get("humidity")
    condition = (wj.get("weather") or [{}])[0].get("description", "")

    aq = requests.get(OWM_AQI, params={"lat": lat, "lon": lon, "appid": key}, timeout=10)
    if not _ok(aq):
        return _err("OpenWeather AQI error", status=aq.status_code, body=aq.text)
    aj = aq.json() or {}
    aqi = None
    try:
        aqi = float(((aj.get("list") or [{}])[0].get("main") or {}).get("aqi"))
    except Exception:
        aqi = None

    return {
        "temperature": float(temperature) if temperature is not None else None,
        "humidity": float(humidity) if humidity is not None else None,
        "condition": condition,
        "air_quality_index": aqi,
        "source": "openweathermap",
    }


def get_weather_data(lat: float | str, lon: float | str):
    """Return weather + AQI given coordinates using provider in WEATHER_API_PROVIDER.

    Default provider: open-meteo. Fallback to OpenWeatherMap if key present.
    Returns JSON or error dict with details.
    """
    provider = (os.environ.get("WEATHER_API_PROVIDER") or "open-meteo").strip().lower()
    try:
        if provider == "openweather":
            key = os.environ.get("OPENWEATHERMAP_KEY", "")
            if not key:
                return _err("OPENWEATHERMAP_KEY missing; set provider to open-meteo or provide key")
            return _from_openweather(lat, lon, key)

        # Default: open-meteo
        res = _from_open_meteo(lat, lon)
        if isinstance(res, dict) and res.get("error"):
            # try fallback if key available
            key = os.environ.get("OPENWEATHERMAP_KEY", "")
            if key:
                return _from_openweather(lat, lon, key)
        return res
    except requests.Timeout:
        return _err("Timeout contacting provider", status=503)
    except Exception as e:
        return _err(str(e))

