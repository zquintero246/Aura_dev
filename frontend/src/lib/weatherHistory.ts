export type HistoryPoint = { time: string; temperature?: number; humidity?: number; code?: number };

export async function fetchWeatherHistory(lat: number, lon: number) {
  const url = new URL('https://api.open-meteo.com/v1/forecast');
  url.searchParams.set('latitude', String(lat));
  url.searchParams.set('longitude', String(lon));
  url.searchParams.set('hourly', 'temperature_2m,relative_humidity_2m,weather_code');
  url.searchParams.set('past_days', '1');
  url.searchParams.set('forecast_days', '0');
  url.searchParams.set('timezone', 'auto');

  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`Open-Meteo HTTP ${res.status}`);
  const j = await res.json();
  const h = j?.hourly || {};
  const times: string[] = h.time || [];
  const temps: number[] = h.temperature_2m || [];
  const hums: number[] = h.relative_humidity_2m || [];
  const codes: number[] = h.weather_code || [];

  const data: HistoryPoint[] = times.map((t: string, i: number) => ({
    time: t,
    temperature: Number.isFinite(temps[i]) ? temps[i] : undefined,
    humidity: Number.isFinite(hums[i]) ? hums[i] : undefined,
    code: Number.isFinite(codes[i]) ? codes[i] : undefined,
  }));
  return data;
}

export function wmoToText(code?: number): string {
  if (code == null) return '';
  const map: Record<number, string> = {
    0: 'Clear sky',
    1: 'Mainly clear',
    2: 'Partly cloudy',
    3: 'Overcast',
    45: 'Fog',
    48: 'Rime fog',
    51: 'Drizzle (light)',
    53: 'Drizzle (moderate)',
    55: 'Drizzle (dense)',
    56: 'Freezing drizzle (light)',
    57: 'Freezing drizzle (dense)',
    61: 'Rain (slight)',
    63: 'Rain (moderate)',
    65: 'Rain (heavy)',
    66: 'Freezing rain (light)',
    67: 'Freezing rain (heavy)',
    71: 'Snow fall (slight)',
    73: 'Snow fall (moderate)',
    75: 'Snow fall (heavy)',
    77: 'Snow grains',
    80: 'Rain showers (slight)',
    81: 'Rain showers (moderate)',
    82: 'Rain showers (violent)',
    85: 'Snow showers (slight)',
    86: 'Snow showers (heavy)',
    95: 'Thunderstorm',
    96: 'Thunderstorm (slight hail)',
    99: 'Thunderstorm (heavy hail)',
  };
  return map[code] || String(code);
}

