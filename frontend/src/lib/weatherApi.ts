// Client for the microservices weather API (Azure Maps based)

const BASE = (import.meta as any)?.env?.VITE_MICROSERVICES_URL || 'http://127.0.0.1:5050';

export type WeatherData = {
  temperature: number | null;
  humidity: number | null;
  condition: string;
  air_quality_index: number | null;
  source: string;
} | { error: string; details?: any };

async function http<T>(url: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const { timeoutMs = 8000, ...opts } = init || {};
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...opts, signal: ctrl.signal });
    if (!res.ok) throw Object.assign(new Error(`HTTP ${res.status}`), { status: res.status });
    return (await res.json()) as T;
  } finally {
    clearTimeout(t);
  }
}

export async function fetchDashboardByUser(userId: string): Promise<WeatherData | null> {
  try {
    return await http<WeatherData>(`${BASE}/api/dashboard?user_id=${encodeURIComponent(userId)}`);
  } catch {
    return null;
  }
}

export async function fetchDashboardByCoords(lat: number, lon: number): Promise<WeatherData | null> {
  try {
    return await http<WeatherData>(`${BASE}/api/dashboard?lat=${encodeURIComponent(String(lat))}&lon=${encodeURIComponent(String(lon))}`);
  } catch {
    return null;
  }
}
