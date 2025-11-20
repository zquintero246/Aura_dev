import axios from 'axios';

const CN = axios.create({
  baseURL: 'https://countriesnow.space/api/v0.1',
  timeout: 10000,
});

export type LatamCountry = { code: string; name: string };

// ISO2 list for Latin America + Caribbean commonly used in domotics contexts
const LATAM_ISO2 = new Set([
  'AR','BO','BR','CL','CO','CR','CU','DO','EC','SV','GT','HN','MX','NI','PA','PY','PE','PR','UY','VE','BZ','GY','SR','TT'
]);

let cacheCountries: LatamCountry[] | null = null;
const cacheCities = new Map<string, string[]>(); // key: country name
const cacheStates = new Map<string, string[]>(); // key: country name
const cacheStateCities = new Map<string, string[]>(); // key: `${country}|${state}`

export async function getLatamCountries(): Promise<LatamCountry[]> {
  if (cacheCountries) return cacheCountries;
  try {
    const res = await CN.get('/countries');
    const data = (res.data?.data || []) as any[];
    const items = data
      .filter((c) => LATAM_ISO2.has(String(c.iso2).toUpperCase()))
      .map((c) => ({ code: String(c.iso2).toUpperCase(), name: String(c.country) }));
    // Sort by Spanish-like order (fallback to name)
    items.sort((a, b) => a.name.localeCompare(b.name, 'es'));
    cacheCountries = items;
    return items;
  } catch {
    // Minimal fallback
    cacheCountries = [
      { code: 'AR', name: 'Argentina' },
      { code: 'BO', name: 'Bolivia' },
      { code: 'BR', name: 'Brazil' },
      { code: 'CL', name: 'Chile' },
      { code: 'CO', name: 'Colombia' },
      { code: 'CR', name: 'Costa Rica' },
      { code: 'CU', name: 'Cuba' },
      { code: 'DO', name: 'Dominican Republic' },
      { code: 'EC', name: 'Ecuador' },
      { code: 'SV', name: 'El Salvador' },
      { code: 'GT', name: 'Guatemala' },
      { code: 'HN', name: 'Honduras' },
      { code: 'MX', name: 'Mexico' },
      { code: 'NI', name: 'Nicaragua' },
      { code: 'PA', name: 'Panama' },
      { code: 'PY', name: 'Paraguay' },
      { code: 'PE', name: 'Peru' },
      { code: 'PR', name: 'Puerto Rico' },
      { code: 'UY', name: 'Uruguay' },
      { code: 'VE', name: 'Venezuela' },
    ];
    return cacheCountries;
  }
}

export async function getCitiesByCountry(countryName: string): Promise<string[]> {
  if (!countryName) return [];
  const key = countryName;
  if (cacheCities.has(key)) return cacheCities.get(key)!;
  try {
    const res = await CN.post('/countries/cities', { country: countryName });
    const cities = (res.data?.data || []) as string[];
    const sorted = cities.slice().sort((a, b) => a.localeCompare(b, 'es'));
    cacheCities.set(key, sorted);
    return sorted;
  } catch {
    return [];
  }
}

export async function getStatesByCountry(countryName: string): Promise<string[]> {
  if (!countryName) return [];
  if (cacheStates.has(countryName)) return cacheStates.get(countryName)!;
  try {
    const res = await CN.post('/countries/states', { country: countryName });
    const states = ((res.data?.data?.states as any[]) || []).map((s: any) => String(s?.name || ''))
      .filter(Boolean)
      .sort((a: string, b: string) => a.localeCompare(b, 'es'));
    cacheStates.set(countryName, states);
    return states;
  } catch {
    return [];
  }
}

export async function getCitiesByState(countryName: string, stateName: string): Promise<string[]> {
  if (!countryName || !stateName) return [];
  const key = `${countryName}|${stateName}`;
  if (cacheStateCities.has(key)) return cacheStateCities.get(key)!;
  try {
    const res = await CN.post('/countries/state/cities', { country: countryName, state: stateName });
    const cities = (res.data?.data || []) as string[];
    const sorted = cities.slice().sort((a, b) => a.localeCompare(b, 'es'));
    cacheStateCities.set(key, sorted);
    return sorted;
  } catch {
    return [];
  }
}

export type Position = { lat: number; lon: number; source: 'countriesnow' | 'open-meteo' } | null;

export async function getPositionByCity(countryName: string, city: string, countryIso2?: string): Promise<Position> {
  if (!countryName || !city) return null;

  // 1) Primary: CountriesNow (POST)
  try {
    const res = await CN.post('/countries/positions/q', { country: countryName, city });
    const d = res.data?.data;
    if (d) {
      // Object shape
      const lat = Number(d?.lat ?? d?.latitude ?? d?.Lat ?? d?.LAT);
      const lon = Number(d?.long ?? d?.lng ?? d?.longitude ?? d?.Lon ?? d?.LNG);
      if (isFinite(lat) && isFinite(lon)) return { lat, lon, source: 'countriesnow' };
    }
    // Array shape
    if (Array.isArray(d) && d.length) {
      const match = d.find((v: any) => String(v?.name || v?.city || '').toLowerCase() === city.toLowerCase());
      const cand = match || d[0];
      const la = Number(cand?.lat ?? cand?.latitude);
      const lo = Number(cand?.long ?? cand?.lng ?? cand?.longitude);
      if (isFinite(la) && isFinite(lo)) return { lat: la, lon: lo, source: 'countriesnow' };
    }
  } catch {}

  // 2) Fallback: Open‑Meteo Geocoding (search by city only, filter by country)
  try {
    const qName = encodeURIComponent(city);
    const url = `https://geocoding-api.open-meteo.com/v1/search?name=${qName}&count=10&language=es&format=json`;
    const res = await fetch(url, { method: 'GET' });
    if (res.ok) {
      const j = await res.json();
      const results: any[] = Array.isArray(j?.results) ? j.results : [];
      let r: any = null;
      if (countryIso2) {
        r = results.find((it) => String(it?.country_code || '').toUpperCase() === String(countryIso2).toUpperCase());
      }
      if (!r) {
        r = results.find((it) => String(it?.country || '').toLowerCase() === countryName.toLowerCase());
      }
      if (!r) {
        r = results[0] || null;
      }
      const la = Number(r?.latitude);
      const lo = Number(r?.longitude);
      if (isFinite(la) && isFinite(lo)) return { lat: la, lon: lo, source: 'open-meteo' };
    }
  } catch {}

  return null;
}
