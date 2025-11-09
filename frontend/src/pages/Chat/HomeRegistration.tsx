import React, { useEffect, useMemo, useState } from 'react';
import { saveLocation } from '../../lib/location';
import {
  getLatamCountries,
  getStatesByCountry,
  getCitiesByState,
  getCitiesByCountry,
  getPositionByCity,
  type LatamCountry,
} from '../../lib/countriesNow';
import WorldMap from './components/WorldMap';

type Props = { userId: string; onDone: () => void };

export default function HomeRegistration({ userId, onDone }: Props) {
  const [countries, setCountries] = useState<LatamCountry[]>([]);
  const [countryCode, setCountryCode] = useState<string>('CO');
  const [countryName, setCountryName] = useState<string>('Colombia');
  const [states, setStates] = useState<string[]>([]);
  const [stateName, setStateName] = useState<string>('');
  const [cities, setCities] = useState<string[]>([]);
  const [city, setCity] = useState<string>('Bucaramanga');
  const [coords, setCoords] = useState<{ lat: number; lon: number } | null>(null);
  const [loadingCountries, setLoadingCountries] = useState(false);
  const [loadingStates, setLoadingStates] = useState(false);
  const [loadingCities, setLoadingCities] = useState(false);
  const [loadingPos, setLoadingPos] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  // Helper to display cleaner labels for departments/states
  const prettyState = (s: string) =>
    (s || '')
      .replace(/\s+(Department|State|Province|Region|Governorate|Prefecture|County|District)\s*$/i, '')
      .trim();

  // Cargar países LATAM (cacheados en memoria)
  useEffect(() => {
    (async () => {
      setLoadingCountries(true);
      const cs = await getLatamCountries();
      setCountries(cs);
      const current = cs.find((c) => c.code === countryCode) || cs[0];
      if (current) {
        setCountryCode(current.code);
        setCountryName(current.name);
      }
      setLoadingCountries(false);
    })();
  }, []);

  // Cargar estados/departamentos cuando cambie el país
  useEffect(() => {
    (async () => {
      if (!countryName) return;
      setLoadingStates(true);
      setStates([]);
      setStateName('');
      try {
        const sts = await getStatesByCountry(countryName);
        setStates(sts);
        if (sts.length > 0) {
          setStateName((prev) => (prev && sts.includes(prev) ? prev : sts[0]));
        } else {
          // Si no hay estados, cargar ciudades por país
          setLoadingCities(true);
          const items = await getCitiesByCountry(countryName);
          setCities(items);
          if (!items.includes(city) && items[0]) setCity(items[0]);
          setLoadingCities(false);
        }
      } finally {
        setLoadingStates(false);
      }
    })();
  }, [countryName]);

  // Cargar ciudades cuando cambie el estado
  useEffect(() => {
    (async () => {
      if (!countryName) return;
      if (!states.length) return; // Cuando no hay estados, ya se cargó por país
      setLoadingCities(true);
      const items = await getCitiesByState(countryName, stateName || states[0]);
      setCities(items);
      if (!items.includes(city) && items[0]) setCity(items[0]);
      setLoadingCities(false);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stateName, states.length]);

  // Posición de la ciudad
  useEffect(() => {
    let cancel = false;
    const t = window.setTimeout(async () => {
      try {
        if (!countryName || !city) {
          if (!cancel) setCoords(null);
          if (!cancel) setLoadingPos(false);
          return;
        }
        setLoadingPos(true);
        const p = await getPositionByCity(countryName, city, countryCode);
        if (!cancel) setCoords(p ? { lat: p.lat, lon: p.lon } : null);
      } catch {
        if (!cancel) setCoords(null);
      } finally {
        if (!cancel) setLoadingPos(false);
      }
    }, 300);
    return () => {
      cancel = true;
      window.clearTimeout(t);
    };
  }, [countryName, countryCode, city]);

  const canSubmit = useMemo(() => !!countryCode && !!city && !!coords && !submitting, [countryCode, city, coords, submitting]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    try {
      if (coords) {
        const keyCoords = `aura:home:coords:${userId}`;
        const keyReg = `aura:home:registered:${userId}`;
        localStorage.setItem(keyCoords, JSON.stringify(coords));
        localStorage.setItem(keyReg, '1');
        // Limpia claves antiguas sin scope de usuario (por compatibilidad)
        localStorage.removeItem('home:coords');
        localStorage.removeItem('home:registered');
      }
    } catch {}
    if (coords) {
      await saveLocation({ country: countryName, city, latitude: coords.lat, longitude: coords.lon });
    }
    setSubmitting(false);
    onDone();
  };

  return (
    <div className="w-full h-full grid grid-rows-[auto_1fr_auto]">
      {/* Barra superior */}
      <header className="px-4">
        <div className="mt-1 h-14 rounded-2xl bg-white/5 ring-1 ring-white/10 flex items-center justify-between px-4 text-white/80">
          <div className="font-medium tracking-wide">Configurar mi casa</div>
          <div className="flex items-center gap-3">
            <span className="hidden sm:inline text-sm text-white/70">{city || 'Ciudad'}</span>
            <span className="w-6 h-6 rounded-full bg-white/10" aria-hidden />
            <span className="w-6 h-6 rounded-full bg-white/10" aria-hidden />
            <span className="w-6 h-6 rounded-full bg-white/10" aria-hidden />
          </div>
        </div>
      </header>

      {/* Contenido con mapa + formulario */}
      <section className="px-4 py-6 grid grid-cols-1 lg:grid-cols-[1.25fr_0.85fr] gap-6 items-stretch min-h-0">
        <div className="relative rounded-2xl ring-1 ring-white/10 bg-white/5 overflow-hidden">
          <WorldMap
            className="w-full h-[320px] md:h-full"
            countryCode={countryCode}
            onCountrySelect={(code) => {
              const iso2 = String(code).slice(0, 2).toUpperCase();
              setCountryCode(iso2);
              const found = countries.find((c) => c.code === iso2);
              if (found) setCountryName(found.name);
            }}
          />
        </div>
        <form onSubmit={handleSubmit} className="rounded-2xl ring-1 ring-white/10 bg-white/5 p-4 flex flex-col gap-4">
          <label className="text-sm text-white/70">País</label>
          <select
            value={countryCode}
            onChange={(e) => {
              const next = e.target.value;
              setCountryCode(next);
              const found = countries.find((c) => c.code === next);
              setCountryName(found?.name || '');
            }}
            className="h-11 rounded-lg bg-[#0b1020] ring-1 ring-white/15 px-3 outline-none focus:ring-2 focus:ring-[#8B3DFF]"
          >
            {loadingCountries ? (
              <option>Cargando países…</option>
            ) : (
              countries.map((c) => (
                <option key={c.code} value={c.code}>{c.name}</option>
              ))
            )}
          </select>

          {states.length > 0 && (
            <>
              <label className="text-sm text-white/70">Departamento/Estado</label>
              <select
                value={stateName}
                onChange={(e) => setStateName(e.target.value)}
                className="h-11 rounded-lg bg-[#0b1020] ring-1 ring-white/15 px-3 outline-none focus:ring-2 focus:ring-[#8B3DFF]"
              >
                {loadingStates ? (
                  <option>Cargando departamentos…</option>
                ) : (
                  states.map((s) => (
                    <option key={s} value={s}>{prettyState(s)}</option>
                  ))
                )}
              </select>
            </>
          )}

          <label className="text-sm text-white/70">Ciudad</label>
          <select
            value={city}
            onChange={(e) => setCity(e.target.value)}
            className="h-11 rounded-lg bg-[#0b1020] ring-1 ring-white/15 px-3 outline-none focus:ring-2 focus:ring-[#8B3DFF]"
          >
            {loadingCities ? (
              <option>Cargando ciudades…</option>
            ) : (
              cities.map((n) => (
                <option key={n} value={n}>{n}</option>
              ))
            )}
          </select>

          {/* Coordenadas */}
          <div className="text-sm text-white/70">
            <div>Coordenadas</div>
            <div className="mt-1 rounded-lg bg-[#0b1020] ring-1 ring-white/15 px-3 py-2 text-white/80">
              {loadingPos
                ? 'Buscando coordenadas...'
                : coords
                ? `Coordenadas: ${coords.lat.toFixed(5)}, ${coords.lon.toFixed(5)}`
                : '—'}
            </div>
          </div>

          <button
            type="submit"
            disabled={!canSubmit}
            className="mt-2 h-11 rounded-xl bg-[#8B3DFF] hover:bg-[#7d2fff] disabled:opacity-60 disabled:pointer-events-none transition shadow-[0_10px_30px_-10px_rgba(139,61,255,0.6)]"
          >
            {submitting ? 'Guardando...' : 'Guardar y continuar'}
          </button>
          <p className="text-xs text-white/50">Podrás cambiar tu ciudad luego en Configuración.</p>
        </form>
      </section>

      {/* Barra inferior */}
      <footer className="px-4 pb-3">
        <div className="relative mx-auto max-w-[820px]">
          <div className="h-10 rounded-full bg-gradient-to-r from-white/5 via-white/8 to-white/5 ring-1 ring-white/10" />
          <button className="absolute left-1/2 -translate-x-1/2 -top-4 w-12 h-12 rounded-full grid place-items-center bg-[#8B3DFF]" aria-label="Acción principal">
            <SparkleIcon />
          </button>
        </div>
      </footer>
    </div>
  );
}

function SparkleIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <path d="M12 3l1.6 3.5L17 8l-3.4 1.5L12 13l-1.6-3.5L7 8l3.4-1.5L12 3z" fill="currentColor" />
    </svg>
  );
}
