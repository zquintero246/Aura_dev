import React, { useEffect, useMemo, useState } from 'react';

type Props = {
  countryCode?: string; // ISO2
  onCountrySelect?: (code: string, name: string) => void;
  className?: string;
};

// Minimal mapping ISO2 -> name to find feature when dataset lacks ISO2
const ISO2_TO_NAME: Record<string, string> = {
  // LATAM
  AR: 'Argentina',
  BO: 'Bolivia',
  BR: 'Brazil',
  CL: 'Chile',
  CO: 'Colombia',
  CR: 'Costa Rica',
  CU: 'Cuba',
  DO: 'Dominican Republic',
  EC: 'Ecuador',
  SV: 'El Salvador',
  GT: 'Guatemala',
  HN: 'Honduras',
  MX: 'Mexico',
  NI: 'Nicaragua',
  PA: 'Panama',
  PY: 'Paraguay',
  PE: 'Peru',
  PR: 'Puerto Rico',
  UY: 'Uruguay',
  VE: 'Venezuela',
  BZ: 'Belize',
  GY: 'Guyana',
  SR: 'Suriname',
  TT: 'Trinidad and Tobago',
  // Extras
  US: 'United States of America',
  ES: 'Spain',
  FR: 'France',
  DE: 'Germany',
  IT: 'Italy',
  GB: 'United Kingdom',
  CN: 'China',
  JP: 'Japan',
  KR: 'Korea, Republic of',
  IN: 'India',
  CA: 'Canada',
  AU: 'Australia',
  NZ: 'New Zealand',
};

// Fallback animated background (no external deps)
function AnimatedBackdrop({ highlight }: { highlight?: { lat: number; lon: number } }) {
  // Convert lat/lon to percentage for a rough highlight in the box (equirectangular)
  const pos = useMemo(() => {
    if (!highlight) return { left: '50%', top: '50%', opacity: 0 } as const;
    const left = `${((highlight.lon + 180) / 360) * 100}%`;
    const top = `${((90 - highlight.lat) / 180) * 100}%`;
    return { left, top, opacity: 0.35 } as const;
  }, [highlight]);
  return (
    <div className="absolute inset-0">
      <div className="absolute inset-0 bg-[radial-gradient(1000px_600px_at_50%_120%,rgba(139,61,255,0.18),transparent_70%)]" />
      <div className="absolute inset-0 opacity-[0.07] bg-[conic-gradient(from_180deg_at_50%_50%,#ffffff40_0deg,transparent_120deg,transparent_240deg,#ffffff40_360deg)] animate-[spin_30s_linear_infinite]" />
      <div className="absolute inset-0">
        {Array.from({ length: 120 }).map((_, i) => (
          <span
            key={i}
            className="absolute w-[3px] h-[3px] rounded-full bg-white/30"
            style={{
              left: `${(i * 37) % 100}%`,
              top: `${(i * 19) % 100}%`,
              opacity: 0.4 + (i % 5) / 10,
            }}
          />
        ))}
        <span
          className="absolute w-24 h-24 -ml-12 -mt-12 rounded-full bg-[#8B3DFF] opacity-30 blur-2xl"
          style={pos}
        />
      </div>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );
}

export default function WorldMap({ countryCode, onCountrySelect, className }: Props) {
  const [lib, setLib] = useState<any | null>(null);
  const [geos, setGeos] = useState<any[]>([]);
  const [center, setCenter] = useState<[number, number]>([0, 20]);
  const [zoom, setZoom] = useState<number>(1.1);
  const [loadFailed, setLoadFailed] = useState(false);

  // Try to load react-simple-maps dynamically; fall back to animated backdrop if not present
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        // @ts-ignore - defer resolution; optional dependency at runtime
        // @vite-ignore
        const m = await import('react-simple-maps');
        if (!mounted) return;
        setLib(m);
      } catch {
        setLoadFailed(true);
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  // When country changes, compute viewport
  useEffect(() => {
    if (!countryCode) return;
    const code = countryCode.toUpperCase();
    // If we have geographies, find feature and center on it
    if (geos.length) {
      const feature =
        geos.find((g: any) => g.properties?.ISO_A2 === code || g.properties?.iso_a2 === code || g.id === code) ||
        geos.find(
          (g: any) =>
            ISO2_TO_NAME[code] &&
            [g.properties?.name, g.properties?.NAME, g.properties?.NAME_LONG, g.properties?.ADMIN, g.properties?.BRK_NAME]
              .filter(Boolean)
              .some((v: any) => String(v) === ISO2_TO_NAME[code])
        );
      if (feature) {
        const bbox = getBBox(feature);
        const cx = (bbox[0] + bbox[2]) / 2;
        const cy = (bbox[1] + bbox[3]) / 2;
        setCenter([cx, cy]);
        const span = Math.max(bbox[2] - bbox[0], bbox[3] - bbox[1]);
        let z = 2.2;
        if (span > 60) z = 1.4;
        else if (span > 40) z = 1.8;
        else if (span > 25) z = 2.4;
        else if (span > 15) z = 3.0;
        else if (span > 8) z = 3.6;
        else z = 4.2;
        setZoom(z);
        return;
      }
    }
    // Fallback center estimates
    const fallback = fallbackCenter(code);
    if (fallback) {
      setCenter([fallback.lon, fallback.lat]);
      setZoom(2.0);
    }
  }, [countryCode, geos]);

  // Fallback UI: static animated background linked to selection
  if (!lib || loadFailed) {
    const c = countryCode?.toUpperCase() || '';
    const fb = fallbackCenter(c);
    return (
      <div className={`relative ${className || ''}`}>
        <div className="w-full h-full rounded-[28px] ring-1 ring-white/15 bg-[#0b1020] overflow-hidden">
          <AnimatedBackdrop highlight={fb || undefined} />
        </div>
      </div>
    );
  }

  const { ComposableMap, Geographies, Geography, ZoomableGroup } = lib;
  const geoUrl = 'https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json';

  return (
    <div className={className}>
      <ComposableMap projectionConfig={{ scale: 150 }} style={{ width: '100%', height: '100%' }}>
        <ZoomableGroup center={center as any} zoom={zoom} animate={true}>
          <Geographies geography={geoUrl}>
            {({ geographies }: any) => {
              const filtered = (geographies || []).filter((g: any) => {
                const name = g.properties?.name || g.properties?.NAME || '';
                const iso2 = g.properties?.ISO_A2 || g.properties?.iso_a2 || g.id;
                return name !== 'Antarctica' && iso2 !== 'AQ';
              });
              if (geos.length === 0 && filtered?.length) setGeos(filtered);
              const selected = String(countryCode || '').toUpperCase();
              return filtered.map((geo: any) => {
                const iso2 = String(geo.properties?.ISO_A2 || geo.properties?.iso_a2 || geo.id || '').toUpperCase();
                const name = String(geo.properties?.name || geo.properties?.NAME || geo.properties?.NAME_LONG || '');
                const isSelected =
                  iso2 === selected || (ISO2_TO_NAME[selected] && namesEqual(name, ISO2_TO_NAME[selected]));
                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    onClick={() => {
                      const code = geo.properties?.ISO_A2 || geo.properties?.iso_a2 || iso2FromName(name) || '';
                      onCountrySelect?.(String(code), String(name));
                    }}
                    style={{
                      default: { fill: isSelected ? 'rgba(139,61,255,0.45)' : '#0b1020', stroke: 'rgba(255,255,255,0.08)', outline: 'none', cursor: 'pointer' },
                      hover: { fill: 'rgba(139,61,255,0.35)', stroke: 'rgba(255,255,255,0.12)', outline: 'none', cursor: 'pointer' },
                      pressed: { fill: 'rgba(139,61,255,0.55)', stroke: 'rgba(255,255,255,0.15)', outline: 'none' },
                    }}
                  />
                );
              });
            }}
          </Geographies>
        </ZoomableGroup>
      </ComposableMap>
    </div>
  );
}

function getBBox(feature: any): [number, number, number, number] {
  const coords: number[][] = [];
  const geom = feature?.geometry;
  const pushCoords = (arr: any) => {
    for (const p of arr) {
      if (typeof p[0] === 'number') coords.push(p as number[]);
      else pushCoords(p);
    }
  };
  if (geom?.type === 'Polygon') pushCoords(geom.coordinates);
  else if (geom?.type === 'MultiPolygon') pushCoords(geom.coordinates);
  let minX = 180,
    minY = 90,
    maxX = -180,
    maxY = -90;
  for (const [x, y] of coords) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  return [minX, minY, maxX, maxY];
}

function fallbackCenter(code: string): { lat: number; lon: number } | null {
  const map: Record<string, { lat: number; lon: number }> = {
    CO: { lat: 4.6, lon: -74.1 },
    US: { lat: 39, lon: -98 },
    MX: { lat: 23.6, lon: -102.5 },
    ES: { lat: 40.3, lon: -3.7 },
    BR: { lat: -14.2, lon: -51.9 },
    AR: { lat: -38.4, lon: -63.6 },
    CL: { lat: -35.7, lon: -71.5 },
    PE: { lat: -9.2, lon: -75.0 },
    VE: { lat: 8.0, lon: -66.9 },
    FR: { lat: 46.2, lon: 2.2 },
    DE: { lat: 51.2, lon: 10.4 },
    IT: { lat: 41.9, lon: 12.6 },
    GB: { lat: 55.4, lon: -3.4 },
    CN: { lat: 35.9, lon: 104.2 },
    JP: { lat: 36.2, lon: 138.3 },
    KR: { lat: 36.5, lon: 127.9 },
    IN: { lat: 20.6, lon: 78.9 },
    CA: { lat: 56.1, lon: -106.3 },
    AU: { lat: -25.3, lon: 133.8 },
    NZ: { lat: -41.3, lon: 174.8 },
  };
  return map[code] || null;
}

// ---- Helpers for robust matching ----
function readName(g: any): string {
  return (
    g?.properties?.name ||
    g?.properties?.NAME ||
    g?.properties?.NAME_LONG ||
    g?.properties?.ADMIN ||
    g?.properties?.BRK_NAME ||
    ''
  );
}

function getISO2(g: any): string | null {
  const v = g?.properties?.ISO_A2 || g?.properties?.iso_a2 || null;
  return v ? String(v) : null;
}

function normalizeName(s: string): string {
  return String(s)
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[\.']/g, '')
    .toLowerCase()
    .trim();
}

function namesEqual(a: string, b: string): boolean {
  return normalizeName(a) === normalizeName(b);
}

function iso2FromName(name: string): string | null {
  const n = normalizeName(name);
  for (const [code, label] of Object.entries(ISO2_TO_NAME)) {
    if (normalizeName(label) === n) return code;
  }
  return null;
}
