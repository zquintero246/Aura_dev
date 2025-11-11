import React, { useEffect, useMemo, useRef, useState } from 'react';
import { getMyLocation } from '../../lib/location';
import {
  fetchDashboardByUser,
  fetchDashboardByCoords,
  type WeatherData,
} from '../../lib/weatherApi';
import { me } from '../../lib/auth';
import HomeRegistration from './HomeRegistration';
import { fetchWeatherHistory, wmoToText, type HistoryPoint } from '../../lib/weatherHistory';
import { listDevices, powerDevice, updateDevice, type Device } from '../../lib/devices';
import {
  WiThermometer,
  WiHumidity,
  WiDaySunny,
  WiCloudy,
  WiRain,
  WiSnow,
  WiThunderstorm,
  WiFog,
} from 'react-icons/wi';
import { FiPlus, FiClock, FiSun, FiCloud, FiSettings, FiMinus, FiBarChart2 } from 'react-icons/fi';
import {
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';

export default function HomeAssistantPanel() {
  const [needsRegistration, setNeedsRegistration] = useState<boolean | null>(null);
  const [userId, setUserId] = useState<string>('');
  const [wx, setWx] = useState<WeatherData | null>(null);
  const [loadingDash, setLoadingDash] = useState(false);
  const [now, setNow] = useState<Date>(new Date());
  const [home, setHome] = useState<{
    city?: string;
    country?: string;
    latitude?: number;
    longitude?: number;
  } | null>(null);
  const [history, setHistory] = useState<HistoryPoint[] | null>(null);
  const [modal, setModal] = useState<null | 'temp' | 'hum' | 'cond'>(null);
  const [devices, setDevices] = useState<Device[]>([]);
  const [simMode, setSimMode] = useState(false);
  const [editing, setEditing] = useState<Device | null>(null);
  const [addModal, setAddModal] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [telemetryMode, setTelemetryMode] = useState(false);
  const [zoom, setZoom] = useState<number>(1);
  const zoomKey = useMemo(() => (userId ? `aura:dash:zoom:${userId}` : ''), [userId]);
  useEffect(() => {
    if (!zoomKey) return;
    try {
      const raw = localStorage.getItem(zoomKey);
      if (raw) setZoom(Math.max(0.6, Math.min(1.6, Number(raw) || 1)));
    } catch {}
  }, [zoomKey]);
  useEffect(() => {
    if (!zoomKey) return;
    try {
      localStorage.setItem(zoomKey, String(zoom));
    } catch {}
  }, [zoomKey, zoom]);
  const defaultAddables = useMemo(
    () => [
      { type: 'coordinator', name: 'Zigbee Coordinator' },
      { type: 'controller', name: 'Z-Wave Controller' },
      { type: 'contact_sensor', name: 'Aqara Door/Window' },
      { type: 'motion_sensor', name: 'Aqara Motion' },
      { type: 'relay', name: 'Shelly Plus 1' },
      { type: 'smart_plug', name: 'TP-Link Kasa Plug' },
      { type: 'light_bulb', name: 'Philips Hue Bulb' },
      { type: 'thermostat', name: 'Ecobee Thermostat' },
      { type: 'chromecast', name: 'Google Chromecast' },
      { type: 'speaker', name: 'Sonos Speaker' },
    ],
    []
  );
  const addables = useMemo(() => {
    if (!devices || devices.length === 0) return defaultAddables;
    // Derivar por tipo de los existentes (únicos)
    const seen = new Set<string>();
    const out: { type: string; name: string }[] = [];
    for (const d of devices) {
      const t = String(d.type || 'device');
      if (!seen.has(t)) {
        seen.add(t);
        out.push({ type: t, name: d.name || t });
      }
    }
    // Garantizar que estén los 10 tipos si faltan
    for (const def of defaultAddables) {
      if (!seen.has(def.type)) out.push(def);
    }
    return out;
  }, [devices, defaultAddables]);

  const isWxOk = wx && !(wx as any).error;
  const tempBadge =
    isWxOk && (wx as any).temperature != null ? `${Math.round((wx as any).temperature)}°` : '—';
  const humBadge =
    isWxOk && (wx as any).humidity != null ? `${Math.round((wx as any).humidity)}%` : '—';
  const condText = isWxOk ? String((wx as any).condition || '') : '';
  const condEmoji = (() => {
    const s = condText.toLowerCase();
    if (!s) return '—';
    if (s.includes('thunder')) return '⛈';
    if (s.includes('rain') || s.includes('shower')) return '🌧';
    if (s.includes('snow')) return '❄';
    if (s.includes('fog') || s.includes('mist') || s.includes('haze')) return '🌫';
    if (s.includes('clear')) return '☀';
    if (s.includes('cloud')) return '☁';
    return '⛅';
  })();

  const CondIcon = useMemo(() => {
    const s = condText.toLowerCase();
    if (s.includes('thunder')) return WiThunderstorm;
    if (s.includes('rain') || s.includes('shower')) return WiRain;
    if (s.includes('snow')) return WiSnow;
    if (s.includes('fog') || s.includes('mist') || s.includes('haze')) return WiFog;
    if (s.includes('clear')) return WiDaySunny;
    if (s.includes('cloud')) return WiCloudy;
    return WiDaySunny;
  }, [condText]);
  const tempLabel =
    isWxOk && (wx as any).temperature != null
      ? `${(wx as any).temperature.toFixed ? (wx as any).temperature.toFixed(1) : (wx as any).temperature} °C`
      : '—';
  const humLabel =
    isWxOk && (wx as any).humidity != null
      ? `${(wx as any).humidity.toFixed ? (wx as any).humidity.toFixed(0) : (wx as any).humidity} %`
      : '—';
  const condLabel = condText || '—';

  useEffect(() => {
    (async () => {
      try {
        const res = await me();
        const id = res?.user?.id ?? '';
        setUserId(String(id));
        let required = true;
        const h = await getMyLocation();
        setHome(h || null);
        required = !h;
        // fallback local if backend not reachable
        if (h == null) {
          const keyReg = id ? `aura:home:registered:${id}` : '';
          required = keyReg ? localStorage.getItem(keyReg) !== '1' : true;
        }
        setNeedsRegistration(required);

        // Fetch dashboard if we have a location
        if (h) {
          setLoadingDash(true);
          const dash =
            (await fetchDashboardByCoords(Number(h.latitude), Number(h.longitude))) ||
            (id ? await fetchDashboardByUser(String(id)) : null);
          setWx(dash);
          setLoadingDash(false);
          // Load history for charts
          try {
            const his = await fetchWeatherHistory(Number(h.latitude), Number(h.longitude));
            setHistory(his);
          } catch (_) {
            /* ignore */
          }
          // Load devices
          try {
            const devs = await listDevices();
            setDevices(
              Array.isArray(devs) && devs.length > 0 ? withOverrides(devs) : seedSimDevices()
            );
          } catch {}
        } else {
          // Fallback: use localStorage coords if present
          try {
            const raw = id ? localStorage.getItem(`aura:home:coords:${id}`) : null;
            if (raw) {
              const c = JSON.parse(raw);
              if (c?.lat != null && c?.lon != null) {
                setLoadingDash(true);
                const dash = await fetchDashboardByCoords(Number(c.lat), Number(c.lon));
                setWx(dash);
                setLoadingDash(false);
                try {
                  const his = await fetchWeatherHistory(Number(c.lat), Number(c.lon));
                  setHistory(his);
                } catch {}
                try {
                  const devs = await listDevices();
                  setDevices(
                    Array.isArray(devs) && devs.length > 0 ? withOverrides(devs) : seedSimDevices()
                  );
                } catch {
                  setDevices(seedSimDevices());
                }
              }
            }
          } catch {}
        }
      } catch {
        setNeedsRegistration(true);
      }
    })();
  }, []);

  // After registration completes, re-fetch location + weather
  useEffect(() => {
    if (needsRegistration === false) {
      (async () => {
        try {
          const h = await getMyLocation();
          setHome(h || null);
          if (h) {
            setLoadingDash(true);
            const dash = await fetchDashboardByCoords(Number(h.latitude), Number(h.longitude));
            setWx(dash);
            setLoadingDash(false);
          }
        } catch {}
      })();
    }
  }, [needsRegistration]);

  // Poll devices periodically to reflect simulated updates
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const devs = await listDevices();
        if (!cancelled && Array.isArray(devs))
          setDevices(devs.length > 0 ? withOverrides(devs) : seedSimDevices());
      } catch {
        if (!cancelled) setDevices(seedSimDevices());
      }
    };
    // Initial and interval
    poll();
    const id = window.setInterval(poll, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  // --- Dashboard tile order (drag & drop) ---
  type TileId =
    | 'time'
    | 'lights'
    | 'aqi'
    | 'temp'
    | 'hum'
    | 'chart'
    | 'location'
    | 'actions'
    | 'gauge'
    | 'summary';
  const defaultTileOrder: TileId[] = useMemo(
    () => [
      'time',
      'lights',
      'aqi',
      'temp',
      'hum',
      'chart',
      'location',
      'actions',
      'gauge',
      'summary',
    ],
    []
  );
  const orderKey = useMemo(() => (userId ? `aura:dash:order:${userId}` : ''), [userId]);
  const [tileOrder, setTileOrder] = useState<TileId[]>(defaultTileOrder);
  useEffect(() => {
    if (!orderKey) return;
    try {
      const raw = localStorage.getItem(orderKey);
      if (raw) {
        const saved = JSON.parse(raw) as string[];
        const allow = new Set(defaultTileOrder);
        const merged: TileId[] = saved.filter((id) => allow.has(id as TileId)) as TileId[];
        for (const id of defaultTileOrder) if (!merged.includes(id)) merged.push(id);
        setTileOrder(merged);
      } else {
        setTileOrder(defaultTileOrder);
      }
    } catch {
      setTileOrder(defaultTileOrder);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [orderKey]);
  useEffect(() => {
    if (!orderKey) return;
    try {
      localStorage.setItem(orderKey, JSON.stringify(tileOrder));
    } catch {}
  }, [orderKey, tileOrder]);

  const [dragging, setDragging] = useState<TileId | null>(null);
  const startDrag = (id: TileId) => (e: React.DragEvent) => {
    if (!editMode) {
      e.preventDefault();
      return;
    }
    e.dataTransfer?.setData('text/plain', id);
    e.dataTransfer.effectAllowed = 'move';
    setDragging(id);
  };
  const overTile = (id: TileId) => (e: React.DragEvent) => {
    if (!editMode) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };
  const dropOnTile = (id: TileId) => (e: React.DragEvent) => {
    if (!editMode) return;
    e.preventDefault();
    const from = (dragging || (e.dataTransfer?.getData('text/plain') as TileId)) as TileId;
    if (!from || from === id) return;
    const a = tileOrder.indexOf(from);
    const b = tileOrder.indexOf(id);
    if (a < 0 || b < 0) return;
    const next = tileOrder.slice();
    next.splice(a, 1);
    next.splice(b, 0, from);
    setTileOrder(next);
    setDragging(null);
  };
  const dropAtEnd = (e: React.DragEvent) => {
    if (!editMode) return;
    e.preventDefault();
    const from = (dragging || (e.dataTransfer?.getData('text/plain') as TileId)) as TileId;
    if (!from) return;
    const a = tileOrder.indexOf(from);
    if (a < 0) return;
    const next = tileOrder.slice();
    next.splice(a, 1);
    next.push(from);
    setTileOrder(next);
    setDragging(null);
  };

  // Tile sizes with flexible grid units (snap-to-grid)
  // w/h are in grid cells; grid has configurable columns and row height
  type TileSizes = Record<TileId, { w: number; h: number }>;
  const defaultTileSizes: TileSizes = {
    time: { w: 6, h: 5 },
    lights: { w: 6, h: 5 },
    aqi: { w: 6, h: 5 },
    temp: { w: 6, h: 5 },
    hum: { w: 6, h: 5 },
    chart: { w: 14, h: 10 },
    location: { w: 14, h: 10 },
    actions: { w: 12, h: 6 },
    gauge: { w: 12, h: 10 },
    summary: { w: 12, h: 6 },
  };
  const sizeKey = useMemo(() => (userId ? `aura:dash:sizes:${userId}` : ''), [userId]);
  const [tileSizes, setTileSizes] = useState<TileSizes>(defaultTileSizes);
  useEffect(() => {
    if (!sizeKey) return;
    try {
      const raw = localStorage.getItem(sizeKey);
      if (raw) {
        const saved = JSON.parse(raw) as TileSizes;
        setTileSizes({ ...defaultTileSizes, ...saved });
      } else {
        setTileSizes(defaultTileSizes);
      }
    } catch {
      setTileSizes(defaultTileSizes);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sizeKey]);
  useEffect(() => {
    if (!sizeKey) return;
    try {
      localStorage.setItem(sizeKey, JSON.stringify(tileSizes));
    } catch {}
  }, [sizeKey, tileSizes]);
  // Grid configuration + resize helpers
  const gridKey = useMemo(() => (userId ? `aura:dash:grid:${userId}` : ''), [userId]);
  const [gridCols, setGridCols] = useState<number>(24);
  const [rowUnit, setRowUnit] = useState<number>(24); // px per row
  const [stepX, setStepX] = useState<number>(1);
  const [stepY, setStepY] = useState<number>(1);
  // Grid ref and pointer-resize state
  const gridRef = useRef<HTMLDivElement | null>(null);
  const resizeRef = useRef<{
    id: TileId | null;
    startX: number;
    startY: number;
    startW: number;
    startH: number;
  } | null>(null);
  useEffect(() => {
    if (!gridKey) return;
    try {
      const raw = localStorage.getItem(gridKey);
      if (raw) {
        const s = JSON.parse(raw) as {
          cols?: number;
          rowUnit?: number;
          stepX?: number;
          stepY?: number;
        };
        if (s?.cols) setGridCols(Math.max(8, Math.min(48, Number(s.cols))));
        if (s?.rowUnit) setRowUnit(Math.max(12, Math.min(64, Number(s.rowUnit))));
        if (s?.stepX) setStepX(Math.max(1, Math.min(10, Number(s.stepX))));
        if (s?.stepY) setStepY(Math.max(1, Math.min(10, Number(s.stepY))));
      }
    } catch {}
  }, [gridKey]);
  useEffect(() => {
    if (!gridKey) return;
    try {
      localStorage.setItem(gridKey, JSON.stringify({ cols: gridCols, rowUnit, stepX, stepY }));
    } catch {}
  }, [gridKey, gridCols, rowUnit, stepX, stepY]);
  const resizeTile = (id: TileId, dim: 'w' | 'h', delta: 1 | -1) => {
    setTileSizes((prev) => {
      const cur = prev[id] || { w: 6, h: 5 };
      const inc = (dim === 'w' ? stepX : stepY) * delta;
      const nextRaw = { ...cur, [dim]: (cur as any)[dim] + inc } as {
        w: number;
        h: number;
      };
      const maxW = gridCols;
      const maxH = 100; // pragmatic upper bound
      const next = {
        ...prev,
        [id]: {
          w: Math.max(1, Math.min(maxW, Math.round(nextRaw.w))),
          h: Math.max(1, Math.min(maxH, Math.round(nextRaw.h))),
        },
      } as TileSizes;
      return next;
    });
  };

  // Pointer-based corner resizing for tiles (bottom-right handle)
  const onResizeStart = (id: TileId) => (e: React.PointerEvent) => {
    try {
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    } catch {}
    const cur = tileSizes[id] || { w: 6, h: 5 };
    resizeRef.current = {
      id,
      startX: e.clientX,
      startY: e.clientY,
      startW: cur.w,
      startH: cur.h,
    };
    e.preventDefault();
  };
  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const st = resizeRef.current;
      if (!st || !st.id) return;
      const gridEl = gridRef.current;
      if (!gridEl) return;
      const cs = getComputedStyle(gridEl);
      const colGap = parseFloat(cs.columnGap || '0') || 0;
      const rowGap = parseFloat(cs.rowGap || '0') || 0;
      const gridWidth = gridEl.clientWidth || 1;
      const colWidth = (gridWidth - colGap * (gridCols - 1)) / gridCols;
      const incW = colWidth + colGap;
      const incH = rowUnit + rowGap;
      const dx = (e.clientX - st.startX) / (zoom || 1);
      const dy = (e.clientY - st.startY) / (zoom || 1);
      const nextW = Math.max(1, Math.min(gridCols, Math.round(st.startW + dx / incW)));
      const nextH = Math.max(1, Math.min(100, Math.round(st.startH + dy / incH)));
      setTileSizes((prev) => ({ ...prev, [st.id as TileId]: { w: nextW, h: nextH } }));
    };
    const onUp = () => {
      resizeRef.current = null;
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  }, [gridCols, rowUnit, tileSizes, zoom]);

  // Derived quick stats and clock refresh
  const lightsOn = useMemo(
    () => devices.filter((d) => d.type === 'light_bulb' && d.is_on).length,
    [devices]
  );
  const plugsOn = useMemo(
    () => devices.filter((d) => d.type === 'smart_plug' && d.is_on).length,
    [devices]
  );
  const totalDevices = devices.length;
  useEffect(() => {
    const t = window.setInterval(() => setNow(new Date()), 30000);
    return () => window.clearInterval(t);
  }, []);

  // Simulated devices telemetry buffer (multi-series per device)
  type TelemetrySample = { t: number; y: number };
  type TeleSeries = { label: string; samples: TelemetrySample[] };
  type TeleDB = Record<string, { series: Record<string, TeleSeries> }>;
  const telemetryRef = useRef<TeleDB>({});
  const teleStoreKey = useMemo(
    () => (userId ? `aura:telemetry:${userId}` : 'aura:telemetry:anon'),
    [userId]
  );
  const TELE_CAP = 5000;
  // Load persisted telemetry history (keeps history across toggles/repaints)
  useEffect(() => {
    if (!teleStoreKey) return;
    try {
      const raw = localStorage.getItem(teleStoreKey);
      if (!raw) return;
      const saved = JSON.parse(raw) as any;
      if (saved && typeof saved === 'object') {
        // Back-compat: migrate single-series format { key,label,samples } to multi-series
        const out: TeleDB = {};
        for (const did of Object.keys(saved)) {
          const val = saved[did];
          if (val && val.series) {
            out[did] = { series: val.series as Record<string, TeleSeries> };
          } else if (val && val.samples) {
            const k = val.key || 'metric';
            const lbl = val.label || String(k);
            out[did] = {
              series: {
                [k]: { label: lbl, samples: Array.isArray(val.samples) ? val.samples : [] },
              },
            };
          }
        }
        telemetryRef.current = out;
      }
    } catch {}
  }, [teleStoreKey]);
  const getMetricFor = (d: Device): { key: string; label: string } | null => {
    const t = String(d.type || '');
    switch (t) {
      case 'smart_plug':
        return { key: 'is_on', label: 'Estado' };
      case 'relay':
        return { key: 'power_w', label: 'Potencia (W)' };
      case 'light_bulb':
        return { key: 'is_on', label: 'Estado' };
      case 'thermostat':
        return { key: 'current_c', label: 'Temperatura (C)' };
      case 'motion_sensor':
        return { key: 'motion_detected', label: 'Movimiento' };
      case 'contact_sensor':
        return { key: 'opened', label: 'Apertura' };
      case 'speaker':
        return { key: 'playback_state', label: 'Reproduccion' };
      case 'chromecast':
        return { key: 'playback_state', label: 'Reproduccion' };
      case 'coordinator':
        return { key: 'connected_devices', label: 'Conectados' };
      case 'controller':
        return { key: 'node_count', label: 'Nodos' };
      default:
        // Fallback: choose first numeric/boolean property to chart
        const skip = new Set(['id', 'name', 'type', 'protocol', 'is_on', 'last_seen']);
        const preferred = [
          'power_w',
          'current_c',
          'brightness',
          'lux',
          'volume',
          'connected_devices',
          'node_count',
        ];
        for (const k of preferred) {
          const v = (d as any)[k];
          if (typeof v === 'number' || typeof v === 'boolean') {
            return { key: k, label: k };
          }
        }
        for (const k of Object.keys(d)) {
          if (skip.has(k)) continue;
          const v = (d as any)[k];
          if (typeof v === 'number' || typeof v === 'boolean') {
            return { key: k, label: k };
          }
        }
        return null;
    }
  };
  const extraSeriesFor = (d: Device): { key: string; label: string }[] => {
    const t = String(d.type || '');
    switch (t) {
      case 'relay':
        return [{ key: 'is_on', label: 'Estado' }];
      case 'chromecast':
        return [{ key: 'volume', label: 'Volumen (%)' }];
      default:
        return [];
    }
  };
  const numVal = (v: any): number => {
    if (typeof v === 'number' && isFinite(v)) return v;
    if (typeof v === 'boolean') return v ? 1 : 0;
    const n = Number(v);
    return isFinite(n) ? n : 0;
  };
  const BOOL_KEYS = new Set(['is_on', 'opened', 'motion_detected', 'muted']);
  const ENUM_SERIES: Record<string, string[]> = {
    playback_state: ['idle', 'stopped', 'paused', 'playing'],
    hvac_mode: ['off', 'heat', 'cool', 'auto', 'fan_only'],
  };
  const BOOL_LABELS: Record<string, [string, string]> = {
    is_on: ['Off', 'On'],
    opened: ['Cerrado', 'Abierto'],
    motion_detected: ['Sin movimiento', 'Movimiento'],
    muted: ['Normal', 'Silenciado'],
  };
  const ENUM_LABELS: Record<string, Record<string, string>> = {
    playback_state: {
      idle: 'Idle',
      stopped: 'Detenido',
      paused: 'Pausado',
      playing: 'Reproduciendo',
    },
    hvac_mode: {
      off: 'Apagado',
      heat: 'Calor',
      cool: 'Frio',
      auto: 'Auto',
      fan_only: 'Ventilacion',
    },
  };
  const encodeMetric = (key: string, raw: any): number => {
    if (BOOL_KEYS.has(key)) return raw ? 1 : 0;
    const enumArr = ENUM_SERIES[key];
    if (enumArr) {
      const idx = enumArr.indexOf(String(raw ?? ''));
      return idx >= 0 ? idx : 0;
    }
    return numVal(raw);
  };
  const labelFromEncoded = (key: string, encoded: number, raw?: any): string => {
    if (BOOL_KEYS.has(key)) {
      const [off, on] = BOOL_LABELS[key] || ['0', '1'];
      return encoded >= 1 ? on : off;
    }
    const enumArr = ENUM_SERIES[key];
    if (enumArr) {
      const val = enumArr[Math.max(0, Math.min(enumArr.length - 1, Math.round(encoded)))];
      return (ENUM_LABELS[key] && (ENUM_LABELS[key] as any)[val]) || String(val);
    }
    const v = typeof raw === 'number' ? raw : encoded;
    return String(v);
  };

  // Colors per device chart (default palette + customizable)
  const DEFAULT_TELE_COLORS = [
    '#8B3DFF',
    '#06B6D4',
    '#F59E0B',
    '#10B981',
    '#D946EF',
    '#EF4444',
    '#3B82F6',
    '#F97316',
    '#22C55E',
    '#EAB308',
    '#14B8A6',
    '#A855F7',
  ];
  const hashStr = (s: string): number => {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
    return Math.abs(h);
  };
  const defaultTeleColor = (id: string): string =>
    DEFAULT_TELE_COLORS[hashStr(id) % DEFAULT_TELE_COLORS.length];
  const [telemetryColors, setTelemetryColors] = useState<Record<string, string>>({});
  const teleColorKey = useMemo(() => (userId ? `aura:tele:colors:${userId}` : ''), [userId]);
  useEffect(() => {
    if (!teleColorKey) return;
    try {
      const raw = localStorage.getItem(teleColorKey);
      if (raw) setTelemetryColors(JSON.parse(raw));
    } catch {}
  }, [teleColorKey]);
  useEffect(() => {
    if (!teleColorKey) return;
    try {
      localStorage.setItem(teleColorKey, JSON.stringify(telemetryColors));
    } catch {}
  }, [teleColorKey, telemetryColors]);

  // Telemetry resolution (aggregation buckets)
  type TeleRes = '1m' | '5m' | '20m' | '30m' | '1h' | '1d';
  const [telemetryResolution, setTelemetryResolution] = useState<TeleRes>('1m');
  const teleResKey = useMemo(
    () => (userId ? `aura:tele:res:${userId}` : 'aura:tele:res:anon'),
    [userId]
  );
  useEffect(() => {
    try {
      const raw = localStorage.getItem(teleResKey);
      if (raw) setTelemetryResolution(raw as TeleRes);
    } catch {}
  }, [teleResKey]);
  useEffect(() => {
    try {
      localStorage.setItem(teleResKey, telemetryResolution);
    } catch {}
  }, [teleResKey, telemetryResolution]);

  const bucketMsFor = (res: TeleRes): number => {
    switch (res) {
      case '1m':
        return 60 * 1000;
      case '5m':
        return 5 * 60 * 1000;
      case '20m':
        return 20 * 60 * 1000;
      case '30m':
        return 30 * 60 * 1000;
      case '1h':
        return 60 * 60 * 1000;
      case '1d':
        return 24 * 60 * 60 * 1000;
    }
  };
  const aggregateSeries = (
    samples: TelemetrySample[],
    opts: { kind: 'bool' | 'enum' | 'num'; enumKey?: string; res: TeleRes }
  ): TelemetrySample[] => {
    if (!samples || samples.length === 0) return [];
    const size = bucketMsFor(opts.res);
    const byBucket = new Map<number, TelemetrySample[]>();
    for (const s of samples) {
      const bucket = Math.floor(s.t / size) * size;
      const arr = byBucket.get(bucket);
      if (arr) arr.push(s);
      else byBucket.set(bucket, [s]);
    }
    const out: TelemetrySample[] = [];
    const buckets = Array.from(byBucket.keys()).sort((a, b) => a - b);
    for (const b of buckets) {
      const arr = byBucket.get(b)!;
      if (opts.kind === 'num') {
        const avg = arr.reduce((acc, s) => acc + (s.y || 0), 0) / arr.length;
        out.push({ t: b, y: Number(avg.toFixed(4)) });
      } else if (opts.kind === 'bool' || opts.kind === 'enum') {
        const counts = new Map<number, number>();
        for (const s of arr) counts.set(s.y, (counts.get(s.y) || 0) + 1);
        let best = arr[arr.length - 1].y;
        let max = -1;
        for (const [val, c] of counts) {
          if (c > max) {
            max = c;
            best = val;
          }
        }
        out.push({ t: b, y: best });
      }
    }
    return out;
  };
  useEffect(() => {
    const ts = Date.now();
    for (const d of devices) {
      const primary = getMetricFor(d);
      if (!primary) continue;
      const extras = extraSeriesFor(d);
      const dev = telemetryRef.current[d.id] || { series: {} };
      const all = [primary, ...extras];
      for (const s of all) {
        const raw = (d as any)[s.key];
        const enc = encodeMetric(s.key, raw);
        const ser: TeleSeries = dev.series[s.key] || { label: s.label, samples: [] };
        // if label changed, update
        ser.label = s.label;
        ser.samples.push({ t: ts, y: enc });
        if (ser.samples.length > TELE_CAP) ser.samples.splice(0, ser.samples.length - TELE_CAP);
        dev.series[s.key] = ser;
      }
      telemetryRef.current[d.id] = dev;
    }
    // persist
    try {
      if (teleStoreKey) localStorage.setItem(teleStoreKey, JSON.stringify(telemetryRef.current));
    } catch {}
  }, [devices]);

  // --- Left stats DnD order ---
  type StatId = 'time' | 'lights' | 'aqi' | 'temp' | 'hum';
  const defaultLeftOrder: StatId[] = useMemo(() => ['time', 'lights', 'aqi', 'temp', 'hum'], []);
  const leftKey = useMemo(() => (userId ? `aura:dash:left:${userId}` : ''), [userId]);
  const [leftOrder, setLeftOrder] = useState<StatId[]>(defaultLeftOrder);
  const [leftDrag, setLeftDrag] = useState<StatId | null>(null);
  useEffect(() => {
    if (!leftKey) return;
    try {
      const raw = localStorage.getItem(leftKey);
      if (raw) {
        const saved = JSON.parse(raw) as string[];
        const allow = new Set(defaultLeftOrder);
        const merged: StatId[] = saved.filter((s) => allow.has(s as StatId)) as StatId[];
        for (const s of defaultLeftOrder) if (!merged.includes(s)) merged.push(s);
        setLeftOrder(merged);
      } else {
        setLeftOrder(defaultLeftOrder);
      }
    } catch {
      setLeftOrder(defaultLeftOrder);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [leftKey]);
  useEffect(() => {
    if (!leftKey) return;
    try {
      localStorage.setItem(leftKey, JSON.stringify(leftOrder));
    } catch {}
  }, [leftKey, leftOrder]);
  const dropLeftOn = (to: StatId) => {
    const from = leftDrag;
    if (!from || from === to) return;
    const a = leftOrder.indexOf(from);
    const b = leftOrder.indexOf(to);
    if (a < 0 || b < 0) return;
    const next = leftOrder.slice();
    next.splice(a, 1);
    next.splice(b, 0, from);
    setLeftOrder(next);
    setLeftDrag(null);
  };

  // --- Devices drag & drop order (frontend-only) ---
  const devicesOrderKey = useMemo(() => (userId ? `aura:devices:order:${userId}` : ''), [userId]);
  const [dragDevice, setDragDevice] = useState<string | null>(null);
  const [deviceOrder, setDeviceOrder] = useState<string[] | null>(null);
  const deviceSizeKey = useMemo(() => (userId ? `aura:devices:size:${userId}` : ''), [userId]);
  const [deviceSizes, setDeviceSizes] = useState<Record<string, 's' | 'm' | 'l'>>({});
  const deviceHeightsKey = useMemo(
    () => (userId ? `aura:devices:heights:${userId}` : ''),
    [userId]
  );
  const [deviceHeights, setDeviceHeights] = useState<Record<string, number>>({});
  const deviceResizeRef = useRef<{ id: string; startY: number; startH: number } | null>(null);
  // Initialize/sanitize order when devices change or key available
  useEffect(() => {
    const ids = (devices || []).map((d) => String(d.id));
    if (!devicesOrderKey) {
      setDeviceOrder(ids);
    } else {
      try {
        const raw = localStorage.getItem(devicesOrderKey);
        if (raw) {
          let saved = (JSON.parse(raw) as string[]).filter((id) => ids.includes(String(id)));
          for (const id of ids) if (!saved.includes(id)) saved.push(id);
          setDeviceOrder(saved);
        } else {
          setDeviceOrder(ids);
        }
      } catch {
        setDeviceOrder(ids);
      }
    }
    // bootstrap sizes
    if (deviceSizeKey) {
      try {
        const rawS = localStorage.getItem(deviceSizeKey);
        if (rawS) {
          const parsed = JSON.parse(rawS) as Record<string, 's' | 'm' | 'l'>;
          const next: Record<string, 's' | 'm' | 'l'> = {};
          for (const id of ids) if (parsed[id]) next[id] = parsed[id];
          setDeviceSizes(next);
        }
      } catch {}
    }
  }, [devicesOrderKey, deviceSizeKey, devices]);
  // Load saved per-device explicit heights
  useEffect(() => {
    if (!deviceHeightsKey) return;
    try {
      const raw = localStorage.getItem(deviceHeightsKey);
      if (raw) {
        const parsed = JSON.parse(raw) as Record<string, number>;
        const ids = (devices || []).map((d) => String(d.id));
        const next: Record<string, number> = {};
        for (const id of ids) if (parsed[id] != null) next[id] = Number(parsed[id]);
        setDeviceHeights(next);
      }
    } catch {}
  }, [deviceHeightsKey, devices]);
  useEffect(() => {
    if (!devicesOrderKey || !deviceOrder) return;
    try {
      localStorage.setItem(devicesOrderKey, JSON.stringify(deviceOrder));
    } catch {}
  }, [devicesOrderKey, deviceOrder]);
  useEffect(() => {
    if (!deviceSizeKey) return;
    try {
      localStorage.setItem(deviceSizeKey, JSON.stringify(deviceSizes));
    } catch {}
  }, [deviceSizeKey, deviceSizes]);
  useEffect(() => {
    if (!deviceHeightsKey) return;
    try {
      localStorage.setItem(deviceHeightsKey, JSON.stringify(deviceHeights));
    } catch {}
  }, [deviceHeightsKey, deviceHeights]);
  const sortedDevices = useMemo(() => {
    if (!Array.isArray(devices)) return [] as Device[];
    if (!deviceOrder) return devices;
    const idx = new Map(deviceOrder.map((id, i) => [String(id), i] as const));
    return [...devices].sort((a, b) => (idx.get(String(a.id)) ?? 0) - (idx.get(String(b.id)) ?? 0));
  }, [devices, deviceOrder]);
  const onDeviceDrop = (overId: string) => {
    if (!dragDevice || !deviceOrder) return;
    if (dragDevice === overId) return;
    const next = deviceOrder.slice();
    const a = next.indexOf(dragDevice);
    const b = next.indexOf(overId);
    if (a < 0 || b < 0) return;
    next.splice(a, 1);
    next.splice(b, 0, dragDevice);
    setDeviceOrder(next);
    setDragDevice(null);
  };

  // --- Devices grid sizes (w/h in grid units) ---
  const devSizesKey = useMemo(() => (userId ? `aura:devgrid:sizes:${userId}` : ''), [userId]);
  const [devSizes, setDevSizes] = useState<Record<string, { w: number; h: number }>>({});
  const devGridRef = useRef<HTMLDivElement | null>(null);
  const devResizeRef = useRef<{
    id: string;
    startX: number;
    startY: number;
    startW: number;
    startH: number;
  } | null>(null);
  useEffect(() => {
    // bootstrap from storage
    if (!devSizesKey) return;
    try {
      const raw = localStorage.getItem(devSizesKey);
      if (raw) setDevSizes(JSON.parse(raw) as Record<string, { w: number; h: number }>);
    } catch {}
  }, [devSizesKey]);
  useEffect(() => {
    // ensure all current ids have defaults
    const ids = (sortedDevices || []).map((d) => String(d.id));
    setDevSizes((prev) => {
      const next = { ...prev } as Record<string, { w: number; h: number }>;
      let mutated = false;
      for (const id of ids)
        if (!next[id]) {
          next[id] = { w: 6, h: 5 };
          mutated = true;
        }
      return mutated ? next : prev;
    });
  }, [sortedDevices]);
  useEffect(() => {
    if (!devSizesKey) return;
    try {
      localStorage.setItem(devSizesKey, JSON.stringify(devSizes));
    } catch {}
  }, [devSizesKey, devSizes]);
  const startDevResize = (id: string) => (e: React.PointerEvent) => {
    const sz = devSizes[id] || { w: 6, h: 5 };
    try {
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    } catch {}
    devResizeRef.current = { id, startX: e.clientX, startY: e.clientY, startW: sz.w, startH: sz.h };
    e.preventDefault();
    e.stopPropagation();
  };
  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const st = devResizeRef.current;
      if (!st) return;
      const gridEl = devGridRef.current;
      if (!gridEl) return;
      const cs = getComputedStyle(gridEl);
      const colGap = parseFloat(cs.columnGap || '0') || 0;
      const rowGap = parseFloat(cs.rowGap || '0') || 0;
      const gridWidth = gridEl.clientWidth || 1;
      const colWidth = (gridWidth - colGap * (gridCols - 1)) / gridCols;
      const incW = colWidth + colGap;
      const incH = rowUnit + rowGap;
      const dx = (e.clientX - st.startX) / (zoom || 1);
      const dy = (e.clientY - st.startY) / (zoom || 1);
      const nextW = Math.max(3, Math.min(gridCols, Math.round(st.startW + dx / incW)));
      const nextH = Math.max(3, Math.min(100, Math.round(st.startH + dy / incH)));
      setDevSizes((prev) => ({ ...prev, [st.id]: { w: nextW, h: nextH } }));
    };
    const onUp = () => {
      devResizeRef.current = null;
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  }, [gridCols, rowUnit, zoom]);
  // Start pointer-based height resize for a device card
  const startResizeDevice = (id: string) => (e: React.PointerEvent) => {
    const parent = (e.currentTarget as HTMLElement).parentElement as HTMLElement | null;
    const curH = Number(parent?.style.minHeight?.replace('px', '')) || deviceHeights[id] || 96;
    try {
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    } catch {}
    deviceResizeRef.current = { id, startY: e.clientY, startH: curH };
    e.preventDefault();
    e.stopPropagation();
  };
  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const st = deviceResizeRef.current;
      if (!st) return;
      const dy = (e.clientY - st.startY) / (zoom || 1);
      const next = Math.max(64, Math.min(360, Math.round(st.startH + dy)));
      setDeviceHeights((prev) => ({ ...prev, [st.id]: next }));
    };
    const onUp = () => {
      deviceResizeRef.current = null;
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
  }, [zoom]);

  if (needsRegistration === null) {
    return <div className="w-full h-full grid place-items-center text-white/60">Cargando…</div>;
  }

  if (needsRegistration) {
    return <HomeRegistration userId={userId} onDone={() => setNeedsRegistration(false)} />;
  }

  return (
    <div className="w-full h-full flex flex-col select-none">
      <header className="px-2 md:px-4">
        <div
          className="relative mt-1 h-16 md:h-[72px] rounded-2xl bg-white/5 ring-1 ring-white/10 px-4 py-2 text-white/80 flex items-center justify-between"
          role="region"
          aria-label="Resumen del hogar"
        >
          {/* Left */}
          <div className="font-medium tracking-wide flex items-center gap-2">
            <span className="inline-block w-4 h-4 rounded-sm bg-[#8B3DFF]" />
            Smart Home
          </div>

          {/* Center cluster */}
          <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-5">
            <span className="hidden sm:inline text-sm text-white/70">{home?.city || '—'}</span>
            <div className="flex flex-col items-center gap-[6px]">
              <button
                onClick={() => setModal('temp')}
                className="w-10 h-10 rounded-full bg-white/10 grid place-items-center text-white/90 hover:bg-white/20 transition"
                title={
                  isWxOk
                    ? `Temperatura: ${(wx as any).temperature} °C`
                    : 'Temperatura no disponible'
                }
              >
                <WiThermometer className="w-6 h-6" />
              </button>
              <span className="text-[9px] text-white/70 min-w-[44px] leading-none text-center">
                {tempLabel}
              </span>
            </div>
            <div className="flex flex-col items-center gap-[6px]">
              <button
                onClick={() => setModal('hum')}
                className="w-10 h-10 rounded-full bg-white/10 grid place-items-center text-white/90 hover:bg-white/20 transition"
                title={isWxOk ? `Humedad: ${(wx as any).humidity} %` : 'Humedad no disponible'}
              >
                <WiHumidity className="w-6 h-6" />
              </button>
              <span className="text-[9px] text-white/70 min-w-[44px] leading-none text-center">
                {humLabel}
              </span>
            </div>
            <div className="flex flex-col items-center gap-[6px]">
              <button
                onClick={() => setModal('cond')}
                className="w-10 h-10 rounded-full bg-white/10 grid place-items-center text-white/90 hover:bg-white/20 transition"
                title={isWxOk ? `Condición: ${condText}` : 'Condición no disponible'}
              >
                {React.createElement(CondIcon, { className: 'w-6 h-6' })}
              </button>
              <span
                className="text-[9px] text-white/70 max-w-[140px] leading-none truncate text-center"
                title={condLabel}
              >
                {condLabel}
              </span>
            </div>
          </div>

          {/* Right actions */}
          <div className="flex items-center gap-2">
            {/* Zoom controls */}
            <div className="hidden sm:flex items-center gap-1 mr-1 text-white/70">
              <button
                className="w-8 h-9 rounded-lg bg-white/10 hover:bg-white/20"
                title="Zoom -"
                onClick={() => setZoom((z) => Math.max(0.6, Number((z - 0.1).toFixed(2))))}
              >
                <FiMinus className="mx-auto" />
              </button>
              <div className="px-2 text-sm tabular-nums">{Math.round(zoom * 100)}%</div>
              <button
                className="w-8 h-9 rounded-lg bg-white/10 hover:bg-white/20"
                title="Zoom +"
                onClick={() => setZoom((z) => Math.min(1.6, Number((z + 0.1).toFixed(2))))}
              >
                <FiPlus className="mx-auto" />
              </button>
            </div>
            <button
              onClick={() => setEditMode((v) => !v)}
              className={`px-3 h-9 rounded-lg text-sm inline-flex items-center gap-2 ${editMode ? 'bg-[#8B3DFF] text-white' : 'bg-white/10 hover:bg-white/20 text-white/80'}`}
              title={editMode ? 'Salir de configuracion' : 'Configurar dashboard'}
            >
              <FiSettings className="w-4 h-4" />
              {editMode ? 'Listo' : 'Configurar'}
            </button>
            <button
              onClick={() => setTelemetryMode((v) => !v)}
              className={`px-3 h-9 rounded-lg text-sm inline-flex items-center gap-2 ${telemetryMode ? 'bg-[#8B3DFF] text-white' : 'bg-white/10 hover:bg-white/20 text-white/80'}`}
              title={telemetryMode ? 'Ocultar telemetria' : 'Ver telemetria'}
            >
              <FiBarChart2 className="w-4 h-4" />
              Telemetria
            </button>
            <button
              onClick={() => setEditMode((v) => !v)}
              className={`hidden px-3 h-9 rounded-lg text-sm inline-flex items-center gap-2 ${editMode ? 'bg-[#8B3DFF] text-white' : 'bg-white/10 hover:bg-white/20 text-white/80'}`}
              title={editMode ? 'Salir de configuración' : 'Configurar dashboard'}
            >
              <FiSettings className="w-4 h-4" />
              {editMode ? 'Listo' : 'Configurar'}
            </button>
            <button
              onClick={async () => {
                try {
                  const devs = await listDevices();
                  if (devs?.length) setDevices(withOverrides(devs));
                  else setDevices(seedSimDevices());
                } catch {
                  setDevices(seedSimDevices());
                }
                setAddModal(true);
              }}
              className="hidden px-3 h-9 rounded-lg bg-white/10 hover:bg-white/20 text-white/80 text-sm inline-flex items-center gap-2"
              title="Agregar dispositivo"
            >
              <FiPlus className="w-4 h-4" />
              Agregar dispositivo
            </button>
          </div>
        </div>
      </header>
      <section className="flex-1 min-h-0 px-2 md:px-4 py-4">
        {/* Scrollbar hidden for the zoomable area */}
        <style>{`#dash-scroll{scrollbar-width:none;-ms-overflow-style:none}#dash-scroll::-webkit-scrollbar{display:none;width:0;height:0}`}</style>
        {/* Zoomable content wrapper; header/footer remain intact */}
        <div
          id="dash-scroll"
          className="relative h-full overflow-auto"
          style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
        >
          {telemetryMode && (
            <div className="absolute inset-0 z-10 overflow-auto p-2 md:p-4 bg-black/30 backdrop-blur-sm">
              <div className="mb-3 flex items-center justify-start text-white/90">
                <div className="font-medium tracking-wide flex items-center gap-2">
                  <span className="inline-block w-4 h-4 rounded-sm bg-[#8B3DFF]" /> Telemetria
                  (simulada)
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                {(devices && devices.length ? devices : seedSimDevices()).map((d) => {
                  const metric = getMetricFor(d);
                  if (!metric) return null;
                  const second = extraSeriesFor(d)[0] || null;
                  const devMeta = telemetryRef.current[d.id];
                  const nowTs = Date.now();
                  const s1 = devMeta?.series?.[metric.key]?.samples;
                  const s1Samples =
                    s1 && s1.length > 0
                      ? s1
                      : [{ t: nowTs, y: encodeMetric(metric.key, (d as any)[metric.key]) }];
                  const s1IsBool = ['is_on', 'opened', 'motion_detected', 'muted'].includes(
                    metric.key
                  );
                  const s1IsEnum = !!(ENUM_SERIES as any)[metric.key];
                  const s1Kind: 'bool' | 'enum' | 'num' = s1IsBool
                    ? 'bool'
                    : s1IsEnum
                      ? 'enum'
                      : 'num';
                  let g1 = aggregateSeries(s1Samples, {
                    kind: s1Kind,
                    enumKey: s1IsEnum ? metric.key : undefined,
                    res: telemetryResolution,
                  });
                  if (g1.length < 2) {
                    const raw = [...s1Samples].sort((a, b) => a.t - b.t);
                    if (raw.length >= 2) g1 = raw.slice(-2);
                    else if (raw.length === 1)
                      g1 = [
                        {
                          t: raw[0].t - Math.max(10000, bucketMsFor(telemetryResolution) / 6),
                          y: raw[0].y,
                        },
                        raw[0],
                      ];
                  }
                  let s2Samples: TelemetrySample[] | null = null;
                  let s2IsBool = false,
                    s2IsEnum = false,
                    s2Kind: 'bool' | 'enum' | 'num' = 'num';
                  if (second) {
                    const t2 = devMeta?.series?.[second.key]?.samples;
                    s2Samples =
                      t2 && t2.length > 0
                        ? t2
                        : [{ t: nowTs, y: encodeMetric(second.key, (d as any)[second.key]) }];
                    s2IsBool = ['is_on', 'opened', 'motion_detected', 'muted'].includes(second.key);
                    s2IsEnum = !!(ENUM_SERIES as any)[second.key];
                    s2Kind = s2IsBool ? 'bool' : s2IsEnum ? 'enum' : 'num';
                  }
                  let g2: TelemetrySample[] | null = null;
                  if (s2Samples) {
                    g2 = aggregateSeries(s2Samples, {
                      kind: s2Kind,
                      enumKey: s2IsEnum ? second!.key : undefined,
                      res: telemetryResolution,
                    });
                    if (g2.length < 2) {
                      const raw = [...s2Samples].sort((a, b) => a.t - b.t);
                      if (raw.length >= 2) g2 = raw.slice(-2);
                      else if (raw.length === 1)
                        g2 = [
                          {
                            t: raw[0].t - Math.max(10000, bucketMsFor(telemetryResolution) / 6),
                            y: raw[0].y,
                          },
                          raw[0],
                        ];
                    }
                  }
                  // Merge by bucket time
                  const times = new Set<number>(g1.map((p) => p.t));
                  if (g2) for (const p of g2) times.add(p.t);
                  const sortedTimes = Array.from(times).sort((a, b) => a - b);
                  const map1 = new Map<number, number>();
                  g1.forEach((p) => map1.set(p.t, p.y));
                  const map2 = new Map<number, number>();
                  if (g2) g2.forEach((p) => map2.set(p.t, p.y));
                  const data = sortedTimes.map((tval) => ({
                    t: ((): string => {
                      const dt = new Date(tval);
                      if (telemetryResolution === '1d') return dt.toLocaleDateString();
                      return dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                    })(),
                    y: map1.get(tval),
                    y2: g2 ? map2.get(tval) : undefined,
                  }));
                  const latest = s1Samples[s1Samples.length - 1]?.y ?? 0;
                  const color = telemetryColors[d.id] || defaultTeleColor(d.id);
                  const color2 = defaultTeleColor(d.id + ':2');
                  const domain1: any = s1IsBool
                    ? [0, 1]
                    : s1IsEnum
                      ? [0, (ENUM_SERIES as any)[metric.key].length - 1]
                      : ['auto', 'auto'];
                  const domain2: any = second
                    ? s2IsBool
                      ? [0, 1]
                      : s2IsEnum
                        ? [0, (ENUM_SERIES as any)[second.key].length - 1]
                        : ['auto', 'auto']
                    : undefined;
                  const tickFmt1 = (v: any) =>
                    s1IsBool || s1IsEnum ? labelFromEncoded(metric.key, Number(v)) : String(v);
                  const tickFmt2 = (v: any) => {
                    if (!second) return String(v);
                    return s2IsBool || s2IsEnum
                      ? labelFromEncoded(second.key, Number(v))
                      : String(v);
                  };
                  const tooltipFmt = (value: any, name: any, props: any) => {
                    const dk: string = props?.dataKey;
                    if (dk === 'y2' && second)
                      return [
                        s2IsBool || s2IsEnum
                          ? labelFromEncoded(second.key, Number(value))
                          : String(value),
                        second.label,
                      ];
                    return [
                      s1IsBool || s1IsEnum
                        ? labelFromEncoded(metric.key, Number(value))
                        : String(value),
                      metric.label,
                    ];
                  };
                  const latestLabel =
                    s1IsBool || s1IsEnum
                      ? labelFromEncoded(metric.key, Number(latest))
                      : String(latest);
                  const yWidth1 = s1IsEnum ? 64 : s1IsBool ? 42 : 36;
                  const yWidth2 = second ? (s2IsEnum ? 64 : s2IsBool ? 42 : 40) : 0;
                  return (
                    <div
                      key={d.id}
                      className="rounded-2xl bg-white/5 ring-1 ring-white/10 p-3 text-white/80"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <div className="text-sm">
                          <div className="font-medium text-white">{d.name}</div>
                          <div className="text-white/60">
                            {metric.label}
                            {second ? ' + ' + second.label : ''}
                          </div>
                        </div>
                        <div className="text-right text-white/70 flex items-center gap-2">
                          <input
                            type="color"
                            title="Cambiar color"
                            className="w-5 h-5 rounded-md border border-white/20 bg-transparent cursor-pointer"
                            value={color}
                            onChange={(e) =>
                              setTelemetryColors((prev) => ({ ...prev, [d.id]: e.target.value }))
                            }
                          />
                          <div className="text-lg font-semibold text-white">{latestLabel}</div>
                        </div>
                      </div>
                      <div className="h-32">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
                            <defs>
                              <linearGradient id={`tele-${d.id}`} x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={color} stopOpacity={0.6} />
                                <stop offset="95%" stopColor={color} stopOpacity={0.05} />
                              </linearGradient>
                              {second && (
                                <linearGradient id={`tele-${d.id}-2`} x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%" stopColor={color2} stopOpacity={0.6} />
                                  <stop offset="95%" stopColor={color2} stopOpacity={0.05} />
                                </linearGradient>
                              )}
                            </defs>
                            <XAxis dataKey="t" hide />
                            <YAxis
                              tick={{ fill: '#CBD5E1' }}
                              stroke="#64748B"
                              width={yWidth1}
                              domain={domain1}
                              allowDecimals={!s1IsBool && !s1IsEnum}
                              tickFormatter={tickFmt1}
                            />
                            {second && (
                              <YAxis
                                yAxisId="R"
                                orientation="right"
                                tick={{ fill: '#CBD5E1' }}
                                stroke="#64748B"
                                width={yWidth2}
                                domain={domain2 as any}
                                allowDecimals={!s2IsBool && !s2IsEnum}
                                tickFormatter={tickFmt2}
                              />
                            )}
                            <Tooltip
                              contentStyle={{
                                background: 'rgba(17, 24, 39, 0.85)',
                                border: '1px solid rgba(255,255,255,0.1)',
                                borderRadius: 12,
                              }}
                              labelStyle={{ color: '#E2E8F0' }}
                              itemStyle={{ color: '#E2E8F0' }}
                              formatter={tooltipFmt as any}
                            />
                            <Area
                              type="monotone"
                              dataKey="y"
                              name={metric.label}
                              stroke={color}
                              fill={`url(#tele-${d.id})`}
                              strokeWidth={2}
                              connectNulls
                            />
                            {second && (
                              <Area
                                type="monotone"
                                dataKey="y2"
                                yAxisId="R"
                                name={second.label}
                                stroke={color2}
                                fill={`url(#tele-${d.id}-2)`}
                                strokeWidth={2}
                                connectNulls
                              />
                            )}
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          <div
            style={{
              transform: `scale(${zoom})`,
              transformOrigin: '0 0',
              width: `${100 / zoom}%`,
              display: telemetryMode ? 'none' : undefined,
            }}
          >
            {/* Reorderable + resizable dashboard grid */}
            <div
              className="grid grid-flow-dense gap-3"
              style={{
                gridTemplateColumns: `repeat(${gridCols}, minmax(0, 1fr))`,
                gridAutoRows: `${rowUnit}px`,
              }}
              onDragOver={(e) => (editMode ? e.preventDefault() : undefined)}
              onDrop={dropAtEnd}
              ref={gridRef}
            >
              {tileOrder.map((tid) => {
                const sz = tileSizes[tid] || { w: 6, h: 5 };
                const cls = `relative group`;
                return (
                  <div
                    key={tid}
                    className={cls}
                    style={{
                      gridColumn: `span ${Math.max(1, Math.min(gridCols, Math.round(sz.w)))}`,
                      gridRow: `span ${Math.max(1, Math.round(sz.h))}`,
                    }}
                    draggable={editMode}
                    onDragStart={startDrag(tid)}
                    onDragOver={overTile(tid)}
                    onDrop={dropOnTile(tid)}
                    onDragEnd={() => setDragging(null)}
                    aria-grabbed={editMode && dragging === tid}
                  >
                    {tid === 'time' && (
                      <AutoHideCard
                        icon={<FiClock className="w-6 h-6" />}
                        className={`h-full ${dragging === tid ? 'ring-2 ring-[#8B3DFF]' : ''}`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="text-sm text-white/70 flex items-center gap-2">
                            <IconChip color="#8B3DFF">
                              <FiClock className="w-4 h-4" />
                            </IconChip>{' '}
                            Tiempo
                          </div>
                          <div className="text-xs text-white/50">{now.toLocaleDateString()}</div>
                        </div>
                        <div className="mt-2 text-2xl md:text-3xl font-semibold text-white">
                          {now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </AutoHideCard>
                    )}
                    {tid === 'lights' && (
                      <AutoHideCard
                        icon={<FiSun className="w-6 h-6" />}
                        className={`h-full ${dragging === tid ? 'ring-2 ring-[#8B3DFF]' : ''}`}
                      >
                        <div className="text-sm text-white/70 flex items-center gap-2">
                          <IconChip color="#F59E0B">
                            <FiSun className="w-4 h-4" />
                          </IconChip>{' '}
                          Luces
                        </div>
                        <div className="mt-2 text-white text-xl font-semibold">
                          {lightsOn} encendidas
                        </div>
                        <div className="text-xs text-white/50">{totalDevices} dispositivos</div>
                      </AutoHideCard>
                    )}
                    {tid === 'aqi' && (
                      <AutoHideCard
                        icon={<WiFog className="w-6 h-6" />}
                        className={`h-full ${dragging === tid ? 'ring-2 ring-[#8B3DFF]' : ''}`}
                      >
                        <div className="text-sm text-white/70 flex items-center gap-2">
                          <IconChip color="#22C55E">
                            <WiFog className="w-4 h-4" />
                          </IconChip>{' '}
                          Calidad del aire (AQI)
                        </div>
                        <div className="mt-2 text-white text-xl font-semibold">
                          {isWxOk && (wx as any).air_quality_index != null
                            ? Math.round((wx as any).air_quality_index as number)
                            : '—'}
                        </div>
                        <div className="text-xs text-white/50">
                          Fuente: {isWxOk ? (wx as any).source : '—'}
                        </div>
                      </AutoHideCard>
                    )}
                    {tid === 'temp' && (
                      <AutoHideCard
                        icon={<WiThermometer className="w-6 h-6" />}
                        className={`h-full ${dragging === tid ? 'ring-2 ring-[#8B3DFF]' : ''}`}
                      >
                        <div className="text-sm text-white/70 flex items-center gap-2">
                          <IconChip color="#EF4444">
                            <WiThermometer className="w-4 h-4" />
                          </IconChip>{' '}
                          Temperatura
                        </div>
                        <div className="mt-2 text-white text-3xl font-semibold">
                          {isWxOk && (wx as any).temperature != null
                            ? `${(wx as any).temperature} °C`
                            : '—'}
                        </div>
                        <div className="text-xs text-white/50">{home?.city || ''}</div>
                      </AutoHideCard>
                    )}
                    {tid === 'hum' && (
                      <AutoHideCard
                        icon={<WiHumidity className="w-6 h-6" />}
                        className={`h-full ${dragging === tid ? 'ring-2 ring-[#8B3DFF]' : ''}`}
                      >
                        <div className="text-sm text-white/70 flex items-center gap-2">
                          <IconChip color="#06B6D4">
                            <WiHumidity className="w-4 h-4" />
                          </IconChip>{' '}
                          Humedad
                        </div>
                        <div className="mt-2 text-white text-3xl font-semibold">
                          {isWxOk && (wx as any).humidity != null
                            ? `${(wx as any).humidity} %`
                            : '—'}
                        </div>
                        <div className="text-xs text-white/50">Relativa</div>
                      </AutoHideCard>
                    )}
                    {tid === 'chart' && (
                      <AutoHideCard
                        icon={<WiThermometer className="w-6 h-6\" />}
                        className={`h-full ${dragging === tid ? 'ring-2 ring-[#8B3DFF]' : ''}`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="text-sm text-white/70 flex items-center gap-2">
                            <IconChip color="#8B3DFF">
                              <WiThermometer className="w-4 h-4" />
                            </IconChip>{' '}
                            Temperatura y humedad
                          </div>
                          <button
                            onClick={() => setModal('temp')}
                            className="text-xs px-2 h-7 rounded-md bg-white/10 hover:bg-white/20"
                            title="Ampliar"
                          >
                            Ampliar
                          </button>
                        </div>
                        <div className="mt-2 h-[calc(100%-2rem)]" style={{ minHeight: 220 }}>
                          {history ? (
                            <ResponsiveContainer width="100%" height="100%">
                              <AreaChart
                                data={history.map((p) => ({
                                  t: p.time.slice(11, 16),
                                  temp: p.temperature,
                                  hum: p.humidity,
                                }))}
                                margin={{ top: 5, right: 16, left: 0, bottom: 10 }}
                              >
                                <defs>
                                  <linearGradient id="dashGradT" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#8B3DFF" stopOpacity={0.7} />
                                    <stop offset="95%" stopColor="#8B3DFF" stopOpacity={0.05} />
                                  </linearGradient>
                                  <linearGradient id="dashGradH" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.7} />
                                    <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.05} />
                                  </linearGradient>
                                </defs>
                                <CartesianGrid stroke="#ffffff12" />
                                <XAxis dataKey="t" minTickGap={24} stroke="#9aa4b2" />
                                <YAxis stroke="#9aa4b2" />
                                <Tooltip
                                  contentStyle={{
                                    background: '#0b1020',
                                    border: '1px solid #22293f',
                                    color: '#cbd5e1',
                                  }}
                                />
                                <Area
                                  type="monotone"
                                  dataKey="temp"
                                  stroke="#8B3DFF"
                                  fill="url(#dashGradT)"
                                />
                                <Area
                                  type="monotone"
                                  dataKey="hum"
                                  stroke="#22d3ee"
                                  fill="url(#dashGradH)"
                                />
                              </AreaChart>
                            </ResponsiveContainer>
                          ) : (
                            <div className="w-full h-full grid place-items-center text-white/60">
                              Sin datos
                            </div>
                          )}
                        </div>
                      </AutoHideCard>
                    )}
                    {tid === 'location' && (
                      <AutoHideCard
                        icon={<FiCloud className="w-6 h-6" />}
                        className={`h-full ${dragging === tid ? 'ring-2 ring-[#8B3DFF]' : ''}`}
                      >
                        <div className="text-sm text-white/70 flex items-center gap-2">
                          <IconChip color="#6366F1">
                            <FiCloud className="w-4 h-4" />
                          </IconChip>{' '}
                          Ubicación
                        </div>
                        <div className="mt-1 text-white font-medium">
                          {home?.city || '—'} {home?.country ? `· ${home.country}` : ''}
                        </div>
                        <div className="mt-2 overflow-hidden rounded-lg ring-1 ring-white/10">
                          <div className="h-36 w-full bg-gradient-to-br from-[#1c2240] via-[#242a52] to-[#0b1020] relative">
                            <div
                              className="absolute inset-0 opacity-50"
                              style={{
                                backgroundImage:
                                  'radial-gradient(circle at 30% 20%, #8B3DFF55, transparent 50%), radial-gradient(circle at 70% 80%, #22d3ee44, transparent 45%)',
                              }}
                            />
                            <div className="absolute left-3 bottom-2 text-white/80 text-sm">
                              {condLabel}
                            </div>
                          </div>
                        </div>
                        <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-white/60">
                          <div>Lat: {home?.latitude ?? '—'}</div>
                          <div>Lon: {home?.longitude ?? '—'}</div>
                        </div>
                      </AutoHideCard>
                    )}
                    {tid === 'actions' && (
                      <AutoHideCard
                        icon={<FiSun className="w-6 h-6" />}
                        className={`h-full ${dragging === tid ? 'ring-2 ring-[#8B3DFF]' : ''}`}
                      >
                        <div className="grid grid-cols-3 gap-3">
                          <ActionCard
                            color="#f97316"
                            label="Luces: encender"
                            onClick={async () => {
                              try {
                                await Promise.all(
                                  devices
                                    .filter((d) => d.type === 'light_bulb' && !d.is_on)
                                    .map((d) => powerDevice(d.id, 'on'))
                                );
                                const devs = await listDevices();
                                setDevices(
                                  Array.isArray(devs) && devs.length > 0
                                    ? withOverrides(devs)
                                    : seedSimDevices()
                                );
                              } catch {}
                            }}
                          />
                          <ActionCard
                            color="#fb923c"
                            label="Luces: apagar"
                            onClick={async () => {
                              try {
                                await Promise.all(
                                  devices
                                    .filter((d) => d.type === 'light_bulb' && d.is_on)
                                    .map((d) => powerDevice(d.id, 'off'))
                                );
                                const devs = await listDevices();
                                setDevices(
                                  Array.isArray(devs) && devs.length > 0
                                    ? withOverrides(devs)
                                    : seedSimDevices()
                                );
                              } catch {}
                            }}
                          />
                          <ActionCard
                            color="#22d3ee"
                            label="Plugs: togglear"
                            onClick={async () => {
                              try {
                                await Promise.all(
                                  devices
                                    .filter((d) => d.type === 'smart_plug')
                                    .map((d) => powerDevice(d.id, 'toggle'))
                                );
                                const devs = await listDevices();
                                setDevices(withOverrides(devs));
                              } catch {}
                            }}
                          />
                        </div>
                      </AutoHideCard>
                    )}
                    {tid === 'gauge' && (
                      <AutoHideCard
                        icon={<WiDaySunny className="w-6 h-6" />}
                        className={`h-full ${dragging === tid ? 'ring-2 ring-[#8B3DFF]' : ''}`}
                      >
                        <div className="text-sm text-white/70">Progreso de confort</div>
                        <div className="mt-2 flex items-center gap-6 h-[calc(100%-0.5rem)]">
                          <Gauge
                            percent={Math.max(
                              0,
                              Math.min(
                                100,
                                Number(
                                  isWxOk && (wx as any).humidity != null ? (wx as any).humidity : 0
                                )
                              )
                            )}
                          />
                          <div className="text-white/70 text-sm">
                            <div>
                              Temperatura:{' '}
                              {isWxOk && (wx as any).temperature != null
                                ? `${(wx as any).temperature} °C`
                                : '—'}
                            </div>
                            <div>
                              Humedad:{' '}
                              {isWxOk && (wx as any).humidity != null
                                ? `${(wx as any).humidity} %`
                                : '—'}
                            </div>
                          </div>
                        </div>
                      </AutoHideCard>
                    )}
                    {tid === 'summary' && (
                      <AutoHideCard
                        icon={<FiCloud className="w-6 h-6" />}
                        className={`h-full ${dragging === tid ? 'ring-2 ring-[#8B3DFF]' : ''}`}
                      >
                        <div className="text-sm text-white/70">Resumen</div>
                        <div className="mt-2 grid grid-cols-3 gap-2 text-white">
                          <Stat label="Dispositivos" value={String(totalDevices)} />
                          <Stat label="Luces ON" value={String(lightsOn)} />
                          <Stat label="Plugs ON" value={String(plugsOn)} />
                        </div>
                      </AutoHideCard>
                    )}
                    {/* Corner resize handle */}
                    <div
                      className="absolute right-1 bottom-1 w-3 h-3 rounded-sm bg-white/70 cursor-se-resize opacity-70 group-hover:opacity-100"
                      onPointerDown={onResizeStart(tid)}
                      title="Arrastra para redimensionar"
                    />
                  </div>
                );
              })}
            </div>
            <div className="hidden grid grid-cols-12 gap-3 auto-rows-min">
              {/* Left column stats */}
              <div className="col-span-12 lg:col-span-3 grid gap-3 content-start">
                <Card>
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-white/70 flex items-center gap-2">
                      <IconChip color="#8B3DFF">
                        <FiClock className="w-4 h-4" />
                      </IconChip>{' '}
                      Tiempo
                    </div>
                    <div className="text-xs text-white/50">{now.toLocaleDateString()}</div>
                  </div>
                  <div className="mt-2 text-2xl md:text-3xl font-semibold text-white">
                    {now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </Card>
                <Card>
                  <div className="text-sm text-white/70 flex items-center gap-2">
                    <IconChip color="#F59E0B">
                      <FiSun className="w-4 h-4" />
                    </IconChip>{' '}
                    Luces
                  </div>
                  <div className="mt-2 text-white text-xl font-semibold">{lightsOn} encendidas</div>
                  <div className="text-xs text-white/50">{totalDevices} dispositivos</div>
                </Card>
                <Card>
                  <div className="text-sm text-white/70 flex items-center gap-2">
                    <IconChip color="#22C55E">
                      <WiFog className="w-4 h-4" />
                    </IconChip>{' '}
                    Calidad del aire (AQI)
                  </div>
                  <div className="mt-2 text-white text-xl font-semibold">
                    {isWxOk && (wx as any).air_quality_index != null
                      ? Math.round((wx as any).air_quality_index as number)
                      : '—'}
                  </div>
                  <div className="text-xs text-white/50">
                    Fuente: {isWxOk ? (wx as any).source : '—'}
                  </div>
                </Card>
                <Card>
                  <div className="text-sm text-white/70 flex items-center gap-2">
                    <IconChip color="#EF4444">
                      <WiThermometer className="w-4 h-4" />
                    </IconChip>{' '}
                    Temperatura
                  </div>
                  <div className="mt-2 text-white text-3xl font-semibold">
                    {isWxOk && (wx as any).temperature != null
                      ? `${(wx as any).temperature} °C`
                      : '—'}
                  </div>
                  <div className="text-xs text-white/50">{home?.city || ''}</div>
                </Card>
                <Card>
                  <div className="text-sm text-white/70 flex items-center gap-2">
                    <IconChip color="#06B6D4">
                      <WiHumidity className="w-4 h-4" />
                    </IconChip>{' '}
                    Humedad
                  </div>
                  <div className="mt-2 text-white text-3xl font-semibold">
                    {isWxOk && (wx as any).humidity != null ? `${(wx as any).humidity} %` : '—'}
                  </div>
                  <div className="text-xs text-white/50">Relativa</div>
                </Card>
              </div>

              {/* Middle chart */}
              <div className="col-span-12 lg:col-span-5">
                <Card>
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-white/70 flex items-center gap-2">
                      <IconChip color="#8B3DFF">
                        <WiThermometer className="w-4 h-4" />
                      </IconChip>{' '}
                      Temperatura y humedad
                    </div>
                    <button
                      onClick={() => setModal('temp')}
                      className="text-xs px-2 h-7 rounded-md bg-white/10 hover:bg-white/20"
                      title="Ampliar"
                    >
                      Ampliar
                    </button>
                  </div>
                  <div className="mt-2 h-48 md:h-56">
                    {history ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart
                          data={history.map((p) => ({
                            t: p.time.slice(11, 16),
                            temp: p.temperature,
                            hum: p.humidity,
                          }))}
                          margin={{ top: 5, right: 16, left: 0, bottom: 10 }}
                        >
                          <defs>
                            <linearGradient id="dashGradT" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#8B3DFF" stopOpacity={0.7} />
                              <stop offset="95%" stopColor="#8B3DFF" stopOpacity={0.05} />
                            </linearGradient>
                            <linearGradient id="dashGradH" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.7} />
                              <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.05} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid stroke="#ffffff12" />
                          <XAxis dataKey="t" minTickGap={24} stroke="#9aa4b2" />
                          <YAxis stroke="#9aa4b2" />
                          <Tooltip
                            contentStyle={{
                              background: '#0b1020',
                              border: '1px solid #22293f',
                              color: '#cbd5e1',
                            }}
                          />
                          <Area
                            type="monotone"
                            dataKey="temp"
                            stroke="#8B3DFF"
                            fill="url(#dashGradT)"
                          />
                          <Area
                            type="monotone"
                            dataKey="hum"
                            stroke="#22d3ee"
                            fill="url(#dashGradH)"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="w-full h-full grid place-items-center text-white/60">
                        Sin datos
                      </div>
                    )}
                  </div>
                </Card>
              </div>

              {/* Right location card */}
              <div className="col-span-12 lg:col-span-4">
                <Card>
                  <div className="text-sm text-white/70 flex items-center gap-2">
                    <IconChip color="#6366F1">
                      <FiCloud className="w-4 h-4" />
                    </IconChip>{' '}
                    Ubicación
                  </div>
                  <div className="mt-1 text-white font-medium">
                    {home?.city || '—'} {home?.country ? `· ${home.country}` : ''}
                  </div>
                  <div className="mt-2 overflow-hidden rounded-lg ring-1 ring-white/10">
                    <div className="h-36 w-full bg-gradient-to-br from-[#1c2240] via-[#242a52] to-[#0b1020] relative">
                      <div
                        className="absolute inset-0 opacity-50"
                        style={{
                          backgroundImage:
                            'radial-gradient(circle at 30% 20%, #8B3DFF55, transparent 50%), radial-gradient(circle at 70% 80%, #22d3ee44, transparent 45%)',
                        }}
                      />
                      <div className="absolute left-3 bottom-2 text-white/80 text-sm">
                        {condLabel}
                      </div>
                    </div>
                  </div>
                  <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-white/60">
                    <div>Lat: {home?.latitude ?? '—'}</div>
                    <div>Lon: {home?.longitude ?? '—'}</div>
                  </div>
                </Card>
              </div>

              {/* Action row */}
              <div className="col-span-12 grid grid-cols-12 gap-3">
                <ActionCard
                  className="col-span-12 md:col-span-4"
                  color="#f97316"
                  label="Luces: encender"
                  onClick={async () => {
                    try {
                      await Promise.all(
                        devices
                          .filter((d) => d.type === 'light_bulb' && !d.is_on)
                          .map((d) => powerDevice(d.id, 'on'))
                      );
                      const devs = await listDevices();
                      setDevices(devs?.length ? withOverrides(devs) : seedSimDevices());
                    } catch {
                      setDevices(seedSimDevices());
                    }
                  }}
                />
                <ActionCard
                  className="col-span-12 md:col-span-4"
                  color="#fb923c"
                  label="Luces: apagar"
                  onClick={async () => {
                    try {
                      await Promise.all(
                        devices
                          .filter((d) => d.type === 'light_bulb' && d.is_on)
                          .map((d) => powerDevice(d.id, 'off'))
                      );
                      const devs = await listDevices();
                      setDevices(devs?.length ? withOverrides(devs) : seedSimDevices());
                    } catch {
                      setDevices(seedSimDevices());
                    }
                  }}
                />
                <ActionCard
                  className="col-span-12 md:col-span-4"
                  color="#22d3ee"
                  label="Plugs: togglear"
                  onClick={async () => {
                    try {
                      await Promise.all(
                        devices
                          .filter((d) => d.type === 'smart_plug')
                          .map((d) => powerDevice(d.id, 'toggle'))
                      );
                      const devs = await listDevices();
                      setDevices(devs?.length ? withOverrides(devs) : seedSimDevices());
                    } catch {
                      setDevices(seedSimDevices());
                    }
                  }}
                />
              </div>

              {/* Gauge and summary */}
              <div className="col-span-12 grid grid-cols-12 gap-3">
                <Card className="col-span-12 md:col-span-6">
                  <div className="text-sm text-white/70">Progreso de confort</div>
                  <div className="mt-2 flex items-center gap-6">
                    <Gauge
                      percent={Math.max(
                        0,
                        Math.min(
                          100,
                          Number(isWxOk && (wx as any).humidity != null ? (wx as any).humidity : 0)
                        )
                      )}
                    />
                    <div className="text-white/70 text-sm">
                      <div>
                        Temperatura:{' '}
                        {isWxOk && (wx as any).temperature != null
                          ? `${(wx as any).temperature} °C`
                          : '—'}
                      </div>
                      <div>
                        Humedad:{' '}
                        {isWxOk && (wx as any).humidity != null ? `${(wx as any).humidity} %` : '—'}
                      </div>
                    </div>
                  </div>
                </Card>
                <Card className="col-span-12 md:col-span-6">
                  <div className="text-sm text-white/70">Resumen</div>
                  <div className="mt-2 grid grid-cols-3 gap-2 text-white">
                    <Stat label="Dispositivos" value={String(totalDevices)} />
                    <Stat label="Luces ON" value={String(lightsOn)} />
                    <Stat label="Plugs ON" value={String(plugsOn)} />
                  </div>
                </Card>
              </div>

              {/* Devices masonry (reordenable) */}
              <div className="col-span-12">
                <Masonry>
                  {sortedDevices.map((d) => (
                    <div
                      key={d.id}
                      draggable
                      onDragStart={() => setDragDevice(String(d.id))}
                      onDragOver={(e) => e.preventDefault()}
                      onDrop={() => onDeviceDrop(String(d.id))}
                      onDragEnd={() => setDragDevice(null)}
                      className="break-inside-avoid"
                    >
                      <DeviceCard
                        d={d}
                        onOpen={() => setEditing(d)}
                        onChanged={(next) =>
                          setDevices((prev) => prev.map((p) => (p.id === next.id ? next : p)))
                        }
                      />
                    </div>
                  ))}
                  {sortedDevices.length === 0 && (
                    <div className="rounded-xl bg-white/5 ring-1 ring-white/10 p-4 text-white/60">
                      Sin dispositivos
                    </div>
                  )}
                </Masonry>
              </div>
            </div>
            {/* Devices grid (resizable like tiles) */}
            <div className="mt-4">
              <div
                ref={devGridRef}
                className="grid grid-flow-dense gap-3"
                style={{
                  gridTemplateColumns: `repeat(${gridCols}, minmax(0, 1fr))`,
                  gridAutoRows: `${rowUnit}px`,
                }}
              >
                {sortedDevices.map((d) => {
                  const sid = String(d.id);
                  const sz = devSizes[sid] || { w: 6, h: 5 };
                  return (
                    <div
                      key={sid}
                      draggable
                      onDragStart={() => setDragDevice(sid)}
                      onDragOver={(e) => e.preventDefault()}
                      onDrop={() => onDeviceDrop(sid)}
                      onDragEnd={() => setDragDevice(null)}
                      className="relative"
                      style={{
                        gridColumn: `span ${Math.max(3, Math.min(gridCols, Math.round(sz.w)))}`,
                        gridRow: `span ${Math.max(3, Math.round(sz.h))}`,
                      }}
                    >
                      <DeviceCard
                        d={d}
                        onOpen={() => setEditing(d)}
                        onChanged={(next) =>
                          setDevices((prev) => prev.map((p) => (p.id === next.id ? next : p)))
                        }
                      />
                      <div
                        className="absolute right-1 bottom-1 w-3 h-3 rounded-sm bg-white/70 cursor-se-resize opacity-80"
                        onPointerDown={startDevResize(sid)}
                        title="Arrastra para redimensionar"
                      />
                    </div>
                  );
                })}
                {sortedDevices.length === 0 && (
                  <div className="rounded-xl bg-white/5 ring-1 ring-white/10 p-4 text-white/60">
                    Sin dispositivos
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>
      <footer className="px-2 md:px-4 pb-3">
        {telemetryMode && (
          <div className="relative mx-auto max-w-[980px] mb-2">
            <div className="rounded-2xl bg-white/5 ring-1 ring-white/10 p-2 md:p-3 text-white/80">
              <div className="flex flex-wrap items-center justify-center gap-2">
                <span className="text-white/60 text-sm mr-1">Resolucion:</span>
                {(
                  [
                    { key: '1m', label: '1 min' },
                    { key: '5m', label: '5 min' },
                    { key: '20m', label: '20 min' },
                    { key: '30m', label: '30 min' },
                    { key: '1h', label: '1 h' },
                    { key: '1d', label: 'Dias' },
                  ] as { key: TeleRes; label: string }[]
                ).map((opt) => (
                  <button
                    key={opt.key}
                    onClick={() => setTelemetryResolution(opt.key)}
                    className={`px-3 py-1.5 rounded-full text-sm transition ${
                      telemetryResolution === opt.key
                        ? 'bg-[#8B3DFF] text-white'
                        : 'bg-white/10 hover:bg-white/20 text-white/80'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
        <div
          className="relative mx-auto max-w-[820px]"
          style={{ display: telemetryMode ? 'none' : undefined }}
        >
          <div className="h-10 rounded-full bg-gradient-to-r from-white/5 via-white/8 to-white/5 ring-1 ring-white/10" />
          <button
            className="absolute left-1/2 -translate-x-1/2 -top-4 w-12 h-12 rounded-full grid place-items-center bg-[#8B3DFF] text-white shadow-[0_10px_40px_-10px_rgba(139,61,255,0.7)] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#8B3DFF]/60 focus:ring-offset-[#070a14]"
            aria-label="Acción principal"
          >
            <SparkleIcon />
          </button>
        </div>
      </footer>

      {modal && (
        <div
          className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm"
          onClick={() => setModal(null)}
        >
          <div
            className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[92vw] max-w-[860px] h-[70vh] bg-[#0b1020] ring-1 ring-white/10 rounded-2xl p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between text-white/80">
              <div className="font-medium">
                {modal === 'temp' && 'Temperatura (últimas 48h)'}
                {modal === 'hum' && 'Humedad (últimas 48h)'}
                {modal === 'cond' && 'Condición (últimas 48h)'}
              </div>
              <button
                onClick={() => setModal(null)}
                className="px-3 py-1 rounded-md bg-white/10 hover:bg-white/20"
              >
                Cerrar
              </button>
            </div>
            <div className="mt-3 w-full h-[calc(100%-3rem)]" style={{ minHeight: 300 }}>
              {history ? (
                modal === 'cond' ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={history.map((p) => ({ t: p.time.slice(11, 16), code: p.code }))}
                      margin={{ top: 10, right: 20, left: 0, bottom: 20 }}
                    >
                      <CartesianGrid stroke="#ffffff0f" />
                      <XAxis dataKey="t" minTickGap={24} stroke="#9aa4b2" />
                      <YAxis stroke="#9aa4b2" domain={[0, 100]} />
                      <Tooltip
                        contentStyle={{
                          background: '#0b1020',
                          border: '1px solid #22293f',
                          color: '#cbd5e1',
                        }}
                        formatter={(v) => [wmoToText(Number(v)), 'Condición']}
                        labelFormatter={(l) => `Hora: ${l}`}
                      />
                      <Line type="monotone" dataKey="code" stroke="#8B3DFF" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={history.map((p) => ({
                        t: p.time.slice(11, 16),
                        temp: p.temperature,
                        hum: p.humidity,
                      }))}
                      margin={{ top: 10, right: 20, left: 0, bottom: 20 }}
                    >
                      <defs>
                        <linearGradient id="gradT" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#8B3DFF" stopOpacity={0.7} />
                          <stop offset="95%" stopColor="#8B3DFF" stopOpacity={0.05} />
                        </linearGradient>
                        <linearGradient id="gradH" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.7} />
                          <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.05} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="#ffffff0f" />
                      <XAxis dataKey="t" minTickGap={24} stroke="#9aa4b2" />
                      <YAxis stroke="#9aa4b2" />
                      <Tooltip
                        contentStyle={{
                          background: '#0b1020',
                          border: '1px solid #22293f',
                          color: '#cbd5e1',
                        }}
                      />
                      {modal === 'temp' && (
                        <Area type="monotone" dataKey="temp" stroke="#8B3DFF" fill="url(#gradT)" />
                      )}
                      {modal === 'hum' && (
                        <Area type="monotone" dataKey="hum" stroke="#22d3ee" fill="url(#gradH)" />
                      )}
                    </AreaChart>
                  </ResponsiveContainer>
                )
              ) : (
                <div className="w-full h-full grid place-items-center text-white/60">
                  Cargando histórico…
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {addModal && (
        <div
          className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm"
          onClick={() => setAddModal(false)}
        >
          <div
            className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[92vw] max-w-[780px] bg-[#0b1020] ring-1 ring-white/10 rounded-2xl p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between text-white/80">
              <div className="font-medium">Agregar dispositivo</div>
              <button
                onClick={() => setAddModal(false)}
                className="px-3 py-1 rounded-md bg-white/10 hover:bg-white/20"
              >
                Cerrar
              </button>
            </div>
            <div className="mt-3 grid grid-cols-2 md:grid-cols-3 gap-3">
              {addables.map((d) => (
                <div
                  key={`pick-${d.type}`}
                  className="rounded-lg bg-white/5 ring-1 ring-white/10 p-3 text-white/80"
                >
                  <div className="text-sm font-medium truncate">{d.name}</div>
                  <div className="text-xs text-white/60 truncate">Tipo: {d.type}</div>
                  <button
                    className="mt-2 w-full h-8 rounded-md bg-[#8B3DFF] hover:bg-[#7a2cf0] text-white text-sm"
                    onClick={() => {
                      const id = `local-${crypto.randomUUID()}`;
                      const base: Device = {
                        id,
                        name: d.name,
                        type: d.type,
                        protocol: 'sim',
                        is_on: true,
                      } as any;
                      switch (d.type) {
                        case 'light_bulb':
                          Object.assign(base, { brightness: 50, color_temp: 3000 });
                          break;
                        case 'smart_plug':
                          Object.assign(base, { power_w: 5.0, voltage_v: 120.0 });
                          break;
                        case 'contact_sensor':
                          Object.assign(base, { opened: false });
                          break;
                        case 'motion_sensor':
                          Object.assign(base, { motion_detected: false });
                          break;
                        case 'thermostat':
                          Object.assign(base, { current_c: 22.0, hvac_mode: 'auto' });
                          break;
                        case 'chromecast':
                          Object.assign(base, { playback_state: 'idle', app_name: 'Idle' });
                          break;
                        case 'speaker':
                          Object.assign(base, { playback_state: 'stopped', volume: 30 });
                          break;
                        case 'controller':
                          Object.assign(base, { node_count: 0 });
                          break;
                        case 'coordinator':
                          Object.assign(base, { connected_devices: 0 });
                          break;
                        case 'relay':
                          Object.assign(base, { power_w: 0.0 });
                          break;
                      }
                      setDevices((prev) => [...prev, base]);
                      setAddModal(false);
                    }}
                  >
                    Agregar
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {editing && (
        <DeviceEditModal
          device={editing}
          onClose={() => setEditing(null)}
          onChanged={(next) => {
            setDevices((prev) => prev.map((p) => (p.id === next.id ? next : p)));
          }}
        />
      )}
    </div>
  );
}

// --- UI helpers ---
function Card({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <div
      className={`rounded-2xl bg-white/5 ring-1 ring-white/10 p-4 text-white/80 overflow-hidden ${className}`}
    >
      {children}
    </div>
  );
}

// Card that hides all content when very small and shows only an icon
function AutoHideCard({
  icon,
  className = '',
  children,
}: {
  icon: React.ReactNode;
  className?: string;
  children?: React.ReactNode;
}) {
  const ref = React.useRef<HTMLDivElement | null>(null);
  const [tiny, setTiny] = React.useState(false);
  React.useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver((entries) => {
      const cr = entries[0]?.contentRect as DOMRect | undefined;
      const w = cr?.width ?? el.clientWidth;
      const h = cr?.height ?? el.clientHeight;
      // threshold for icon-only view
      setTiny(w < 180 || h < 110);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return (
    <Card className={className}>
      <div ref={ref} className="w-full h-full">
        {tiny || !children ? (
          <div className="w-full h-full grid place-items-center text-white/80">{icon}</div>
        ) : (
          children
        )}
      </div>
    </Card>
  );
}

function Masonry({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`columns-1 sm:columns-2 md:columns-3 xl:columns-4 gap-x-3 ${className}`}>
      {React.Children.map(children, (child) => (
        <div className="mb-3 break-inside-avoid">{child as React.ReactNode}</div>
      ))}
    </div>
  );
}

function IconChip({ color, children }: { color: string; children: React.ReactNode }) {
  return (
    <span
      className="inline-flex items-center justify-center w-6 h-6 rounded-md"
      style={{ backgroundColor: `rgba(${hexToRgb(color)}, 0.18)`, color }}
    >
      {children}
    </span>
  );
}

function ActionCard({
  className = '',
  color = '#8B3DFF',
  label,
  onClick,
}: {
  className?: string;
  color?: string;
  label: string;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-2xl px-4 h-16 text-white text-sm font-medium shadow-[0_8px_24px_-10px_rgba(0,0,0,0.5)] ${className}`}
      style={{ backgroundColor: `${color}CC` }}
    >
      {label}
    </button>
  );
}

function Gauge({ percent, size }: { percent: number; size?: number }) {
  const p = Math.max(0, Math.min(100, percent));
  const boxRef = React.useRef<HTMLDivElement | null>(null);
  const [dim, setDim] = React.useState<number>(size || 112);
  React.useEffect(() => {
    if (size) return; // explicit size wins
    const el = boxRef.current?.parentElement as HTMLElement | null;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect as DOMRect | undefined;
      const w = r?.width ?? el.clientWidth;
      const h = r?.height ?? el.clientHeight;
      const next = Math.max(72, Math.min(Math.min(w, h) * 0.7, 280));
      setDim(next);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [size]);
  const d = size ?? dim;
  return (
    <div ref={boxRef} className="relative" style={{ width: d, height: d }}>
      <div
        className="absolute inset-0 rounded-full"
        style={{
          background: `conic-gradient(#8B3DFF ${p * 3.6}deg, #ffffff1a 0deg)`,
          WebkitMask: 'radial-gradient(circle, transparent 58%, black 60%)',
          mask: 'radial-gradient(circle, transparent 58%, black 60%)',
        }}
      />
      <div
        className="absolute inset-0 grid place-items-center text-white font-semibold"
        style={{ fontSize: Math.max(14, Math.min(d * 0.22, 26)) }}
      >
        {p}%
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-white/5 ring-1 ring-white/10 p-3">
      <div className="text-xs text-white/60">{label}</div>
      <div className="text-white text-xl font-semibold">{value}</div>
    </div>
  );
}

// small util: hex #RRGGBB -> r,g,b
function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  const bigint = parseInt(h, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `${r}, ${g}, ${b}`;
}
import { FaLightbulb, FaPlug, FaDoorOpen, FaChromecast, FaVolumeUp } from 'react-icons/fa';
import { MdMotionPhotosOn, MdLan, MdHub, MdPower } from 'react-icons/md';

function DeviceCard({
  d,
  onOpen,
  onChanged,
}: {
  d: Device;
  onOpen: () => void;
  onChanged?: (next: Device) => void;
}) {
  const btnRef = useRef<HTMLDivElement | null>(null);
  // Shims to avoid referencing parent state inside this child component
  // (real simulation/ticking lives in the parent component)
  // These prevent ReferenceError when this file is bundled.
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const devices: Device[] = [];
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const setDevices = (_updater: any) => {};
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const setSimMode = (_b: boolean) => {};
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const simMode = false;
  // Modes: tiny (icon-only), compact (name+value), full (all)
  const [isTiny, setIsTiny] = useState(false);
  const [isCompact, setIsCompact] = useState(false);
  const [dims, setDims] = useState<{ w: number; h: number }>({ w: 320, h: 80 });
  const [local, setLocal] = useState<Device>({ ...d });
  useEffect(
    () => setLocal({ ...d }),
    [d.id, d.is_on, (d as any).brightness, (d as any).volume, (d as any).target_c]
  );
  useEffect(() => {
    const el = btnRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const target: HTMLElement = (el.parentElement as HTMLElement) || el;
    const update = (rect?: DOMRect) => {
      const w = rect?.width ?? target.clientWidth;
      const h = rect?.height ?? target.clientHeight;
      setDims({ w, h });
      const tiny = w < 150 || h < 58;
      const compact = !tiny && (w < 280 || h < 88);
      setIsTiny(tiny);
      setIsCompact(compact);
    };
    const ro = new ResizeObserver((entries) => {
      const found = entries.find((e) => e.target === target) || entries[0];
      const cr = (found as any)?.contentRect as DOMRect | undefined;
      update(cr);
    });
    ro.observe(target);
    // initial measure
    update();
    return () => ro.disconnect();
  }, []);

  // Local simulated devices fallback (when microservice is down/empty)
  function seedSimDevices(): Device[] {
    return [
      {
        id: 'sim-zigbee',
        name: 'Zigbee Coordinator',
        type: 'coordinator',
        is_on: true,
        connected_devices: 3,
        network_channel: 15,
      } as any,
      {
        id: 'sim-zwave',
        name: 'Z-Wave Controller',
        type: 'controller',
        is_on: true,
        node_count: 5,
        region: 'EU',
      } as any,
      {
        id: 'sim-aq-door',
        name: 'Aqara Door/Window',
        type: 'contact_sensor',
        is_on: true,
        opened: false,
      } as any,
      {
        id: 'sim-aq-motion',
        name: 'Aqara Motion',
        type: 'motion_sensor',
        is_on: true,
        motion_detected: false,
        occupancy_timeout_s: 60,
        lux: 50,
      } as any,
      { id: 'sim-shelly', name: 'Shelly Plus 1', type: 'relay', is_on: true, power_w: 0 } as any,
      {
        id: 'sim-kasa',
        name: 'TP-Link Kasa Plug',
        type: 'smart_plug',
        is_on: true,
        power_w: 5.0,
        voltage_v: 120.0,
      } as any,
      {
        id: 'sim-hue',
        name: 'Philips Hue Bulb',
        type: 'light_bulb',
        is_on: true,
        brightness: 50,
        color_temp: 3000,
      } as any,
      {
        id: 'sim-ecobee',
        name: 'Ecobee Thermostat',
        type: 'thermostat',
        is_on: true,
        hvac_mode: 'auto',
        current_c: 22.0,
        target_c: 22.0,
      } as any,
      {
        id: 'sim-chromecast',
        name: 'Google Chromecast',
        type: 'chromecast',
        is_on: true,
        playback_state: 'idle',
        app_name: 'Idle',
        volume: 50,
      } as any,
      {
        id: 'sim-sonos',
        name: 'Sonos Speaker',
        type: 'speaker',
        is_on: true,
        playback_state: 'stopped',
        volume: 30,
        muted: false,
      } as any,
    ];
  }

  // Detect if we are using local simulated devices
  useEffect(() => {
    setSimMode((devices || []).some((d) => String(d.id).startsWith('sim-')));
  }, [devices]);

  // Frontend-only simulation tick for fallback devices
  useEffect(() => {
    if (!simMode) return;
    const iv = window.setInterval(() => {
      setDevices((prev) => simulateLocalTick(prev));
    }, 2000);
    return () => window.clearInterval(iv);
  }, [simMode]);

  function clamp(n: number, lo: number, hi: number) {
    return Math.max(lo, Math.min(hi, n));
  }

  function simulateLocalTick(prev: Device[]): Device[] {
    const now = new Date().toISOString();
    return prev.map((d) => {
      if (!String(d.id).startsWith('sim-')) return d;
      const t = String(d.type || '');
      const on = !!d.is_on;
      const next: any = { ...d, last_seen: now };
      switch (t) {
        case 'light_bulb':
          if (on)
            next.brightness = clamp(
              Math.round((next.brightness ?? 50) + (Math.random() * 10 - 5)),
              0,
              100
            );
          break;
        case 'smart_plug':
          next.power_w = on ? clamp((next.power_w ?? 0) + Math.random() * 5, 0, 100) : 0;
          next.voltage_v = clamp((next.voltage_v ?? 120) + (Math.random() - 0.5), 100, 240);
          break;
        case 'relay':
          // keep as is; could flip power_w slightly
          next.power_w = on ? clamp((next.power_w ?? 0) + (Math.random() - 0.5) * 2, 0, 50) : 0;
          break;
        case 'contact_sensor':
          if (on && Math.random() < 0.05) next.opened = !next.opened;
          break;
        case 'motion_sensor':
          if (on) {
            if (Math.random() < 0.1) next.motion_detected = true;
            else if (Math.random() < 0.3) next.motion_detected = false;
            next.lux = clamp((next.lux ?? 50) + (Math.random() * 10 - 5), 0, 1000);
          }
          break;
        case 'thermostat':
          if (on && next.hvac_mode !== 'off') {
            const cur = Number(next.current_c ?? 22);
            const tgt = Number(next.target_c ?? 22);
            const delta = tgt - cur;
            next.current_c = Math.round((cur + clamp(delta * 0.1, -0.2, 0.2)) * 10) / 10;
          } else {
            next.current_c =
              Math.round(((next.current_c ?? 22) + (Math.random() * 0.1 - 0.05)) * 10) / 10;
          }
          break;
        case 'chromecast':
          if (on && Math.random() < 0.05) {
            next.playback_state = ['playing', 'paused', 'idle'][Math.floor(Math.random() * 3)];
            next.app_name =
              next.playback_state === 'idle'
                ? 'Idle'
                : ['YouTube', 'Netflix', 'Spotify'][Math.floor(Math.random() * 3)];
          }
          break;
        case 'speaker':
          if (on && Math.random() < 0.05)
            next.playback_state = ['playing', 'paused', 'stopped'][Math.floor(Math.random() * 3)];
          break;
        case 'coordinator':
          if (on)
            next.connected_devices = clamp(
              (next.connected_devices ?? 0) + Math.floor(Math.random() * 5) - 2,
              0,
              5000
            );
          break;
        case 'controller':
          if (on)
            next.node_count = clamp(
              (next.node_count ?? 0) + Math.floor(Math.random() * 3) - 1,
              0,
              5000
            );
          break;
      }
      return next as Device;
    });
  }
  // Local simulated devices fallback (when microservice is down/empty)
  function seedSimDevices(): Device[] {
    return [
      {
        id: 'sim-zigbee',
        name: 'Zigbee Coordinator',
        type: 'coordinator',
        is_on: true,
        connected_devices: 3,
        network_channel: 15,
      },
      {
        id: 'sim-zwave',
        name: 'Z-Wave Controller',
        type: 'controller',
        is_on: true,
        node_count: 5,
        region: 'EU',
      },
      {
        id: 'sim-aq-door',
        name: 'Aqara Door/Window',
        type: 'contact_sensor',
        is_on: true,
        opened: false,
      },
      {
        id: 'sim-aq-motion',
        name: 'Aqara Motion',
        type: 'motion_sensor',
        is_on: true,
        motion_detected: false,
        occupancy_timeout_s: 60,
      },
      { id: 'sim-shelly', name: 'Shelly Plus 1', type: 'relay', is_on: true, power_w: 0 },
      {
        id: 'sim-kasa',
        name: 'TP-Link Kasa Plug',
        type: 'smart_plug',
        is_on: true,
        power_w: 5.0,
        voltage_v: 120.0,
      },
      {
        id: 'sim-hue',
        name: 'Philips Hue Bulb',
        type: 'light_bulb',
        is_on: true,
        brightness: 50,
        color_temp: 3000,
      },
      {
        id: 'sim-ecobee',
        name: 'Ecobee Thermostat',
        type: 'thermostat',
        is_on: true,
        hvac_mode: 'auto',
        current_c: 22.0,
        target_c: 22.0,
      },
      {
        id: 'sim-chromecast',
        name: 'Google Chromecast',
        type: 'chromecast',
        is_on: true,
        playback_state: 'idle',
        app_name: 'Idle',
        volume: 50,
      },
      {
        id: 'sim-sonos',
        name: 'Sonos Speaker',
        type: 'speaker',
        is_on: true,
        playback_state: 'stopped',
        volume: 30,
        muted: false,
      },
    ] as unknown as Device[];
  }
  const icon = (() => {
    switch (d.type) {
      case 'light_bulb':
        return <FaLightbulb className="w-5 h-5" />;
      case 'smart_plug':
        return <FaPlug className="w-5 h-5" />;
      case 'contact_sensor':
        return <FaDoorOpen className="w-5 h-5" />;
      case 'motion_sensor':
        return <MdMotionPhotosOn className="w-5 h-5" />;
      case 'thermostat':
        return <WiThermometer className="w-5 h-5" />;
      case 'chromecast':
        return <FaChromecast className="w-5 h-5" />;
      case 'speaker':
        return <FaVolumeUp className="w-5 h-5" />;
      case 'controller':
        return <MdHub className="w-5 h-5" />;
      case 'coordinator':
        return <MdLan className="w-5 h-5" />;
      case 'relay':
        return <MdPower className="w-5 h-5" />;
      default:
        return <MdPower className="w-5 h-5" />;
    }
  })();

  const TYPE_TONES: Record<string, { fg: string; on: string; off: string }> = {
    light_bulb: { fg: '#F59E0B', on: 'rgba(245, 158, 11, 0.25)', off: 'rgba(148, 163, 184, 0.20)' },
    smart_plug: { fg: '#06B6D4', on: 'rgba(6, 182, 212, 0.25)', off: 'rgba(148, 163, 184, 0.20)' },
    contact_sensor: {
      fg: '#F43F5E',
      on: 'rgba(244, 63, 94, 0.25)',
      off: 'rgba(148, 163, 184, 0.20)',
    },
    motion_sensor: {
      fg: '#84CC16',
      on: 'rgba(132, 204, 22, 0.25)',
      off: 'rgba(148, 163, 184, 0.20)',
    },
    thermostat: { fg: '#FB923C', on: 'rgba(251, 146, 60, 0.25)', off: 'rgba(148, 163, 184, 0.20)' },
    chromecast: { fg: '#D946EF', on: 'rgba(217, 70, 239, 0.25)', off: 'rgba(148, 163, 184, 0.20)' },
    speaker: { fg: '#0EA5E9', on: 'rgba(14, 165, 233, 0.25)', off: 'rgba(148, 163, 184, 0.20)' },
    controller: { fg: '#8B5CF6', on: 'rgba(139, 92, 246, 0.25)', off: 'rgba(148, 163, 184, 0.20)' },
    coordinator: {
      fg: '#6366F1',
      on: 'rgba(99, 102, 241, 0.25)',
      off: 'rgba(148, 163, 184, 0.20)',
    },
    relay: { fg: '#10B981', on: 'rgba(16, 185, 129, 0.25)', off: 'rgba(148, 163, 184, 0.20)' },
  };
  const tone = TYPE_TONES[d.type] || {
    fg: '#8B3DFF',
    on: 'rgba(139, 61, 255, 0.25)',
    off: 'rgba(148, 163, 184, 0.20)',
  };

  const subtitle = (() => {
    switch (d.type) {
      case 'light_bulb':
        return d.is_on ? `Encendido · ${d.brightness ?? 0}%` : 'Apagado';
      case 'smart_plug':
        return d.is_on ? `On · ${d.power_w ?? 0} W` : 'Off';
      case 'contact_sensor':
        return d.opened ? 'Abierto' : 'Cerrado';
      case 'motion_sensor':
        return d.motion_detected ? 'Movimiento' : 'No detectado';
      case 'thermostat':
        return `${d.current_c ?? '—'}°C · modo ${d.hvac_mode ?? '—'}`;
      case 'chromecast':
        return `${d.playback_state ?? ''} ${d.app_name ? '· ' + d.app_name : ''}`.trim();
      case 'speaker':
        return d.playback_state ?? '—';
      case 'controller':
        return `Nodos: ${d.node_count ?? 0}`;
      case 'coordinator':
        return `Dispositivos: ${d.connected_devices ?? 0}`;
      case 'relay':
        return d.is_on ? 'Cerrado' : 'Abierto';
      default:
        return d.protocol ?? '';
    }
  })();

  // Escala basada en ancho y alto de la card para que el contenido
  // crezca/disminuya en conjunto. Partimos de un tamaño base más pequeño.
  const scaleW = dims.w / 320;
  const scaleH = dims.h / 96;
  const scale = Math.min(1.25, Math.max(0.9, Math.min(scaleW, scaleH)));
  const iconSize = Math.round(28 * scale);
  const titleSize = Math.round(13 * scale);
  const valueSize = Math.round(16 * scale);
  const toggleH = Math.round(26 * scale);

  const canToggle =
    d.type === 'light_bulb' ||
    d.type === 'smart_plug' ||
    d.type === 'relay' ||
    d.type === 'speaker';
  const primaryValue = (() => {
    switch (d.type) {
      case 'light_bulb':
        return local.brightness != null ? `${local.brightness}%` : d.is_on ? 'On' : 'Off';
      case 'smart_plug':
        return d.power_w != null ? `${d.power_w} W` : d.is_on ? 'On' : 'Off';
      case 'thermostat':
        return local.target_c != null
          ? `${local.target_c} °C`
          : d.current_c != null
            ? `${d.current_c} °C`
            : '—';
      case 'chromecast':
        return d.playback_state ?? '—';
      case 'speaker':
        return d.playback_state ?? '—';
      case 'controller':
        return `Nodos: ${d.node_count ?? 0}`;
      case 'coordinator':
        return `Disp: ${d.connected_devices ?? 0}`;
      case 'relay':
        return d.is_on ? 'Cerrado' : 'Abierto';
      case 'motion_sensor':
        return d.motion_detected ? 'Movimiento' : 'No detectado';
      case 'contact_sensor':
        return d.opened ? 'Abierto' : 'Cerrado';
      default:
        return d.protocol ?? '';
    }
  })();
  // Derived current state (optimistic) and labels without special chars
  const current = { ...(d as any), ...(local as any) } as any;
  const isOn = !!current.is_on;
  const subtitle2 = (() => {
    switch (d.type) {
      case 'light_bulb':
        return isOn ? `Encendido - ${current.brightness ?? 0}%` : 'Apagado';
      case 'smart_plug':
        return isOn ? `On - ${current.power_w ?? 0} W` : 'Off';
      case 'contact_sensor':
        return current.opened ? 'Abierto' : 'Cerrado';
      case 'motion_sensor':
        return current.motion_detected ? 'Movimiento' : 'No detectado';
      case 'thermostat':
        return `${current.current_c ?? '--'} C - modo ${current.hvac_mode ?? '--'}`;
      case 'chromecast':
        return `${current.playback_state ?? ''} ${current.app_name ? '- ' + current.app_name : ''}`.trim();
      case 'speaker':
        return current.playback_state ?? '--';
      case 'controller':
        return `Nodos: ${current.node_count ?? 0}`;
      case 'coordinator':
        return `Dispositivos: ${current.connected_devices ?? 0}`;
      case 'relay':
        return isOn ? 'Cerrado' : 'Abierto';
      default:
        return current.protocol ?? '';
    }
  })();
  const primaryDisplay = (() => {
    switch (d.type) {
      case 'light_bulb':
        return current.brightness != null ? `${current.brightness}%` : isOn ? 'On' : 'Off';
      case 'smart_plug':
        return current.power_w != null ? `${current.power_w} W` : isOn ? 'On' : 'Off';
      case 'thermostat':
        return current.target_c != null
          ? `${current.target_c} C`
          : current.current_c != null
            ? `${current.current_c} C`
            : '--';
      case 'chromecast':
        return current.playback_state ?? '--';
      case 'speaker':
        return current.playback_state ?? '--';
      case 'controller':
        return `Nodos: ${current.node_count ?? 0}`;
      case 'coordinator':
        return `Disp: ${current.connected_devices ?? 0}`;
      case 'relay':
        return isOn ? 'Cerrado' : 'Abierto';
      case 'motion_sensor':
        return current.motion_detected ? 'Movimiento' : 'No detectado';
      case 'contact_sensor':
        return current.opened ? 'Abierto' : 'Cerrado';
      default:
        return current.protocol ?? '';
    }
  })();

  const toggle = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      // Optimistic UI: flip local is_on immediately
      setLocal((prev) => ({ ...(prev as any), is_on: !(prev as any).is_on }) as any);
      try {
        const res = await powerDevice(d.id, 'toggle');
        try {
          localStorage.setItem(`aura:device:last:${d.id}`, JSON.stringify(res));
        } catch {}
        onChanged?.(res);
      } catch {
        // Backend caído: actualizar visualmente también en la lista
        onChanged?.({ ...d, is_on: !(d.is_on ?? false) } as Device);
      }
    } catch {}
  };

  // Debounced inline save for sliders on the card
  const inlineTimer = useRef<number | null>(null);
  const [justSaved, setJustSaved] = useState(false);
  const scheduleInlineSave = (payload: Record<string, any>) => {
    try {
      window.clearTimeout(inlineTimer.current as any);
    } catch {}
    inlineTimer.current = window.setTimeout(async () => {
      try {
        // optimistic merge for a snappier feel
        setLocal((prev) => ({ ...(prev as any), ...(payload as any) }) as any);
        const res = await updateDevice(d.id, payload);
        onChanged?.(res);
        try {
          localStorage.setItem(`aura:device:last:${res.id}`, JSON.stringify(res));
        } catch {}
        setJustSaved(true);
        window.setTimeout(() => setJustSaved(false), 800);
      } catch {
        // Si falla, reflejar visual en lista localmente
        onChanged?.({ ...d, ...(payload as any) } as Device);
        setJustSaved(true);
        window.setTimeout(() => setJustSaved(false), 800);
      }
    }, 450);
  };

  return (
    <div
      role="button"
      onPointerDown={(e) => e.stopPropagation()}
      ref={btnRef}
      onClick={onOpen}
      className="w-full h-full overflow-hidden rounded-xl bg-white/5 ring-1 ring-white/10 p-3 text-white/80 flex items-center gap-3 text-left hover:bg-white/10 transition"
    >
      <div
        className="rounded-lg grid place-items-center flex-shrink-0"
        style={{
          width: iconSize,
          height: iconSize,
          background: isOn ? tone.on : tone.off,
          color: tone.fg,
        }}
      >
        {icon}
      </div>
      {!isTiny && (
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-3">
            <div className="font-medium truncate" style={{ fontSize: titleSize }}>
              {d.name || 'Device'}
            </div>
            {canToggle && (
              <button
                onClick={toggle}
                className={`rounded-full px-3 text-white/90 ${isOn ? 'bg-[#8B3DFF] hover:bg-[#7a2cf0]' : 'bg-white/10 hover:bg-white/20'} transition`}
                style={{ height: toggleH }}
              >
                {isOn ? 'On' : 'Off'}
              </button>
            )}
          </div>
          {!isCompact && (
            <div
              className="text-white/60 truncate"
              style={{ fontSize: Math.max(11, Math.round(11 * scale)) }}
            >
              {subtitle2}
            </div>
          )}
          <div className="mt-1 font-semibold" style={{ fontSize: valueSize }}>
            {primaryDisplay}
          </div>
          {/* Inline controls by type */}
          {d.type === 'light_bulb' && !isCompact && (
            <div className="mt-2 flex items-center gap-2">
              <input
                type="range"
                min={0}
                max={100}
                value={Number(local.brightness ?? d.brightness ?? 0)}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setLocal({ ...local, brightness: v } as any);
                  scheduleInlineSave({ brightness: v });
                }}
                className="w-40"
              />
              {justSaved && <span className="text-emerald-400 text-xs">Guardado ✓</span>}
            </div>
          )}
          {d.type === 'speaker' && !isCompact && (
            <div className="mt-2 flex items-center gap-2">
              <input
                type="range"
                min={0}
                max={100}
                value={Number(local.volume ?? d.volume ?? 0)}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setLocal({ ...local, volume: v } as any);
                  scheduleInlineSave({ volume: v });
                }}
                className="w-40"
              />
              {justSaved && <span className="text-emerald-400 text-xs">Guardado ✓</span>}
            </div>
          )}
          {d.type === 'thermostat' && !isCompact && (
            <div className="mt-2 flex items-center gap-2">
              <input
                type="range"
                min={10}
                max={30}
                step={0.5}
                value={Number(local.target_c ?? d.target_c ?? d.current_c ?? 22)}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setLocal({ ...local, target_c: v } as any);
                  scheduleInlineSave({ target_c: v });
                }}
                className="w-40"
              />
              {justSaved && <span className="text-emerald-400 text-xs">Guardado ✓</span>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function DeviceEditModal({
  device,
  onClose,
  onChanged,
}: {
  device: Device;
  onClose: () => void;
  onChanged: (d: Device) => void;
}) {
  const [working, setWorking] = useState(false);
  const [d, setD] = useState<Device>({ ...device });

  const save = async (payload: Record<string, any>) => {
    setWorking(true);
    try {
      const res = await updateDevice(d.id, payload);
      setD(res);
      onChanged(res);
      try {
        localStorage.setItem(`aura:device:last:${res.id}`, JSON.stringify(res));
      } catch {}
    } catch (e) {
      console.error(e);
    } finally {
      setWorking(false);
    }
  };

  const power = async (action: 'on' | 'off' | 'toggle') => {
    setWorking(true);
    try {
      const res = await powerDevice(d.id, action);
      setD(res);
      onChanged(res);
      try {
        localStorage.setItem(`aura:device:last:${res.id}`, JSON.stringify(res));
      } catch {}
    } catch (e) {
      console.error(e);
    } finally {
      setWorking(false);
    }
  };

  const Field = ({ label, children }: { label: string; children: React.ReactNode }) => (
    <div>
      <div className="text-xs text-white/60 mb-1">{label}</div>
      {children}
    </div>
  );

  const Actions = ({ children }: { children?: React.ReactNode }) => (
    <div className="mt-3 flex items-center gap-2">
      <button
        onClick={() => power('toggle')}
        className="px-3 h-9 rounded-md bg-white/10 hover:bg-white/20"
      >
        {d.is_on ? 'Apagar' : 'Encender'}
      </button>
      {children}
      <button
        onClick={onClose}
        className="ml-auto px-3 h-9 rounded-md bg-white/10 hover:bg-white/20"
      >
        Cerrar
      </button>
    </div>
  );

  const NumberInput = ({
    value,
    min,
    max,
    step = 1,
    onChange,
  }: {
    value: number;
    min: number;
    max: number;
    step?: number;
    onChange: (n: number) => void;
  }) => (
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-full h-9 bg-white/5 border border-white/10 rounded-md px-2 text-white/80 focus:border-[#8B3DFF] outline-none"
    />
  );

  const Range = ({
    value,
    min,
    max,
    step = 1,
    onChange,
  }: {
    value: number;
    min: number;
    max: number;
    step?: number;
    onChange: (n: number) => void;
  }) => (
    <input
      type="range"
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-full"
    />
  );

  // Debounced autosave helper for modal controls
  const saveTimer = useRef<number | null>(null);
  const scheduleSave = (payload: Record<string, any>) => {
    try {
      if (saveTimer.current) window.clearTimeout(saveTimer.current);
    } catch {}
    saveTimer.current = window.setTimeout(() => {
      save(payload);
    }, 400);
  };

  const body = () => {
    switch (d.type) {
      case 'light_bulb':
        return (
          <>
            {/* Visual preview estilo lámpara */}
            <div className="w-full grid place-items-center">
              {(() => {
                const temp = Number(d.color_temp ?? 3000);
                const t = Math.max(2000, Math.min(6500, temp));
                const ratio = (t - 2000) / (6500 - 2000);
                const warm = { r: 255, g: 210, b: 121 };
                const cool = { r: 183, g: 223, b: 255 };
                const mix = {
                  r: Math.round(warm.r + (cool.r - warm.r) * ratio),
                  g: Math.round(warm.g + (cool.g - warm.g) * ratio),
                  b: Math.round(warm.b + (cool.b - warm.b) * ratio),
                };
                const rgb = `rgb(${mix.r}, ${mix.g}, ${mix.b})`;
                const bright = Math.max(0.05, Math.min(1, Number(d.brightness ?? 0) / 100));
                return (
                  <div
                    className="relative w-28 h-48 rounded-3xl overflow-hidden ring-1 ring-white/10"
                    style={{
                      background: `linear-gradient(180deg, ${rgb} ${Math.round(bright * 100)}%, rgba(255,255,255,0.04))`,
                    }}
                  >
                    <div className="absolute bottom-2 left-1/2 -translate-x-1/2 w-12 h-2 rounded-full bg-white/60 opacity-70" />
                  </div>
                );
              })()}
            </div>
            <Field label="Brillo">
              <Range
                value={Number(d.brightness ?? 0)}
                min={0}
                max={100}
                onChange={(v) => {
                  setD({ ...d, brightness: v });
                  scheduleSave({ brightness: v });
                }}
              />
            </Field>
            <Field label="Temperatura de color (K)">
              <NumberInput
                value={Number(d.color_temp ?? 3000)}
                min={2000}
                max={6500}
                step={50}
                onChange={(v) => {
                  setD({ ...d, color_temp: v });
                  scheduleSave({ color_temp: v });
                }}
              />
            </Field>
            <Actions />
          </>
        );
      case 'smart_plug':
        return (
          <>
            <div className="text-white/70 text-sm">Potencia actual: {d.power_w ?? 0} W</div>
            <Actions />
          </>
        );
      case 'contact_sensor':
        return (
          <>
            <Field label="Estado">
              <button
                onClick={async () => {
                  const next = !d.opened;
                  setD({ ...d, opened: next });
                  await save({ opened: next });
                }}
                className="px-3 h-9 rounded-md bg-white/10 hover:bg-white/20"
              >
                {d.opened ? 'Cerrar' : 'Abrir'}
              </button>
            </Field>
            <Actions />
          </>
        );
      case 'motion_sensor':
        return (
          <>
            <Field label="Timeout ocupación (s)">
              <NumberInput
                value={Number(d.occupancy_timeout_s ?? 60)}
                min={5}
                max={600}
                onChange={(v) => {
                  setD({ ...d, occupancy_timeout_s: v });
                  scheduleSave({ occupancy_timeout_s: v });
                }}
              />
            </Field>
            <Actions />
          </>
        );
      case 'thermostat':
        return (
          <>
            <Field label="Objetivo (°C)">
              <Range
                value={Number(d.target_c ?? 22)}
                min={10}
                max={30}
                step={0.5}
                onChange={(v) => {
                  setD({ ...d, target_c: v });
                  scheduleSave({ target_c: v });
                }}
              />
            </Field>
            <Field label="Modo HVAC">
              <select
                value={String(d.hvac_mode || 'auto')}
                onChange={(e) => {
                  const m = e.target.value;
                  setD({ ...d, hvac_mode: m });
                  scheduleSave({ hvac_mode: m });
                }}
                className="w-full h-9 bg-white/5 border border-white/10 rounded-md px-2 text-white/80"
              >
                {['off', 'heat', 'cool', 'auto'].map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </Field>
            <Actions />
          </>
        );
      case 'chromecast':
        return (
          <>
            <Field label="Volumen">
              <Range
                value={Number(d.volume ?? 50)}
                min={0}
                max={100}
                step={1}
                onChange={(v) => {
                  setD({ ...d, volume: v });
                  scheduleSave({ volume: v });
                }}
              />
            </Field>
            <Field label="Aplicación">
              <input
                value={String(d.app_name || '')}
                onChange={(e) => setD({ ...d, app_name: e.target.value })}
                className="w-full h-9 bg-white/5 border border-white/10 rounded-md px-2 text-white/80"
              />
            </Field>
            <Actions />
          </>
        );
      case 'speaker':
        return (
          <>
            <Field label="Volumen">
              <Range
                value={Number(d.volume ?? 30)}
                min={0}
                max={100}
                step={1}
                onChange={(v) => {
                  setD({ ...d, volume: v });
                  scheduleSave({ volume: v });
                }}
              />
            </Field>
            <Field label="Mute">
              <button
                onClick={async () => {
                  const next = !d.muted;
                  setD({ ...d, muted: next });
                  await save({ muted: next });
                }}
                className="px-3 h-9 rounded-md bg-white/10 hover:bg-white/20"
              >
                {d.muted ? 'Quitar mute' : 'Poner mute'}
              </button>
            </Field>
            <Actions />
          </>
        );
      case 'controller':
        return (
          <>
            <Field label="Región">
              <select
                value={String(d.region || 'EU')}
                onChange={(e) => setD({ ...d, region: e.target.value })}
                className="w-full h-9 bg-white/5 border border-white/10 rounded-md px-2 text-white/80"
              >
                {['EU', 'US', 'ANZ'].map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </Field>
            <Actions />
          </>
        );
      case 'coordinator':
        return (
          <>
            <Field label="Canal">
              <NumberInput
                value={Number(d.network_channel ?? 15)}
                min={11}
                max={26}
                onChange={(v) => setD({ ...d, network_channel: v })}
              />
            </Field>
            <Actions />
          </>
        );
      case 'relay':
        return (
          <>
            <div className="text-white/70 text-sm">Energía: {d.power_w ?? 0} W</div>
            <Actions />
          </>
        );
      default:
        return (
          <>
            <div className="text-white/70 text-sm">Sin configuración específica.</div>
            <Actions />
          </>
        );
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm" onClick={onClose}>
      <div
        className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[92vw] max-w-[640px] bg-[#0b1020] ring-1 ring-white/10 rounded-2xl p-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between text-white/80">
          <div className="font-medium">Configurar: {d.name}</div>
          <button onClick={onClose} className="px-3 py-1 rounded-md bg-white/10 hover:bg-white/20">
            Cerrar
          </button>
        </div>
        <div className="mt-3 grid gap-3 text-white/80">{body()}</div>
      </div>
    </div>
  );
}

function SparkleIcon() {
  return (
    <img
      src="/images/img_1.svg"
      alt="Arrow icon"
      className="w-[10px] sm:w-[14px] md:w-[17px] lg:w-[20px] h-[10px] sm:h-[14px] md:h-[17px] lg:h-[20px] absolute left-[2px] sm:left-[3px] md:left-[3px] lg:left-spacing-xs"
    />
  );
}
// Persist and hydrate per-device last state locally
const deviceLastKey = (id: string) => `aura:device:last:${id}`;
const mergeLocal = (d: Device): Device => {
  try {
    const raw = localStorage.getItem(deviceLastKey(String(d.id)));
    if (!raw) return d;
    const over = JSON.parse(raw);
    return { ...d, ...over } as Device;
  } catch {
    return d;
  }
};
const withOverrides = (arr: Device[]) => (Array.isArray(arr) ? arr.map(mergeLocal) : arr);
const persistLocal = (d: Device) => {
  try {
    localStorage.setItem(deviceLastKey(String(d.id)), JSON.stringify(d));
  } catch {}
};
