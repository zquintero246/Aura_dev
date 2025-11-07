import React, { useEffect, useMemo, useState } from 'react';
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
import { FiPlus } from 'react-icons/fi';
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
  const [home, setHome] = useState<{
    city?: string;
    country?: string;
    latitude?: number;
    longitude?: number;
  } | null>(null);
  const [history, setHistory] = useState<HistoryPoint[] | null>(null);
  const [modal, setModal] = useState<null | 'temp' | 'hum' | 'cond'>(null);
  const [devices, setDevices] = useState<Device[]>([]);
  const [editing, setEditing] = useState<Device | null>(null);
  const [addModal, setAddModal] = useState(false);
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
          required = localStorage.getItem('home:registered') !== '1';
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
            setDevices(devs);
          } catch {}
        } else {
          // Fallback: use localStorage coords if present
          try {
            const raw = localStorage.getItem('home:coords');
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
                  setDevices(devs);
                } catch {}
              }
            }
          } catch {}
        }
      } catch {
        setNeedsRegistration(localStorage.getItem('home:registered') !== '1');
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
        if (!cancelled && Array.isArray(devs)) setDevices(devs);
      } catch {}
    };
    // Initial and interval
    poll();
    const id = window.setInterval(poll, 5000);
    return () => { cancelled = true; window.clearInterval(id); };
  }, []);

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
          <div className="font-medium tracking-wide">Mi casa</div>

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
            <button
              onClick={async () => {
                try {
                  const devs = await listDevices();
                  if (devs?.length) setDevices(devs);
                } catch {}
                setAddModal(true);
              }}
              className="px-3 h-9 rounded-lg bg-white/10 hover:bg-white/20 text-white/80 text-sm inline-flex items-center gap-2"
              title="Agregar dispositivo"
            >
              <FiPlus className="w-4 h-4" />
              Agregar dispositivo
            </button>
          </div>
        </div>
      </header>
      <section className="flex-1 min-h-0 px-2 md:px-4 py-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-3 content-start">
            {devices.map((d) => (
              <DeviceCard key={d.id} d={d} onOpen={() => setEditing(d)} />
            ))}
          {devices.length === 0 && (
            <div className="rounded-xl bg-white/5 ring-1 ring-white/10 p-4 text-white/60">
              Sin dispositivos
            </div>
          )}
        </div>
      </section>
      <footer className="px-2 md:px-4 pb-3">
        <div className="relative mx-auto max-w-[820px]">
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
            <div className="mt-3 w-full h-[calc(100%-3rem)]">
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
import { FaLightbulb, FaPlug, FaDoorOpen, FaChromecast, FaVolumeUp } from 'react-icons/fa';
import { MdMotionPhotosOn, MdLan, MdHub, MdPower } from 'react-icons/md';

function DeviceCard({ d, onOpen }: { d: Device; onOpen: () => void }) {
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

  return (
    <button onClick={onOpen} className="rounded-xl bg-white/5 ring-1 ring-white/10 p-3 text-white/80 flex items-center gap-3 text-left hover:bg-white/10 transition">
      <div
        className={`w-9 h-9 rounded-lg grid place-items-center ${d.is_on ? 'bg-[#8B3DFF]/30 text-[#caa6ff]' : 'bg-white/10 text-white/70'}`}
      >
        {icon}
      </div>
      <div className="min-w-0">
        <div className="text-sm font-medium truncate">{d.name || 'Device'}</div>
        <div className="text-xs text-white/60 truncate">{subtitle}</div>
      </div>
    </button>
  );
}

function DeviceEditModal({ device, onClose, onChanged }: { device: Device; onClose: () => void; onChanged: (d: Device) => void }) {
  const [working, setWorking] = useState(false);
  const [d, setD] = useState<Device>({ ...device });

  const save = async (payload: Record<string, any>) => {
    setWorking(true);
    try {
      const res = await updateDevice(d.id, payload);
      setD(res);
      onChanged(res);
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
      <button onClick={() => power('toggle')} className="px-3 h-9 rounded-md bg-white/10 hover:bg-white/20">{d.is_on ? 'Apagar' : 'Encender'}</button>
      {children}
      <button onClick={onClose} className="ml-auto px-3 h-9 rounded-md bg-white/10 hover:bg-white/20">Cerrar</button>
    </div>
  );

  const NumberInput = ({ value, min, max, step = 1, onChange }: { value: number; min: number; max: number; step?: number; onChange: (n: number) => void }) => (
    <input type="number" value={value} min={min} max={max} step={step} onChange={(e) => onChange(Number(e.target.value))} className="w-full h-9 bg-white/5 border border-white/10 rounded-md px-2 text-white/80 focus:border-[#8B3DFF] outline-none" />
  );

  const Range = ({ value, min, max, step = 1, onChange }: { value: number; min: number; max: number; step?: number; onChange: (n: number) => void }) => (
    <input type="range" value={value} min={min} max={max} step={step} onChange={(e) => onChange(Number(e.target.value))} className="w-full" />
  );

  const body = () => {
    switch (d.type) {
      case 'light_bulb':
        return (
          <>
            <Field label="Brillo">
              <Range value={Number(d.brightness ?? 0)} min={0} max={100} onChange={(v) => setD({ ...d, brightness: v })} />
            </Field>
            <Field label="Temperatura de color (K)">
              <NumberInput value={Number(d.color_temp ?? 3000)} min={2000} max={6500} step={50} onChange={(v) => setD({ ...d, color_temp: v })} />
            </Field>
            <Actions>
              <button disabled={working} onClick={() => save({ brightness: d.brightness, color_temp: d.color_temp })} className="px-3 h-9 rounded-md bg-[#8B3DFF] hover:bg-[#7a2cf0]">Guardar</button>
            </Actions>
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
              <button onClick={() => setD({ ...d, opened: !d.opened })} className="px-3 h-9 rounded-md bg-white/10 hover:bg-white/20">{d.opened ? 'Cerrar' : 'Abrir'}</button>
            </Field>
            <Actions>
              <button disabled={working} onClick={() => save({ opened: d.opened })} className="px-3 h-9 rounded-md bg-[#8B3DFF] hover:bg-[#7a2cf0]">Guardar</button>
            </Actions>
          </>
        );
      case 'motion_sensor':
        return (
          <>
            <Field label="Timeout ocupación (s)">
              <NumberInput value={Number(d.occupancy_timeout_s ?? 60)} min={5} max={600} onChange={(v) => setD({ ...d, occupancy_timeout_s: v })} />
            </Field>
            <Actions>
              <button disabled={working} onClick={() => save({ occupancy_timeout_s: d.occupancy_timeout_s })} className="px-3 h-9 rounded-md bg-[#8B3DFF] hover:bg-[#7a2cf0]">Guardar</button>
            </Actions>
          </>
        );
      case 'thermostat':
        return (
          <>
            <Field label="Objetivo (°C)">
              <Range value={Number(d.target_c ?? 22)} min={10} max={30} step={0.5} onChange={(v) => setD({ ...d, target_c: v })} />
            </Field>
            <Field label="Modo HVAC">
              <select value={String(d.hvac_mode || 'auto')} onChange={(e) => setD({ ...d, hvac_mode: e.target.value })} className="w-full h-9 bg-white/5 border border-white/10 rounded-md px-2 text-white/80">
                {['off', 'heat', 'cool', 'auto'].map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </Field>
            <Actions>
              <button disabled={working} onClick={() => save({ target_c: d.target_c, hvac_mode: d.hvac_mode })} className="px-3 h-9 rounded-md bg-[#8B3DFF] hover:bg-[#7a2cf0]">Guardar</button>
            </Actions>
          </>
        );
      case 'chromecast':
        return (
          <>
            <Field label="Volumen">
              <Range value={Number(d.volume ?? 50)} min={0} max={100} step={1} onChange={(v) => setD({ ...d, volume: v })} />
            </Field>
            <Field label="Aplicación">
              <input value={String(d.app_name || '')} onChange={(e) => setD({ ...d, app_name: e.target.value })} className="w-full h-9 bg-white/5 border border-white/10 rounded-md px-2 text-white/80" />
            </Field>
            <Actions>
              <button disabled={working} onClick={() => save({ volume: d.volume, app_name: d.app_name })} className="px-3 h-9 rounded-md bg-[#8B3DFF] hover:bg-[#7a2cf0]">Guardar</button>
            </Actions>
          </>
        );
      case 'speaker':
        return (
          <>
            <Field label="Volumen">
              <Range value={Number(d.volume ?? 30)} min={0} max={100} step={1} onChange={(v) => setD({ ...d, volume: v })} />
            </Field>
            <Field label="Mute">
              <button onClick={() => setD({ ...d, muted: !d.muted })} className="px-3 h-9 rounded-md bg-white/10 hover:bg-white/20">{d.muted ? 'Quitar mute' : 'Poner mute'}</button>
            </Field>
            <Actions>
              <button disabled={working} onClick={() => save({ volume: d.volume, muted: d.muted })} className="px-3 h-9 rounded-md bg-[#8B3DFF] hover:bg-[#7a2cf0]">Guardar</button>
            </Actions>
          </>
        );
      case 'controller':
        return (
          <>
            <Field label="Región">
              <select value={String(d.region || 'EU')} onChange={(e) => setD({ ...d, region: e.target.value })} className="w-full h-9 bg-white/5 border border-white/10 rounded-md px-2 text-white/80">
                {['EU', 'US', 'ANZ'].map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </Field>
            <Actions>
              <button disabled={working} onClick={() => save({ region: d.region })} className="px-3 h-9 rounded-md bg-[#8B3DFF] hover:bg-[#7a2cf0]">Guardar</button>
            </Actions>
          </>
        );
      case 'coordinator':
        return (
          <>
            <Field label="Canal">
              <NumberInput value={Number(d.network_channel ?? 15)} min={11} max={26} onChange={(v) => setD({ ...d, network_channel: v })} />
            </Field>
            <Actions>
              <button disabled={working} onClick={() => save({ channel: d.network_channel })} className="px-3 h-9 rounded-md bg-[#8B3DFF] hover:bg-[#7a2cf0]">Guardar</button>
            </Actions>
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
      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[92vw] max-w-[640px] bg-[#0b1020] ring-1 ring-white/10 rounded-2xl p-4" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between text-white/80">
          <div className="font-medium">Configurar: {d.name}</div>
          <button onClick={onClose} className="px-3 py-1 rounded-md bg-white/10 hover:bg-white/20">Cerrar</button>
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
