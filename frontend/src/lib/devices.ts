const BASE = (import.meta as any)?.env?.VITE_MICROSERVICES_URL || 'http://127.0.0.1:5050';

export type Device = {
  id: string;
  name: string;
  type: string;
  protocol?: string;
  is_on?: boolean;
  [key: string]: any;
};

export async function listDevices(): Promise<Device[]> {
  const url = `${BASE}/api/devices`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const j = await res.json();
  return (j?.devices as Device[]) || [];
}

export async function powerDevice(id: string, action: 'on' | 'off' | 'toggle'): Promise<Device> {
  const url = `${BASE}/api/devices/${encodeURIComponent(id)}/power`;
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action }) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return (await res.json()) as Device;
}

export async function updateDevice(id: string, payload: Record<string, any>): Promise<Device> {
  const url = `${BASE}/api/devices/${encodeURIComponent(id)}/update`;
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return (await res.json()) as Device;
}
