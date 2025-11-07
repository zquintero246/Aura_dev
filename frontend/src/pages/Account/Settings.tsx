import React, { useEffect, useState } from 'react';
import { saveLocation, getMyLocation } from '../../lib/location';

export default function Settings() {
  const [country, setCountry] = useState('');
  const [city, setCity] = useState('');
  const [latitude, setLatitude] = useState<string>('');
  const [longitude, setLongitude] = useState<string>('');
  const [status, setStatus] = useState<string>('');

  useEffect(() => {
    (async () => {
      try {
        const existing = await getMyLocation();
        if (existing) {
          setCountry(existing.country || '');
          setCity(existing.city || '');
          setLatitude(String(existing.latitude ?? ''));
          setLongitude(String(existing.longitude ?? ''));
        }
      } catch (_) {
        // ignore
      }
    })();
  }, []);

  const onUseGeolocation = () => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition((pos) => {
      setLatitude(String(pos.coords.latitude.toFixed(6)));
      setLongitude(String(pos.coords.longitude.toFixed(6)));
    });
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('');
    try {
      const lat = Number(latitude);
      const lon = Number(longitude);
      if (Number.isNaN(lat) || Number.isNaN(lon)) {
        setStatus('Lat/Lon inválidos');
        return;
      }
      await saveLocation({ country, city, latitude: lat, longitude: lon });
      setStatus('Ubicación guardada');
    } catch (err: any) {
      const msg = err?.response?.data?.message || 'Error al guardar';
      setStatus(msg);
    }
  };

  return (
    <div className="p-6 text-white max-w-3xl mx-auto">
      <h1 className="text-xl font-semibold">Configuración</h1>
      <p className="text-sm text-white/70 mt-2">Preferencias de la cuenta.</p>

      <div className="mt-8 border border-white/10 rounded-lg p-5 bg-white/5">
        <h2 className="text-lg font-medium">Home Assistant · Ubicación</h2>
        <p className="text-sm text-white/60 mt-1">Registra País, Ciudad y Coordenadas.</p>

        <form onSubmit={onSubmit} className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="md:col-span-1">
            <label className="block text-sm text-white/70 mb-1">País</label>
            <input
              value={country}
              onChange={(e) => setCountry(e.target.value)}
              placeholder="Colombia"
              className="w-full bg-white/5 text-white rounded-md py-2.5 px-3 border border-white/10 focus:border-[#8B3DFF] outline-none"
            />
          </div>
          <div className="md:col-span-1">
            <label className="block text-sm text-white/70 mb-1">Ciudad</label>
            <input
              value={city}
              onChange={(e) => setCity(e.target.value)}
              placeholder="Bogotá"
              className="w-full bg-white/5 text-white rounded-md py-2.5 px-3 border border-white/10 focus:border-[#8B3DFF] outline-none"
            />
          </div>
          <div>
            <label className="block text-sm text-white/70 mb-1">Latitud</label>
            <input
              value={latitude}
              onChange={(e) => setLatitude(e.target.value)}
              placeholder="4.711000"
              className="w-full bg-white/5 text-white rounded-md py-2.5 px-3 border border-white/10 focus:border-[#8B3DFF] outline-none"
            />
          </div>
          <div>
            <label className="block text-sm text-white/70 mb-1">Longitud</label>
            <input
              value={longitude}
              onChange={(e) => setLongitude(e.target.value)}
              placeholder="-74.072100"
              className="w-full bg-white/5 text-white rounded-md py-2.5 px-3 border border-white/10 focus:border-[#8B3DFF] outline-none"
            />
          </div>

          <div className="md:col-span-2 flex items-center gap-3">
            <button
              type="button"
              onClick={onUseGeolocation}
              className="bg-white/10 hover:bg-white/20 px-3 py-2 rounded-md"
            >
              Usar mi ubicación actual
            </button>
            <button
              type="submit"
              className="bg-[#8B3DFF] hover:bg-[#7a2cf0] px-4 py-2 rounded-md"
            >
              Guardar ubicación
            </button>
            {status && (
              <span className="text-sm text-white/70">{status}</span>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}
