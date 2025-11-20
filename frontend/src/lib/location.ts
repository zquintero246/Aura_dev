import api from './api';

export type LocationPayload = {
  country: string;
  city: string;
  latitude: number;
  longitude: number;
};

export async function saveLocation(payload: LocationPayload) {
  const { data } = await api.post('/api/location', payload);
  return data;
}

export async function getMyLocation() {
  const { data } = await api.get('/api/location/me');
  return data?.data ?? null;
}

