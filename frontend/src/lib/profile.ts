import api from './api';
import type { User } from './auth';

export async function updateProfile(formData: FormData): Promise<User | null> {
  const { data } = await api.post('/api/profile', formData);
  return data?.user ?? null;
}
