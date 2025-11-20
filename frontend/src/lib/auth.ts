import api from './api';

export type User = { id: number; name: string; email: string; email_verified_at: string | null; avatar_url?: string | null };

export async function register(name: string, email: string, password: string, password_confirmation: string) {
  const res = await api.post('/api/auth/register', { name, email, password, password_confirmation });
  return res.data as { message: string; user: User };
}

export async function login(email: string, password: string) {
  const res = await api.post('/api/auth/login', { email, password });
  return res.data as { message: string; user: User };
}

export async function logout() {
  const res = await api.post('/api/auth/logout');
  return res.data as { message: string };
}

export async function me() {
  const res = await api.get('/api/auth/me');
  return res.data as { user: User };
}

export async function resendVerification() {
  const res = await api.post('/api/auth/email/resend');
  return res.data as { message: string };
}

// Mint a Sanctum Personal Access Token for the chat microservice
export async function mintChatToken() {
  const res = await api.post('/api/auth/token');
  return res.data as { token: string };
}

export async function ensureChatToken(currentUserId?: string | number): Promise<string | null> {
  try {
    const storedUid = localStorage.getItem('aura:uid') || '';
    const expectedUid = currentUserId != null ? String(currentUserId) : storedUid;
    const existing = localStorage.getItem('aura:pat');
    // If token exists but bound to a different user, discard it
    if (existing && existing.includes('|') && storedUid && expectedUid && storedUid === expectedUid) {
      return existing;
    }
  } catch {}

  try {
    const minted = await mintChatToken();
    const token = (minted as any)?.token as string | undefined;
    if (token) {
      try {
        localStorage.setItem('aura:pat', token);
        if (currentUserId != null) localStorage.setItem('aura:uid', String(currentUserId));
      } catch {}
      return token;
    }
  } catch {}
  return null;
}
