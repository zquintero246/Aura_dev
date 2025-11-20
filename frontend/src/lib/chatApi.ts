import axios from 'axios';
import { ensureChatToken } from './auth';

export type ConversationTheme = {
  bubble_color?: string;
  background_color?: string;
};
export type Conversation = {
  id: string;
  title?: string;
  user_id?: string;
  created_at?: string;
  theme?: ConversationTheme;
};
export type ChatMessageItem = { id: string; role: 'user' | 'assistant' | 'system'; content: string; created_at?: string };

const BASE: string = (import.meta as any)?.env?.VITE_CHAT_API_BASE || 'http://127.0.0.1:5080';
export const CHAT_BASE = BASE;

const chat = axios.create({ baseURL: BASE });

function getStoredChatToken(): string | null {
  try {
    return localStorage.getItem('aura:pat');
  } catch {
    return null;
  }
}

chat.interceptors.request.use(async (config: any) => {
  // Ensure we have a token; mint if missing
  let t = getStoredChatToken();
  if (!t) {
    try { t = await ensureChatToken() as any; } catch {}
  }
  if (t) {
    config.headers = { ...(config.headers || {}), Authorization: `Bearer ${t}` } as any;
  }
  return config;
});

// On 401 from chat service, try to refresh the token once and retry
chat.interceptors.response.use(
  (res) => res,
  async (error) => {
    const status = error?.response?.status;
    const cfg = error?.config || {};
    if (status === 401 && !cfg._retry) {
      try {
        cfg._retry = true;
        // Clear old token and mint a new one
        clearChatToken();
        const t = await ensureChatToken();
        if (t) {
          cfg.headers = { ...(cfg.headers || {}), Authorization: `Bearer ${t}` };
          return chat.request(cfg);
        }
      } catch {}
    }
    throw error;
  }
);

export function setChatToken(token: string) {
  try { localStorage.setItem('aura:pat', token); } catch {}
}

export function clearChatToken() {
  try { localStorage.removeItem('aura:pat'); } catch {}
}

export async function startConversation(
  title?: string,
  opts?: { participants?: string[] },
): Promise<Conversation> {
  console.debug('[chatApi] startConversation ->', { title, participants: opts?.participants });
  const payload: Record<string, any> = { title };
  if (opts?.participants && opts.participants.length) {
    payload.participants = opts.participants;
  }
  const res = await chat.post('/chat/start', payload);
  const data = res.data as Conversation;
  console.debug('[chatApi] startConversation <-', data);
  return data;
}

export async function listConversations(): Promise<Conversation[]> {
  const res = await chat.get('/chat/history');
  return (res.data?.conversations ?? []) as Conversation[];
}

export async function updateConversation(
  conversationId: string,
  updates: {
    title?: string;
    bubbleColor?: string;
    backgroundColor?: string;
  },
): Promise<Conversation> {
  const payload: Record<string, string> = {};
  if (updates.title !== undefined) payload.title = updates.title;
  if (updates.bubbleColor !== undefined) payload.bubble_color = updates.bubbleColor;
  if (updates.backgroundColor !== undefined) payload.background_color = updates.backgroundColor;
  const res = await chat.put(`/chat/conversations/${encodeURIComponent(conversationId)}`, payload);
  return (res.data?.conversation ?? res.data) as Conversation;
}

export async function sendMessage(
  conversationId: string,
  content: string,
  role: 'user' | 'assistant' | 'system' = 'user',
  options?: { clientMessageId?: string },
): Promise<{ ok?: boolean; message?: { id: string; created_at?: string } }> {
  console.debug('[chatApi] sendMessage ->', { conversationId, role, preview: content?.slice(0, 48), clientMessageId: options?.clientMessageId });
  const payload: Record<string, any> = { conversation_id: conversationId, content, role };
  if (options?.clientMessageId) {
    payload.client_message_id = options.clientMessageId;
  }
  const res = await chat.post('/chat/message', payload);
  console.debug('[chatApi] sendMessage <-', res.data);
  return res.data as any;
}

export async function listMessages(conversationId: string): Promise<ChatMessageItem[]> {
  const res = await chat.get(`/chat/conversations/${encodeURIComponent(conversationId)}/messages`);
  return (res.data?.messages ?? []) as ChatMessageItem[];
}

export async function deleteConversation(conversationId: string): Promise<{ ok: boolean }> {
  const res = await chat.delete(`/chat/conversations/${encodeURIComponent(conversationId)}`);
  // Some servers may return empty body. Normalize response.
  const ok = res.status >= 200 && res.status < 300;
  return (res.data && typeof res.data.ok === 'boolean') ? res.data : { ok };
}

export { getStoredChatToken };
