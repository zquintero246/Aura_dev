import api from './api';
import { sendMessage as persistMessage } from './chatApi';

export type ChatMessage = { role: 'system' | 'user' | 'assistant'; content: string };

export type ChatSuccess = { content: string; raw?: any };
export type ApiError = {
  message: string;
  code?: 'rate_limited' | 'no_content' | 'upstream_error' | 'network_error' | 'timeout' | 'unexpected';
  retryable?: boolean;
  retryAfter?: number;
  error?: any;
};

export async function chat(
  messages: ChatMessage[],
  model?: string,
  conversationId?: string,
  options?: { clientMessageId?: string },
): Promise<ChatSuccess> {
  try {
    // Persist the latest user message in Mongo (best-effort)
    try {
      const lastUser = [...messages].reverse().find((m) => m.role === 'user');
      if (conversationId && lastUser?.content) {
        await persistMessage(conversationId, lastUser.content, 'user', options?.clientMessageId);
        // Optimistic: propose an auto-title from the first user message
        try {
          const firstUser = messages.find((m) => m.role === 'user');
          const t = buildAutoTitle(firstUser?.content || lastUser.content || '');
          if (t && typeof window !== 'undefined') {
            try { window.dispatchEvent(new CustomEvent('aura:conversation:titled', { detail: { id: conversationId, title: t } })); } catch {}
          }
        } catch {}
      }
    } catch {}

    const res = await api.post('/api/chat', { messages, model, conversationId });
    const data = res.data as ChatSuccess;

    // Persist assistant reply as well (best-effort)
    try {
      if (conversationId && data?.content) {
        await persistMessage(conversationId, data.content, 'assistant');
      }
    } catch {}

    return data;
  } catch (err: any) {
    const status = err?.response?.status;
    const data = err?.response?.data as ApiError | undefined;
    const apiError: ApiError = {
      message:
        data?.message ||
        (status === 429 ? 'Servidor saturado, intenta de nuevo.' : 'No pude responder ahora. Intenta nuevamente.'),
      code: data?.code,
      retryable: data?.retryable ?? [408, 409, 425, 429, 500, 502, 503, 504].includes(status),
      retryAfter: data?.retryAfter,
      error: data?.error || err?.message,
    };
    // Re-throw so caller can render a friendly bubble
    throw apiError;
  }
}

// Build a concise title from the first messages (max 6 words)
function buildAutoTitle(src: string): string {
  try {
    let t = (src || '').replace(/\s+/g, ' ').trim();
    if (!t) return '';
    t = t.replace(/[\u00BF\u00A1\s]+/u, '').replace(/[\s.?!,;:]+$/u, '');
    const rules: Array<[RegExp, string]> = [
      [/^dame\s+ideas\s+para\s+/i, 'Ideas para '],
      [/^ideas\s+para\s+/i, 'Ideas para '],
      [/^como\s+hacer\s+/i, 'Hacer '],
      [/^como\s+crear\s+/i, 'Crear '],
      [/^como\s+hago\s+/i, 'Crear '],
      [/^como\s+puedo\s+/i, ''],
      [/^quiero\s+/i, ''],
      [/^quisiera\s+/i, ''],
      [/^necesito\s+/i, ''],
      [/^por\s+favor\s+/i, ''],
    ];
    for (const [re, rep] of rules) { if (re.test(t)) { t = t.replace(re, rep); break; } }
    let first = (t.split(/(?<=[.!?])\s+/u)[0] || t).replace(/[\s.?!,;:]+$/u, '');
    const words = first.split(/\s+/u);
    if (words.length > 6) first = words.slice(0, 6).join(' ');
    if (first) first = first.charAt(0).toUpperCase() + first.slice(1);
    const generic = ['nueva conversacion', 'nueva conversación', 'nuevo chat', 'conversacion con ia', 'conversación con ia', 'charla general', 'chat general'];
    if (generic.includes(first.toLowerCase())) return '';
    return first.trim();
  } catch { return ''; }
}



