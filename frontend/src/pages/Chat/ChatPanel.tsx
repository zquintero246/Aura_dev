// ChatPanel.tsx (wired to GPT)
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { chat as chatApi, ChatMessage } from '../../lib/chat';

import { ensureChatToken } from '../../lib/auth';
import { startConversation as startChatConversation } from '../../lib/chatApi';
import { listMessages as loadChatHistory, ChatMessageItem } from '../../lib/chatApi';
import { updateConversationTitle as persistConversationTitle } from '../../lib/conversations';
import MarkdownLite from '../../components/MarkdownLite';
// (inline svg icons, no external deps)

const escapeRegExp = (value: string) =>
  value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const renderHighlightedText = (text: string, query: string) => {
  if (!query) return text;
  const searchLower = query.trim().toLowerCase();
  if (!searchLower) return text;
  const pattern = new RegExp(`(${escapeRegExp(query)})`, 'gi');
  return text.split(pattern).map((segment, index) => {
    if (segment.toLowerCase() === searchLower) {
      return (
        <span key={index} className="text-[#8B3DFF]">
          {segment}
        </span>
      );
    }
    return <span key={index}>{segment}</span>;
  });
};

type Msg = {
  id: string;
  author: 'user' | 'assistant';
  name: string;
  avatar?: string;
  text?: string;
  ts: number;
  typing?: boolean;
  reasoning?: string;
  thinkMs?: number;
};

export default function ChatPanel({
  conversationId,
  conversationTitle,
  userName = 'Tú',
  userAvatar = '/images/avatar_demo.jpg',
  onTitleChange,
  searchQuery,
  onClearSearch,
}: {
  conversationId: string;
  conversationTitle?: string;
  userName?: string;
  userAvatar?: string;
  onTitleChange?: (id: string, title: string) => void;
  searchQuery?: string;
  onClearSearch?: () => void;
}) {
  const isTempConversation =
    typeof conversationId === 'string' && conversationId.startsWith('tmp-');

  const ensureConversationId = async (): Promise<string> => {
    if (!isTempConversation) return conversationId;
    try {
      // try to create with current title
      const titleToUse = title?.trim() || 'Nueva conversación';
      const conv = await createConversation(titleToUse);
      // notify app to replace temp id
      try {
        window.dispatchEvent(
          new CustomEvent('aura:conversation:realized', {
            detail: { tempId: conversationId, newId: conv.id, title: conv.title },
          })
        );
      } catch {}
      return conv.id;
    } catch {
      return conversationId; // fallback, still send without persistence
    }
  };
  const [title, setTitle] = useState('Agregar titulo');
  const [editingTitle, setEditingTitle] = useState(false);
  const defaultModel = (import.meta as any)?.env?.VITE_DEFAULT_MODEL as string | undefined;
  const [model, setModel] = useState<string>(defaultModel || 'gpt-4o-mini');
  const [modelOpen, setModelOpen] = useState(false);
  const [modelLocked, setModelLocked] = useState(false);

  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);
  const createdRef = useRef(false);

  const scrollerRef = useRef<HTMLDivElement>(null);
  const messageRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [highlightedMessageId, setHighlightedMessageId] = useState<string | null>(null);

  const scrollToBottom = (behavior: ScrollBehavior = 'smooth') => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior });
  };
  useEffect(() => {
    scrollToBottom(messages.length <= 2 ? 'auto' : 'smooth');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages.length, conversationId]);

  useEffect(() => {
    if (!searchQuery) {
      setHighlightedMessageId(null);
    }
  }, [searchQuery]);

  const scrollToMessage = (id: string) => {
    const el = messageRefs.current[id];
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      setHighlightedMessageId(id);
      window.setTimeout(() => setHighlightedMessageId(null), 2800);
    }
    onClearSearch?.();
  };

  const filteredMessages = useMemo(() => {
    const query = (searchQuery || '').trim().toLowerCase();
    if (!query) return [];
    return messages
      .filter((m) => !!m.text)
      .map((m) => ({ ...m, text: m.text || '' }))
      .filter((m) => m.text.toLowerCase().includes(query))
      .slice(0, 8);
  }, [messages, searchQuery]);

  // Reset model lock when switching conversation
  useEffect(() => {
    setModelLocked(false);
    setModelOpen(false);
  }, [conversationId]);

  useEffect(() => {
    const resolved = (conversationTitle && conversationTitle.trim()) || 'Agregar titulo';
    setTitle(resolved);
    setEditingTitle(false);
  }, [conversationId, conversationTitle]);

  // Immediately create a real conversation in Mongo when user clicks "Nueva conversación"
  useEffect(() => {
    (async () => {
      if (!isTempConversation) return;
      if (createdRef.current) return;
      createdRef.current = true;
      try {
        const tok = await ensureChatToken();
        if (tok) console.debug('[ChatPanel] Using PAT for chat_service');
      } catch {}
      try {
        const titleToUse = title?.trim() || 'Nueva conversación';
        console.debug('[ChatPanel] Creating conversation on /chat/start â€¦', {
          tempId: conversationId,
          title: titleToUse,
        });
        const conv = await startChatConversation(titleToUse);
        console.debug('[ChatPanel] Conversation created', conv);
        if (conv?.id) {
          try {
            window.dispatchEvent(
              new CustomEvent('aura:conversation:realized', {
                detail: {
                  tempId: conversationId,
                  newId: conv.id,
                  title: conv.title || titleToUse,
                  userId: (conv as any)?.user_id,
                },
              })
            );
          } catch {}
        }
      } catch (e) {
        createdRef.current = false; // allow retry on next mount/switch if needed
        console.error('[ChatPanel] Failed to create conversation on start:', e);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTempConversation, conversationId]);

  // Hydrate conversation from Mongo history when opening a real conversation id
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!conversationId || isTempConversation) {
        setMessages([]);
        return;
      }
      try {
        const items = await loadChatHistory(conversationId);
        if (cancelled) return;
        const mapped: Msg[] = items.map((m: ChatMessageItem) => ({
          id: m.id,
          author: m.role === 'user' ? 'user' : 'assistant',
          name: m.role === 'user' ? userName : 'AURA',
          avatar: m.role === 'user' ? userAvatar : undefined,
          text: m.content,
          ts: m.created_at ? Date.parse(m.created_at) : Date.now(),
        }));
        setMessages(mapped);
      } catch {
        if (!cancelled) setMessages([]);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId, userName, userAvatar]);

  const normalizedTitle = useMemo(() => (title.trim() ? title.trim() : 'Sin titulo'), [title]);
  const commitTitle = async () => {
    setEditingTitle(false);
    const finalTitle = normalizedTitle;
    if (isTempConversation || !conversationId) {
      setTitle(finalTitle);
      onTitleChange?.(conversationId, finalTitle);
      return;
    }
    try {
      await persistConversationTitle(conversationId, finalTitle);
      setTitle(finalTitle);
      onTitleChange?.(conversationId, finalTitle);
    } catch (err) {
      console.error('Failed to update conversation title', err);
      try {
        window.alert('No se pudo actualizar el título. Intenta nuevamente.');
      } catch {}
      setTitle(conversationTitle || finalTitle);
    }
  };

  const handleSend = async () => {
    if (busy) return;
    const text = input.trim();
    if (!text) return;

    if (!modelLocked) {
      setModelLocked(true);
      setModelOpen(false);
    }

    setBusy(true);
    setMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        author: 'user',
        name: userName,
        avatar: userAvatar,
        text,
        ts: Date.now(),
      },
    ]);
    // Autoâ€‘tÃ­tulo: si aÃºn es placeholder, proponemos uno desde el primer mensaje
    try {
      const isPlaceholder = !title || /Agregar\s*t[Ã­iï¿½]tulo|Sin\s*t[Ã­iï¿½]tulo/i.test(title);
      if (isPlaceholder) {
        const plain = text.replace(/\s+/g, ' ').trim();
        if (plain) {
          const sentence = plain.split(/(?<=[.!?])\s+/)[0] || plain;
          const words = sentence.split(' ');
          let candidate = sentence;
          if (candidate.length > 60 || words.length > 10) candidate = words.slice(0, 10).join(' ');
          const auto = candidate.length > 60 ? candidate.slice(0, 57).trim() + 'â€¦' : candidate;
          if (auto) {
            setTitle(auto);
            onTitleChange?.(conversationId, auto);
          }
        }
      }
    } catch {}
    setInput('');
    scrollToBottom();

    const typingId = crypto.randomUUID();
    const startTs = Date.now();
    setMessages((prev) => [
      ...prev,
      { id: typingId, author: 'assistant', name: 'AURA', ts: Date.now(), typing: true },
    ]);

    try {
      const history: ChatMessage[] = messages
        .filter((m) => !!m.text)
        .map((m) => ({ role: m.author === 'user' ? 'user' : 'assistant', content: m.text || '' }));
      const payload: ChatMessage[] = [...history, { role: 'user', content: text }];
      const realConvId = await ensureConversationId();
      const resp = await chatApi(payload, model, realConvId);
      const content = resp.content || '';
      const reasoning = extractReasoning(resp.raw);
      const thinkMs = Date.now() - startTs;
      setMessages((prev) =>
        prev
          .filter((m) => m.id !== typingId)
          .concat({
            id: crypto.randomUUID(),
            author: 'assistant',
            name: 'AURA',
            ts: Date.now(),
            text: content,
            reasoning,
            thinkMs,
          })
      );
      setBusy(false);
    } catch (e: any) {
      const code = e?.code as string | undefined;
      let msg: string;
      if (code === 'rate_limited') {
        msg = 'Servidor saturado, intenta de nuevo.';
      } else if (code === 'no_content') {
        msg = 'El modelo no respondió. Intenta nuevamente en unos segundos.';
      } else if (code === 'network_error' || code === 'timeout') {
        msg = 'Problema de red o tiempo de espera. Reintenta.';
      } else {
        msg = e?.message || 'No pude responder ahora. Intenta nuevamente.';
      }
      setMessages((prev) =>
        prev
          .filter((m) => m.id !== typingId)
          .concat({
            id: crypto.randomUUID(),
            author: 'assistant',
            name: 'AURA',
            ts: Date.now(),
            text: msg,
          })
      );
      setBusy(false);
    }
  };

  return (
    <div className="h-full min-h-0 flex flex-col mx-auto max-w-[1400px] relative">
      {/* header */}
      <div className="sticky top-0 z-10 pt-3 pb-3 bg-[#070a14]/80 backdrop-blur rounded-t-[18px] border-b border-white/10">
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-8 bg-gradient-to-b from-transparent to-[#070a14]/90"></div>
        <div className="relative flex items-center justify-between gap-3">
          {/* Modelo (solo logo) */}
          <div className="relative">
            <button
              onClick={() => !modelLocked && setModelOpen((s) => !s)}
              className="flex items-center gap-2 rounded-xl bg-white/5 ring-1 ring-white/10 hover:bg-white/10 transition px-2 py-2"
              title="Modelo"
              aria-label={model}
              aria-disabled={modelLocked}
            >
              <img
                src={
                  model.startsWith('deepseek')
                    ? '/images/DeepSeek.svg'
                    : model.startsWith('gemini') || model.includes('google/gemini')
                      ? '/images/Gemini.svg'
                      : '/images/logo.svg'
                }
                alt="Modelo"
                className="w-5 h-5 rounded-lg object-contain"
              />
              {!modelLocked && (
                <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 text-white/70">
                  <path d="M5.5 7.5l4.5 4.5 4.5-4.5H5.5z" />
                </svg>
              )}
            </button>
            {modelOpen && !modelLocked && (
              <div className="absolute mt-2 left-0 w-64 rounded-xl bg-[#0f1320] border border-white/10 p-1 shadow-lg z-10">
                <button
                  onClick={() => {
                    setModel('gpt-4o-mini');
                    setModelOpen(false);
                  }}
                  className="w-full text-left px-3 py-2 rounded-lg hover:bg-white/5 text-sm flex items-center gap-2"
                >
                  <img src="/images/logo.svg" alt="AuraV1" className="w-5 h-5" />
                  Aura V1
                </button>
                <div className="mx-2 my-1 h-px bg-white/10" />
                <button
                  onClick={() => {
                    setModel('google/gemini-2.0-flash-exp:free'); // via OpenRouter
                    setModelOpen(false);
                  }}
                  className="w-full text-left px-3 py-2 rounded-lg hover:bg-white/5 text-sm flex items-center gap-2"
                >
                  <img src="/images/Gemini.svg" alt="Gemini via OpenRouter" className="w-5 h-5" />
                  Gemini 2.0 Flash (exp, OpenRouter Free)
                </button>
                <button
                  onClick={() => {
                    setModel('deepseek/deepseek-r1-0528-qwen3-8b:free'); // via OpenRouter
                    setModelOpen(false);
                  }}
                  className="mt-1 w-full text-left px-3 py-2 rounded-lg hover:bg-white/5 text-sm flex items-center gap-2"
                >
                  <img src="/images/DeepSeek.svg" alt="DeepSeek" className="w-5 h-5" />
                  DeepSeek R1 (Qwen3 8B Free)
                </button>
              </div>
            )}
          </div>

          {/* TÃ­tulo */}
          <div className="flex-1 text-center">
            {editingTitle ? (
              <input
                autoFocus
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                onBlur={() => void commitTitle()}
                onKeyDown={(e) => e.key === 'Enter' && void commitTitle()}
                className="mx-auto w-full max-w-[420px] text-center bg-transparent outline-none border-b border-white/10 focus:border-[#8B3DFF] text-white text-[20px] font-semibold"
              />
            ) : (
              <button
                className="text-white text-[20px] font-semibold hover:opacity-90"
                onClick={() => setEditingTitle(true)}
                title="Editar titulo"
              >
                <span className="mr-2">{normalizedTitle}</span>
                <svg
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  className="w-4 h-4 inline-block opacity-80"
                >
                  <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04a1.003 1.003 0 000-1.42l-2.34-2.34a1.003 1.003 0 00-1.42 0l-1.83 1.83 3.75 3.75 1.84-1.82z" />
                </svg>
              </button>
            )}
          </div>
          <div className="shrink-0">
            <button
              onClick={() => {
                try {
                  window.dispatchEvent(new CustomEvent('aura:chat:settings'));
                } catch {}
              }}
              className="grid place-items-center w-9 h-9 rounded-lg bg-white/5 ring-1 ring-white/10 hover:bg-white/10 transition"
              title="ConfiguraciÃ³n del chat"
              aria-label="ConfiguraciÃ³n del chat"
            >
              <svg viewBox="0 0 512 512" fill="currentColor" className="w-5 h-5 text-white/80">
                <g clipPath="url(#clip0_64_6)">
                  <path d="M496.851 212.213L448.045 200.012C443.978 186.041 438.389 172.558 431.379 159.807L457.269 116.656C459.564 112.831 460.514 108.349 459.969 103.922C459.424 99.4954 457.415 95.3778 454.261 92.224L419.776 57.739C416.622 54.5851 412.504 52.5759 408.078 52.0309C403.651 51.4858 399.169 52.4362 395.344 54.731L352.193 80.621C339.442 73.6106 325.959 68.0217 311.988 63.955L299.787 15.15C298.705 10.8227 296.208 6.98119 292.693 4.23613C289.177 1.49106 284.844 6.27129e-06 280.384 1.23161e-09L231.615 1.23161e-09C227.155 -4.94798e-05 222.822 1.49086 219.307 4.23573C215.791 6.9806 213.294 10.8219 212.212 15.149L200.011 63.954C186.04 68.0207 172.557 73.6096 159.806 80.62L116.655 54.73C112.830 52.4351 108.348 51.4846 103.921 52.0296C99.4944 52.5746 95.3768 54.5839 92.223 57.738L57.738 92.223C54.5842 95.377 52.5751 99.4946 52.0301 103.921C51.4851 108.348 52.4354 112.83 54.73 116.655L80.62 159.806C73.6096 172.557 68.0207 186.04 63.954 200.011L15.148 212.212C10.8211 213.294 6.98001 215.791 4.23534 219.307C1.49066 222.823 -0.000105235 227.155 5.57174e-09 231.615L5.57174e-09 280.384C-4.94755e-05 284.844 1.49086 289.177 4.23573 292.692C6.9806 296.208 10.8219 298.705 15.149 299.787L63.955 311.988C68.0217 325.959 73.6106 339.442 80.621 352.193L54.731 395.344C52.4361 399.169 51.4856 403.651 52.0306 408.078C52.5756 412.505 54.5849 416.622 57.739 419.776L92.224 454.261C95.3779 457.415 99.4955 459.424 103.922 459.969C108.349 460.514 112.831 459.564 116.656 457.269L159.807 431.379C172.558 438.389 186.041 443.978 200.012 448.045L212.213 496.85C213.295 501.177 215.792 505.018 219.308 507.763C222.823 510.508 227.156 511.999 231.616 511.999H280.385C284.845 511.999 289.178 510.508 292.693 507.763C296.209 505.018 298.706 501.177 299.788 496.85L311.989 448.045C325.960 443.978 339.443 438.389 352.194 431.379L395.345 457.269C399.170 459.564 403.652 460.514 408.079 459.969C412.506 459.424 416.623 457.415 419.777 454.261L454.262 419.776C457.416 416.622 459.425 412.504 459.970 408.078C460.515 403.651 459.565 399.169 457.270 395.344L431.380 352.193C438.390 339.442 443.979 325.959 448.046 311.988L496.852 299.787C501.179 298.705 505.020 296.208 507.765 292.692C510.510 289.177 512.001 284.844 512.001 280.384V231.615C512.001 227.155 510.510 222.823 507.765 219.307C505.020 215.792 501.178 213.295 496.851 212.213ZM256 336C211.888 336 176 300.112 176 256C176 211.888 211.888 176 256 176C300.112 176 336 211.888 336 256C336 300.112 300.112 336 256 336Z" />
                </g>
                <defs>
                  <clipPath id="clip0_64_6">
                    <rect width="512" height="512" fill="white" />
                  </clipPath>
                </defs>
              </svg>
            </button>
          </div>
        </div>
      </div>

      {filteredMessages.length > 0 && (
        <div className="absolute top-[92px] left-[50%] z-40 w-[min(560px,calc(100%-48px))] -translate-x-1/2">
          <div className="rounded-[28px] border border-white/15 bg-[#0b1320]/90 px-4 py-4 shadow-[0_25px_80px_rgba(1,5,18,0.95)] backdrop-blur">
            <div className="text-[10px] uppercase tracking-[0.35em] text-white/50">Buscar mensajes</div>
            <p className="mt-1 text-sm text-white/70">
              {conversationTitle || 'Conversación actual'} · {filteredMessages.length} resultados
            </p>
            <div className="mt-3 flex max-h-[260px] flex-col gap-2 overflow-y-auto pr-2">
              {filteredMessages.map((msg) => (
                <button
                  key={msg.id}
                  onClick={() => scrollToMessage(msg.id)}
                  className="rounded-2xl border border-white/5 bg-white/5 px-3 py-2 text-left text-sm text-white/80 transition hover:bg-white/10 hover:border-white/10"
                >
                  <div className="text-[12px] text-white/60">
                    {new Date(msg.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                  <p className="mt-1 text-sm leading-relaxed text-white">
                    {renderHighlightedText(msg.text, searchQuery || '')}
                  </p>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* mensajes */}
      <div ref={scrollerRef} className="scroll-slim mt-6 flex-1 min-h-0 overflow-y-auto pr-1">
        <div className="flex flex-col gap-5 pb-[5px]">
          {messages.map((m) => (
            <Bubble
              key={m.id}
              msg={m}
              highlight={highlightedMessageId === m.id}
              forwardedRef={(el) => {
                messageRefs.current[m.id] = el;
              }}
            />
          ))}
        </div>
      </div>

      {/* composer */}
      <div className="sticky bottom-0 left-0 right-0 z-10 relative">
        <div className="pointer-events-none absolute inset-x-0 top-0 h-10 bg-gradient-to-b from-[#070a14] to-transparent" />
        <div className="pointer-events-none h-3 bg-gradient-to-t from-[#070a14] to-transparent" />
        <div className="bg-[#070a14]/90 backdrop-blur border-t border-white/10">
          <div className="px-3 sm:px-4 py-3 pb-[calc(10px+env(safe-area-inset-bottom,0px))]">
            <div className="rounded-[20px] bg-white/5 ring-1 ring-white/10 p-3">
              <div className="flex items-center gap-3">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !busy && handleSend()}
                  placeholder="Escribe tu mensaje..."
                  className="flex-1 bg-transparent outline-none text-white/90 placeholder:text-white/40 px-2"
                />
                <button
                  onClick={handleSend}
                  disabled={busy}
                  className={`px-4 h-9 rounded-lg bg-[#7B2FE3] hover:bg-[#6c29c9] active:scale-95 transition ${busy ? 'opacity-60 cursor-not-allowed' : ''}`}
                >
                  Enviar
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------- Componentes de mensaje ---------- */

function Bubble({
  msg,
  forwardedRef,
  highlight,
}: {
  msg: Msg;
  forwardedRef?: (el: HTMLDivElement | null) => void;
  highlight?: boolean;
}) {
  const isUser = msg.author === 'user';
  const [copied, setCopied] = React.useState(false);
  const [showReasoning, setShowReasoning] = React.useState(false);

  const handleCopy = async () => {
    if (!msg.text) return;
    try {
      await navigator.clipboard.writeText(msg.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {}
  };

  return (
    <div
      ref={forwardedRef}
      className={`group/message flex items-end gap-3 ${isUser ? 'flex-row-reverse' : ''} ${
        highlight ? 'ring-2 ring-[#8B3DFF]/70 rounded-3xl' : ''
      }`}
    >
      <div className="shrink-0">
        {isUser ? (
          <img
            src={msg.avatar || '/images/avatar_demo.jpg'}
            alt={msg.name}
            className="w-8 h-8 rounded-full object-cover ring-1 ring-white/10"
            onError={(e) => {
              const t = e.currentTarget;
              if (t.src.indexOf('/images/avatar_demo.jpg') === -1)
                t.src = '/images/avatar_demo.jpg';
            }}
          />
        ) : (
          <div className="w-8 h-8 rounded-full grid place-items-center bg-white/5 ring-1 ring-white/10">
            <img src="/images/logo.svg" alt="AURA" className="w-4 h-4" />
          </div>
        )}
      </div>
      <div className={`max-w-[78%] ${isUser ? 'items-end text-right' : ''}`}>
        <div className="text-[11px] text-white/50 mb-1">{msg.name}</div>

        {msg.typing ? (
          <div className="space-y-2">
            <ThinkingStripe />
            <div
              className={`inline-flex items-center gap-1 px-3 py-2 rounded-2xl ${isUser ? 'bg-[#7B2FE3] text-white rounded-br-md' : 'bg-white/5 ring-1 ring-white/10 text-white rounded-bl-md'}`}
            >
              <Dot delay="0ms" />
              <Dot delay="150ms" />
              <Dot delay="300ms" />
            </div>
          </div>
        ) : (
          <div
            className={`inline-block px-4 py-2 text-[15px] leading-relaxed rounded-2xl ${isUser ? 'bg-[#7B2FE3] text-white rounded-br-md' : 'bg-white/5 ring-1 ring-white/10 text-white rounded-bl-md'}`}
          >
            {isUser ? (
              msg.text
            ) : (
              <div className="prose-invert max-w-none">
                {msg.reasoning && (
                  <div className="mb-3 rounded-md overflow-hidden ring-1 ring-white/10">
                    <button
                      onClick={() => setShowReasoning((s) => !s)}
                      className="w-full text-left text-[12px] px-3 py-2 bg-white/5 hover:bg-white/10 flex items-center justify-between"
                    >
                      <span>
                        Razonamiento
                        {typeof msg.thinkMs === 'number'
                          ? ` (${(msg.thinkMs / 1000).toFixed(1)}s)`
                          : ''}
                      </span>
                      <span className="opacity-70">{showReasoning ? 'Ocultar' : 'Ver'}</span>
                    </button>
                    {showReasoning && (
                      <div className="px-3 py-2 bg-transparent">
                        <MarkdownLite text={msg.reasoning} />
                      </div>
                    )}
                  </div>
                )}
                <MarkdownLite text={msg.text || ''} />
              </div>
            )}
          </div>
        )}

        <div
          className={`mt-1 flex items-center gap-2 ${isUser ? 'justify-end' : 'justify-start'} opacity-0 group-hover/message:opacity-100 group-focus-within/message:opacity-100 transition-opacity duration-200`}
        >
          <span className="text-[11px] text-white/35">
            {new Date(msg.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>
          {msg.text && (
            <button
              onClick={handleCopy}
              className="p-1.5 rounded-md hover:bg-white/5 ring-1 ring-transparent hover:ring-white/10 active:scale-95 transition"
              aria-label="Copiar mensaje"
              title={copied ? 'Â¡Copiado!' : 'Copiar'}
            >
              <CopyIcon className={`w-4 h-4 ${copied ? 'text-[#8B3DFF]' : 'text-white/60'}`} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function Dot({ delay }: { delay: string }) {
  return (
    <span
      className="inline-block w-1.5 h-1.5 rounded-full bg-white/70 animate-bounce"
      style={{ animationDelay: delay }}
    />
  );
}

function ThinkingStripe() {
  return (
    <div className="w-[220px] h-2 rounded-full overflow-hidden bg-white/5">
      <div className="h-full w-1/3 bg-gradient-to-r from-white/0 via-white/60 to-white/0 animate-[slide_1.2s_linear_infinite]" />
      <style>{`@keyframes slide{0%{transform:translateX(-100%)}100%{transform:translateX(300%)}}`}</style>
    </div>
  );
}

function extractReasoning(raw: any): string | undefined {
  try {
    if (!raw) return undefined;
    const choice = raw?.choices?.[0];
    // OpenAI o3: message.reasoning.content (string or array of objects with type=text)
    const r = choice?.message?.reasoning;
    if (typeof r === 'string') return r;
    if (r?.content) {
      if (typeof r.content === 'string') return r.content;
      if (Array.isArray(r.content)) {
        const textParts = r.content
          .map((p: any) => (typeof p === 'string' ? p : p?.text || p?.content || ''))
          .filter(Boolean);
        if (textParts.length) return textParts.join('\n');
      }
    }
    // Some providers put it in message.metadata.reasoning
    const metaR = choice?.message?.metadata?.reasoning;
    if (typeof metaR === 'string' && metaR.trim()) return metaR;
    // DeepSeek/OpenRouter variants sometimes include 'reasoning' at top level
    if (typeof choice?.reasoning === 'string') return choice.reasoning;
  } catch {}
  return undefined;
}

function CopyIcon(props: any) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M8 7a2 2 0 012-2h7a2 2 0 012 2v9a2 2 0 01-2 2h-7a2 2 0 01-2-2V7zm-3 3h1v8a3 3 0 003 3h8v1H9a4 4 0 01-4-4v-8z" />
    </svg>
  );
}
