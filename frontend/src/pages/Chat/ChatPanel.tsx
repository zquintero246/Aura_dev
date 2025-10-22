// ChatPanel.tsx
import React, { useEffect, useMemo, useRef, useState } from 'react';

type Msg = {
  id: string;
  author: 'user' | 'assistant';
  name: string;
  avatar?: string;
  text?: string;
  ts: number;
  typing?: boolean;
};

export default function ChatPanel({
  conversationId,
  userName = 'Tú',
  userAvatar = '/images/avatar_demo.jpg',
  onTitleChange, // <— NUEVO
}: {
  conversationId: string;
  userName?: string;
  userAvatar?: string;
  onTitleChange?: (id: string, title: string) => void; // <— NUEVO
}) {
  const [title, setTitle] = useState('Agregar título…');
  const [editingTitle, setEditingTitle] = useState(false);
  const [model, setModel] = useState<'Aura v1' | 'Aura v2'>('Aura v1');
  const [modelOpen, setModelOpen] = useState(false);

  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Msg[]>(() => [
    {
      id: 'm1',
      author: 'assistant',
      name: 'Aura',
      ts: Date.now() - 1000 * 60 * 5,
      text: '¡Hola! Soy Aura. ¿En qué puedo ayudarte hoy?',
    },
  ]);

  const scrollerRef = useRef<HTMLDivElement>(null);

  // ------- AUTOSCROLL -------
  const scrollToBottom = (behavior: ScrollBehavior = 'smooth') => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior });
  };
  useEffect(() => {
    scrollToBottom(messages.length <= 2 ? 'auto' : 'smooth');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages.length, conversationId]);

  // ------- GUARDAR TÍTULO (propaga al panel de la izquierda) -------
  const normalizedTitle = useMemo(() => (title.trim() ? title.trim() : 'Sin título'), [title]);

  // dispara cuando el usuario termina de editar (blur o Enter)
  const commitTitle = () => {
    setEditingTitle(false);
    onTitleChange?.(conversationId, normalizedTitle);
  };

  // ------- ENVIAR -------
  const handleSend = async () => {
    const text = input.trim();
    if (!text) return;

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
    setInput('');
    scrollToBottom();

    const typingId = crypto.randomUUID();
    setMessages((prev) => [
      ...prev,
      { id: typingId, author: 'assistant', name: 'Aura', ts: Date.now(), typing: true },
    ]);
    await new Promise((r) => setTimeout(r, 1000));
    setMessages((prev) =>
      prev
        .filter((m) => m.id !== typingId)
        .concat({
          id: crypto.randomUUID(),
          author: 'assistant',
          name: 'Aura',
          ts: Date.now(),
          text: '¡Listo! ¿En qué más te apoyo?',
        })
    );
  };

  return (
    // todo el panel principal
    <div className="h-full min-h-0 flex flex-col mx-auto max-w-[1400px]">
      {/* header */}
      <div className="sticky top-0 z-10 pt-3 pb-3 bg-[#070a14]/80 backdrop-blur rounded-t-[18px] border-b border-white/10">
        <div className="flex items-center justify-between gap-3">
          <div className="w-10" />
          <div className="flex-1 text-center">
            {editingTitle ? (
              <input
                autoFocus
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                onBlur={commitTitle}
                onKeyDown={(e) => e.key === 'Enter' && commitTitle()}
                className="mx-auto w-full max-w-[420px] text-center bg-transparent outline-none border-b border-white/10 focus:border-[#8B3DFF] text-white text-[20px] font-semibold"
              />
            ) : (
              <button
                className="text-white text-[20px] font-semibold hover:opacity-90"
                onClick={() => setEditingTitle(true)}
                title="Editar título"
              >
                {normalizedTitle}
              </button>
            )}
          </div>

          {/* modelo */}
          <div className="relative">
            <button
              onClick={() => setModelOpen((s) => !s)}
              className="flex items-center gap-2 rounded-full bg-white/5 ring-1 ring-white/10 hover:bg-white/10 transition px-3 py-1.5"
            >
              <img src="/images/logo.svg" alt="Aura" className="w-5 h-5" />
              <span className="text-sm text-white/90">{model}</span>
              <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 text-white/70">
                <path d="M5.5 7.5l4.5 4.5 4.5-4.5H5.5z" />
              </svg>
            </button>
            {modelOpen && (
              <div
                className="absolute right-0 mt-2 w-[180px] rounded-xl bg-[#10131d] ring-1 ring-white/10 shadow-xl p-1"
                onMouseLeave={() => setModelOpen(false)}
              >
                {(['Aura v1', 'Aura v2'] as const).map((opt) => (
                  <button
                    key={opt}
                    onClick={() => {
                      setModel(opt);
                      setModelOpen(false);
                    }}
                    className={`w-full text-left px-3 py-2 rounded-lg hover:bg-white/5 text-sm ${
                      model === opt ? 'text-white' : 'text-white/80'
                    }`}
                  >
                    {opt}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* mensajes (scroll) */}
      <div ref={scrollerRef} className="scroll-slim mt-6 flex-1 min-h-0 overflow-y-auto pr-1">
        <div className="flex flex-col gap-5 pb-[116px]">
          {messages.map((m) => (
            <Bubble key={m.id} msg={m} />
          ))}
        </div>
      </div>

      {/* composer fijo abajo */}
      <div className="sticky bottom-0 left-0 right-0 z-10">
        <div className="pointer-events-none h-3 bg-gradient-to-t from-[#070a14] to-transparent" />
        <div className="bg-[#070a14]/90 backdrop-blur border-t border-white/10">
          <div className="px-3 sm:px-4 py-3 pb-[calc(10px+env(safe-area-inset-bottom,0px))]">
            <div className="rounded-[20px] bg-white/5 ring-1 ring-white/10 p-3">
              <div className="flex items-center gap-3">
                <button
                  className="grid place-items-center w-9 h-9 rounded-lg bg-white/5 ring-1 ring-white/10 hover:bg-white/10 transition"
                  title="Adjuntar"
                >
                  <svg viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5 text-white/80">
                    <path d="M12 2a5 5 0 015 5v8a3 3 0 11-6 0V8a1 1 0 112 0v7a1 1 0 102 0V7a3 3 0 10-6 0v8a5 5 0 1010 0V8h2v7a7 7 0 11-14 0V7a5 5 0 015-5z" />
                  </svg>
                </button>

                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                  placeholder="¿En qué puedo ayudarte hoy?"
                  className="flex-1 bg-transparent outline-none text-white placeholder:text-white/40 text-[15px]"
                />

                <button
                  className="grid place-items-center w-9 h-9 rounded-lg bg-white/5 ring-1 ring-white/10 hover:bg-white/10 transition"
                  title="Voz"
                >
                  <svg viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5 text-white/80">
                    <path d="M12 14a3 3 0 003-3V6a3 3 0 10-6 0v5a3 3 0 003 3z" />
                    <path d="M5 11a7 7 0 0014 0h-2a5 5 0 11-10 0H5z" />
                    <path d="M11 19h2v3h-2z" />
                  </svg>
                </button>

                <button
                  onClick={handleSend}
                  className="grid place-items-center w-10 h-10 rounded-full bg-[#7B2FE3] hover:bg-[#6c29c9] active:scale-[0.98] transition"
                  title="Enviar"
                >
                  <svg viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5 text-white">
                    <path d="M3 11l18-8-8 18-2-7-8-3z" />
                  </svg>
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

function Bubble({ msg }: { msg: Msg }) {
  const isUser = msg.author === 'user';
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    if (!msg.text) return;
    try {
      await navigator.clipboard.writeText(msg.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {}
  };

  return (
    <div className={`group/message flex items-end gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div className="shrink-0">
        {isUser ? (
          <img
            src={msg.avatar || '/images/avatar_demo.jpg'}
            alt={msg.name}
            className="w-8 h-8 rounded-full object-cover ring-1 ring-white/10"
          />
        ) : (
          <div className="w-8 h-8 rounded-full grid place-items-center bg-white/5 ring-1 ring-white/10">
            <img src="/images/logo.svg" alt="Aura" className="w-4 h-4" />
          </div>
        )}
      </div>
      {/* Nombre + burbuja */}
      <div className={`max-w-[78%] ${isUser ? 'items-end text-right' : ''}`}>
        <div className="text-[11px] text-white/50 mb-1">{msg.name}</div>

        {msg.typing ? (
          <div
            className={`inline-flex items-center gap-1 px-3 py-2 rounded-2xl ${
              isUser
                ? 'bg-[#7B2FE3] text-white rounded-br-md'
                : 'bg-white/5 ring-1 ring-white/10 text-white rounded-bl-md'
            }`}
          >
            <Dot delay="0ms" />
            <Dot delay="150ms" />
            <Dot delay="300ms" />
          </div>
        ) : (
          <div
            className={`inline-block px-4 py-2 text-[15px] leading-relaxed rounded-2xl ${
              isUser
                ? 'bg-[#7B2FE3] text-white rounded-br-md'
                : 'bg-white/5 ring-1 ring-white/10 text-white rounded-bl-md'
            }`}
          >
            {msg.text}
          </div>
        )}

        {/* Hora + copiar: oculto y aparece al hover/focus */}
        <div
          className={`mt-1 flex items-center gap-2 ${
            isUser ? 'justify-end' : 'justify-start'
          } opacity-0 group-hover/message:opacity-100 group-focus-within/message:opacity-100 transition-opacity duration-200`}
        >
          <span className="text-[11px] text-white/35">
            {new Date(msg.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>

          {msg.text && (
            <button
              onClick={handleCopy}
              className="p-1.5 rounded-md hover:bg-white/5 ring-1 ring-transparent hover:ring-white/10 active:scale-95 transition"
              aria-label="Copiar mensaje"
              title={copied ? '¡Copiado!' : 'Copiar'}
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

function CopyIcon(props: any) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M8 7a2 2 0 012-2h7a2 2 0 012 2v9a2 2 0 01-2 2h-7a2 2 0 01-2-2V7zm-3 3h1v8a3 3 0 003 3h8v1H9a4 4 0 01-4-4v-8z" />
    </svg>
  );
}

<>
  {/* ...tu JSX del ChatPanel... */}
  <style>{`
    /* WebKit (Chrome/Edge/Safari) */
    .scroll-slim::-webkit-scrollbar {
      width: 8px;              /* solo la “barra” fina */
      background: transparent; /* track invisible */
    }
    .scroll-slim::-webkit-scrollbar-track {
      background: transparent; /* track invisible */
    }
    .scroll-slim::-webkit-scrollbar-thumb {
      background-color: #4B535B;    /* "palo" */
      border-radius: 8px;
      border: 2px solid transparent; /* deja espacio visual al track */
      background-clip: content-box;
    }

    /* Firefox */
    .scroll-slim {
      scrollbar-width: thin;
      scrollbar-color: #4B535B transparent; /* thumb / track */
    }
  `}</style>
</>;
