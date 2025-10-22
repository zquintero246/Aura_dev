import React, { useEffect, useRef, useState } from 'react';

/** Tipos */
type Role = 'user' | 'assistant';
export type ChatMessage = { id: string; role: Role; content: string; time?: string };

type ChatsViewProps = {
  /** listado inicial de mensajes (puede venir de tu store/BD) */
  initialMessages?: ChatMessage[];
  /** nombre para saludar (si no hay mensajes) */
  firstName?: string;
  /** se muestra “Aura está escribiendo…” */
  isTyping?: boolean;
  /** callbacks para enganchar con backend más adelante */
  onSend?: (text: string) => void;
  /** valor inicial del título del chat */
  initialTitle?: string;
};

const ChatsView: React.FC<ChatsViewProps> = ({
  initialMessages = [],
  firstName = 'Santiago',
  isTyping = false,
  onSend,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [input, setInput] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  /** Auto-scroll al final cuando llegan mensajes o typing */
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }, [messages.length, isTyping]);

  const showEmpty = messages.length === 0;

  const handleSend = () => {
    const text = input.trim();
    if (!text) return;
    const msg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
    setMessages((prev) => [...prev, msg]);
    setInput('');
    onSend?.(text);
  };

  return (
    // Contenedor principal: usa flexbox para la estructura general y un color de fondo.
    <section className="flex flex-col h-screen w-full bg-[#0d1117] text-white">
      {/* Encabezado: ahora es un elemento de flex fijo en la parte superior. */}
      <header className="flex-shrink-0 w-full px-6 md:px-8 py-4 border-b border-white/5">
        <div className="max-w-[1100px] mx-auto flex items-center justify-between">
          <h1 className="text-white text-[20px] md:text-[22px] font-semibold">Gemini</h1>
          <button
            className="ml-4 grid place-items-center w-9 h-9 rounded-lg border border-white/10 hover:bg-white/5 transition"
            title="Opciones del chat"
          >
            <DotsIcon className="w-4 h-4 text-white/70" />
          </button>
        </div>
      </header>

      {/* Área de Mensajes: Este div crece para ocupar el espacio disponible y permite el scroll. */}
      <main ref={scrollRef} className="flex-1 w-full overflow-y-auto px-3 sm:px-6 md:px-8">
        <div className="max-w-[1100px] mx-auto py-6 md:py-8 space-y-6">
          {/* Estado vacío: mejorado para parecerse a Gemini. */}
          {showEmpty && (
            <div className="flex flex-col items-center justify-center h-full py-24 text-center">
              <div className="w-16 h-16 mb-6 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full flex items-center justify-center">
                <svg
                  className="w-8 h-8 text-white"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M12.93,2.33C12.04,3.22 12.04,4.7 12.93,5.59C13.82,6.48 15.3,6.48 16.19,5.59C17.08,4.7 17.08,3.22 16.19,2.33C15.3,1.44 13.82,1.44 12.93,2.33M5.59,12.93C4.7,12.04 3.22,12.04 2.33,12.93C1.44,13.82 1.44,15.3 2.33,16.19C3.22,17.08 4.7,17.08 5.59,16.19C6.48,15.3 6.48,13.82 5.59,12.93M16.19,18.41C15.3,19.3 13.82,19.3 12.93,18.41C12.04,17.52 12.04,16.04 12.93,15.15C13.82,14.26 15.3,14.26 16.19,15.15C17.08,16.04 17.08,17.52 16.19,18.41M18.41,12.93C17.52,12.04 16.04,12.04 15.15,12.93C14.26,13.82 14.26,15.3 15.15,16.19C16.04,17.08 17.52,17.08 18.41,16.19C19.3,15.3 19.3,13.82 18.41,12.93M8.49,7.55C8.49,7.55 8.49,7.55 8.49,7.55C7.6,6.66 6.12,6.66 5.23,7.55C4.34,8.44 4.34,9.92 5.23,10.81C6.12,11.7 7.6,11.7 8.49,10.81C9.38,9.92 9.38,8.44 8.49,7.55M18.77,5.23C17.88,4.34 16.4,4.34 15.51,5.23C14.62,6.12 14.62,7.6 15.51,8.49C16.4,9.38 17.88,9.38 18.77,8.49C19.66,7.6 19.66,6.12 18.77,5.23Z"
                    fill="currentColor"
                  />
                </svg>
              </div>
              <h2 className="text-[26px] md:text-[28px] font-bold text-white">
                Hola,{' '}
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">
                  {firstName}
                </span>
                .
              </h2>
              <p className="mt-3 text-white/60">¿En qué puedo ayudarte hoy?</p>
            </div>
          )}

          {/* Mensajes */}
          {messages.map((m) => (
            <MessageBubble key={m.id} role={m.role} time={m.time}>
              {m.content}
            </MessageBubble>
          ))}

          {/* Typing */}
          {isTyping && <TypingBubble />}
        </div>
      </main>

      {/* Composer: el panel de entrada de texto, fijo en la parte inferior. */}
      <footer className="flex-shrink-0 w-full border-t border-white/5 px-3 sm:px-6 md:px-8 py-4">
        <div className="max-w-[1100px] mx-auto">
          <div className="flex items-center gap-3">
            <button
              className="grid place-items-center w-9 h-9 rounded-xl border border-white/10 hover:bg-white/5 transition"
              title="Añadir"
            >
              <PlusIcon className="w-4 h-4 text-white/80" />
            </button>
            <div className="flex-1 relative">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                className="w-full h-12 rounded-xl bg-white/5 text-white placeholder:text-white/40 ring-1 ring-white/10 focus:ring-2 focus:ring-[#8B3DFF70] outline-none pl-4 pr-28"
                placeholder="Escribe un mensaje..."
              />
              <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-2">
                <button
                  className="grid place-items-center w-9 h-9 rounded-lg border border-white/10 hover:bg-white/5 transition"
                  title="Hablar"
                >
                  <MicIcon className="w-4 h-4 text-white/70" />
                </button>
                <button
                  onClick={handleSend}
                  disabled={!input.trim()}
                  className="grid place-items-center w-10 h-10 rounded-full bg-[#8B3DFF] hover:bg-[#7a31e6] active:scale-[0.98] transition disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Enviar"
                >
                  <SendIcon className="w-4 h-4 text-white" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </section>
  );
};

export default ChatsView;

/* ===================== Subcomponentes ===================== */

function MessageBubble({
  role,
  children,
  time,
}: {
  role: Role;
  children: React.ReactNode;
  time?: string;
}) {
  const isUser = role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`
          max-w-[78%] md:max-w-[70%] rounded-2xl p-4 shadow
          ${isUser ? 'bg-white/10 text-white' : 'bg-[#12151f] text-white ring-1 ring-white/10'}
        `}
      >
        <div className="whitespace-pre-wrap leading-relaxed text-[14.5px]">{children}</div>
        {time && <div className="mt-2 text-[11px] text-white/40 text-right">{time}</div>}
      </div>
    </div>
  );
}

function TypingBubble() {
  return (
    <div className="flex justify-start">
      <div className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-[#12151f] ring-1 ring-white/10">
        <TypingDots />
        <span className="text-xs text-white/60">Gemini está escribiendo…</span>
      </div>
    </div>
  );
}

function TypingDots() {
  return (
    <span className="inline-flex items-center gap-1">
      <Dot />
      <Dot delay="150ms" />
      <Dot delay="300ms" />
    </span>
  );
}
function Dot({ delay = '0ms' }: { delay?: string }) {
  return (
    <span
      className="w-1.5 h-1.5 rounded-full bg-white/60 inline-block animate-bounce"
      style={{ animationDelay: delay }}
    />
  );
}

/* ===================== Iconos inline ===================== */

function DotsIcon(props: any) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <circle cx="5" cy="12" r="2" />
      <circle cx="12" cy="12" r="2" />
      <circle cx="19" cy="12" r="2" />
    </svg>
  );
}
function PlusIcon(props: any) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M11 5h2v6h6v2h-6v6h-2v-6H5v-2h6z" />
    </svg>
  );
}
function MicIcon(props: any) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M12 14a3 3 0 003-3V6a3 3 0 10-6 0v5a3 3 0 003 3zm5-3a5 5 0 01-10 0H5a7 7 0 0014 0h-2zM11 19h2v3h-2v-3z" />
    </svg>
  );
}
function SendIcon(props: any) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M2 21l21-9L2 3v7l15 2-15 2v7z" />
    </svg>
  );
}
