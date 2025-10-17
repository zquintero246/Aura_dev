import React from 'react';

export type Conversation = {
  id: string;
  title: string;
  pinned?: boolean;
};

type Props = {
  /** Conversaciones ancladas */
  pinned: Conversation[];
  /** Conversaciones recientes */
  recent: Conversation[];
  /** Qué item está seleccionado para pintar el estado activo */
  selectedId?: string;
  /** Buscar (texto libre) */
  onSearch?: (q: string) => void;
  /** Abrir una conversación */
  onSelect?: (id: string) => void;
  /** Crear nueva conversación */
  onCreate?: () => void;
  /** Ancho del IconRail para ubicarnos justo al lado (por defecto 72px) */
  railWidth?: number;
  /** Actualizar título desde el ChatPanel */
  onUpdateTitle?: (id: string, title: string) => void;
};

/**
 * Columna fija para listar conversaciones.
 * Se coloca FIJA a la izquierda, inmediatamente a la derecha del IconRail.
 */

const ConversationsPanel: React.FC<Props> = ({
  pinned,
  recent,
  selectedId,
  onSearch,
  onSelect,
  onCreate,
  railWidth = 72,
}) => {
  return (
    <aside
      className="fixed top-0 bottom-0 z-20
                 bg-[#0B0F1A] text-white/90 border-r border-white/10
                 w-[300px] md:w-[320px]"
      style={{ left: railWidth }}
      aria-label="Listado de conversaciones"
    >
      {/* HEADER */}
      <div className="px-5 pt-5 pb-3">
        <div className="flex items-center gap-2">
          <h2 className="text-[20px] md:text-[22px] font-semibold text-white">Conversaciones</h2>

          {/* (Opcional) icono para acciones rápidas en el título */}
          <button
            className="ml-auto grid place-items-center w-7 h-7 rounded-md
                       hover:bg-white/5 active:scale-95 transition"
            title="Cerrar Barra"
            aria-label="Cerrar Barra"
          >
            <SquareIcon className="w-4 h-4 text-white/70" />
          </button>
        </div>

        {/* Search */}
        <div className="mt-3 relative">
          <span className="absolute left-3 top-1/2 -translate-y-1/2">
            <SearchIcon className="w-4 h-4 text-white/50" />
          </span>
          <input
            type="text"
            placeholder="Buscar conversación"
            onChange={(e) => onSearch?.(e.target.value)}
            className="w-full h-9 pl-9 pr-3 rounded-xl bg-white/5 text-[13px]
                       placeholder:text-white/40 text-white/90 outline-none
                       ring-1 ring-white/10 focus:ring-2 focus:ring-[#8B3DFF66]
                       transition"
          />
        </div>
      </div>

      <div className="h-px bg-white/10" />

      {/* LISTA SCROLLABLE */}
      <div className="flex flex-col h-[calc(100vh-110px)] overflow-y-auto">
        {/* PINNED */}
        <Section title="ANCLADOS">
          {pinned.length === 0 ? (
            <EmptyRow text="Sin anclados por ahora" />
          ) : (
            pinned.map((c) => (
              <Row
                key={c.id}
                active={c.id === selectedId}
                label={c.title}
                onClick={() => onSelect?.(c.id)}
              />
            ))
          )}
        </Section>

        {/* RECENTES */}
        <Section
          title="RECIENTES"
          action={
            <button
              onClick={onCreate}
              className="grid place-items-center w-7 h-7 rounded-lg
                         bg-[#8B3DFF] hover:bg-[#7a2fe3] active:scale-95
                         transition shadow-sm"
              title="Nueva conversación"
              aria-label="Nueva conversación"
            >
              <PlusIcon className="w-4 h-4 text-white" />
            </button>
          }
        >
          {recent.length === 0 ? (
            <EmptyRow text="Aún no hay recientes" />
          ) : (
            recent.map((c) => (
              <Row
                key={c.id}
                active={c.id === selectedId}
                label={c.title}
                onClick={() => onSelect?.(c.id)}
              />
            ))
          )}
        </Section>

        <div className="pb-6" />
      </div>
    </aside>
  );
};

export default ConversationsPanel;

/* ---------- Subcomponentes ---------- */

function Section({
  title,
  action,
  children,
}: {
  title: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="px-3 py-4">
      <div className="flex items-center gap-2 px-2 mb-2">
        <span className="text-[11px] tracking-[0.08em] text-white/45">{title}</span>
        <div className="ml-auto">{action}</div>
      </div>
      <div className="space-y-1">{children}</div>
    </section>
  );
}

function Row({
  label,
  active,
  onClick,
}: {
  label: string;
  active?: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-3 rounded-xl
                  transition select-none
                  ${
                    active
                      ? 'bg-white/6 ring-1 ring-white/10 text-white'
                      : 'hover:bg-white/4 text-white/80'
                  }`}
      title={label}
    >
      <span className="block truncate text-[14px]">{label}</span>
    </button>
  );
}

function EmptyRow({ text }: { text: string }) {
  return <div className="px-3 py-3 rounded-xl text-[13px] text-white/40">{text}</div>;
}

/* ---------- Iconos simples (inline) ---------- */
function SearchIcon(props: any) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M10 3a7 7 0 105.29 12.29l4.21 4.2 1.4-1.4-4.2-4.21A7 7 0 0010 3zm0 2a5 5 0 110 10 5 5 0 010-10z" />
    </svg>
  );
}
function PlusIcon(props: any) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M11 11V6h2v5h5v2h-5v5h-2v-5H6v-2z" />
    </svg>
  );
}
function SquareIcon(props: any) {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M15 2.8125C15 2.31522 14.8025 1.83831 14.4508 1.48667C14.0992 1.13504 13.6223 0.9375 13.125 0.9375H1.875C1.37772 0.9375 0.900806 1.13504 0.549175 1.48667C0.197544 1.83831 0 2.31522 0 2.8125L0 12.1875C0 12.6848 0.197544 13.1617 0.549175 13.5133C0.900806 13.865 1.37772 14.0625 1.875 14.0625H13.125C13.6223 14.0625 14.0992 13.865 14.4508 13.5133C14.8025 13.1617 15 12.6848 15 12.1875V2.8125ZM10.3125 1.875V13.125H1.875C1.62636 13.125 1.3879 13.0262 1.21209 12.8504C1.03627 12.6746 0.9375 12.4361 0.9375 12.1875V2.8125C0.9375 2.56386 1.03627 2.3254 1.21209 2.14959C1.3879 1.97377 1.62636 1.875 1.875 1.875H10.3125ZM11.25 1.875H13.125C13.3736 1.875 13.6121 1.97377 13.7879 2.14959C13.9637 2.3254 14.0625 2.56386 14.0625 2.8125V12.1875C14.0625 12.4361 13.9637 12.6746 13.7879 12.8504C13.6121 13.0262 13.3736 13.125 13.125 13.125H11.25V1.875Z"
        fill="white"
      />
    </svg>
  );
}
