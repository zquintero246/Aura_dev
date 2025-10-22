import React from 'react';

export type Group = {
  id: string;
  name: string;
  members: number;
  unread?: number; // contador opcional
  pinned?: boolean;
};

type Props = {
  myGroups: Group[];
  explore: Group[]; // otros grupos / públicos / sugeridos
  selectedId?: string;
  onSearch?: (q: string) => void;
  onSelect?: (id: string) => void;
  onCreate?: () => void;
  railWidth?: number; // ancho del IconRail (default 72)
};

const GroupsPanel: React.FC<Props> = ({
  myGroups,
  explore,
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
      aria-label="Listado de grupos"
    >
      {/* HEADER */}
      <div className="px-5 pt-5 pb-3">
        <div className="flex items-center gap-2">
          <h2 className="text-[20px] md:text-[22px] font-semibold text-white">Grupos</h2>
          <button
            onClick={onCreate}
            className="ml-auto grid place-items-center w-8 h-8 rounded-lg
                       bg-[#8B3DFF] hover:bg-[#7a2fe3] active:scale-95 transition"
            title="Crear grupo"
            aria-label="Crear grupo"
          >
            <PlusIcon className="w-4 h-4 text-white" />
          </button>
        </div>

        {/* Search */}
        <div className="mt-3 relative">
          <span className="absolute left-3 top-1/2 -translate-y-1/2">
            <SearchIcon className="w-4 h-4 text-white/50" />
          </span>
          <input
            type="text"
            placeholder="Buscar grupo"
            onChange={(e) => onSearch?.(e.target.value)}
            className="w-full h-9 pl-9 pr-3 rounded-xl bg-white/5 text-[13px]
                       placeholder:text-white/40 text-white/90 outline-none
                       ring-1 ring-white/10 focus:ring-2 focus:ring-[#8B3DFF66]
                       transition"
          />
        </div>
      </div>

      <div className="h-px bg-white/10" />

      {/* LISTAS */}
      <div className="flex flex-col h-[calc(100vh-110px)] overflow-y-auto">
        <Section title="MIS GRUPOS">
          {myGroups.length === 0 ? (
            <EmptyRow text="Todavía no perteneces a ningún grupo" />
          ) : (
            myGroups.map((g) => (
              <GroupRow
                key={g.id}
                active={g.id === selectedId}
                name={g.name}
                members={g.members}
                unread={g.unread}
                onClick={() => onSelect?.(g.id)}
              />
            ))
          )}
        </Section>

        <Section title="EXPLORAR">
          {explore.length === 0 ? (
            <EmptyRow text="Sin sugerencias por ahora" />
          ) : (
            explore.map((g) => (
              <GroupRow
                key={g.id}
                active={g.id === selectedId}
                name={g.name}
                members={g.members}
                unread={g.unread}
                onClick={() => onSelect?.(g.id)}
              />
            ))
          )}
        </Section>

        <div className="pb-6" />
      </div>
    </aside>
  );
};

export default GroupsPanel;

/* ---------- Subcomponentes ---------- */

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="px-3 py-4">
      <div className="px-2 mb-2 text-[11px] tracking-[0.08em] text-white/45">{title}</div>
      <div className="space-y-1">{children}</div>
    </section>
  );
}

function GroupRow({
  name,
  members,
  unread,
  active,
  onClick,
}: {
  name: string;
  members: number;
  unread?: number;
  active?: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-3 rounded-xl flex items-center gap-3
                  transition select-none
                  ${active ? 'bg-white/6 ring-1 ring-white/10 text-white' : 'hover:bg-white/4 text-white/85'}`}
      title={name}
    >
      {/* Avatar circular con iniciales */}
      <div className="w-8 h-8 rounded-full bg-white/10 grid place-items-center text-[12px] font-semibold">
        {getInitials(name)}
      </div>

      <div className="min-w-0 flex-1">
        <div className="truncate text-[14px]">{name}</div>
        <div className="mt-[2px] text-[12px] text-white/50">{members} miembros</div>
      </div>

      {typeof unread === 'number' && unread > 0 && (
        <span className="ml-auto px-2 py-[2px] text-[11px] rounded-full bg-[#8B3DFF] text-white">
          {unread}
        </span>
      )}
    </button>
  );
}

function EmptyRow({ text }: { text: string }) {
  return <div className="px-3 py-3 rounded-xl text-[13px] text-white/40">{text}</div>;
}

/* ---------- Iconitos inline ---------- */
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

/* Util */
function getInitials(name: string) {
  return name
    .split(' ')
    .map((n) => n[0]?.toUpperCase())
    .slice(0, 2)
    .join('');
}
