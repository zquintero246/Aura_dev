// AppLayout.tsx
import React, { useEffect, useMemo, useState } from 'react';
import IconRail from './IconRail';
import ConversationsPanel, { Conversation } from './ConversationsPanel';
import GroupsPanel, { Group } from './GroupsPanel';
import MainPanel from './MainPanel';
import ChatPanel from './ChatPanel';
import { me, logout, User } from '../../lib/auth';
import { useNavigate } from 'react-router-dom';

/** Claves del rail */
type SectionKey = 'chats' | 'group' | 'project' | 'telemetry';

/** Lo que está abierto en el panel AZUL */
type Selection =
  | { type: 'chat'; id: string }
  | { type: 'group'; id: string }
  | { type: 'project'; id: string }
  | { type: 'telemetry'; view: 'overview' | 'errors' | 'latency' }
  | null;

const RAIL_W = 72; // ancho del IconRail (rojo)
const SIDE_W = 320; // ancho del panel verde (contextual)

export default function AppLayout() {
  const navigate = useNavigate();
  /** 1) Rail activo (VERDE cambia con esto) */
  const [activeRail, setActiveRail] = useState<SectionKey>('chats');

  /** 2) Selección abierta (AZUL cambia con esto) */
  const [selection, setSelection] = useState<Selection>(null);

  // ---- DATA (ahora en estado para poder actualizar títulos) ----
  const [pinned, setPinned] = useState<Conversation[]>([
    { id: 'c1', title: 'Conversación de ejemplo', pinned: true },
  ]);
  const [recent, setRecent] = useState<Conversation[]>([{ id: 'c2', title: 'Otra conversación' }]);

  const [myGroups] = useState<Group[]>([
    { id: 'g1', name: 'Equipo Backend', members: 12, unread: 3 },
    { id: 'g2', name: 'Diseño UX', members: 7 },
  ]);
  const [explore] = useState<Group[]>([
    { id: 'g3', name: 'IA & LLMs', members: 56 },
    { id: 'g4', name: 'DevOps Lovers', members: 31, unread: 1 },
  ]);
  // ---------------------------------------------------------------

  /** Margen del contenido central (AZUL) para no pisar rail+panel */
  const mainPaddingLeft = useMemo(() => `${RAIL_W + SIDE_W}px`, []);

  /** Crear nueva conversación (demo) */
  const handleCreateChat = () => {
    const id = crypto.randomUUID();
    const nueva: Conversation = { id, title: 'Nueva conversación' };
    setRecent((r) => [nueva, ...r]);
    setSelection({ type: 'chat', id });
  };

  /** Buscar chat (demo) */
  const handleSearchChat = (q: string) => {
    console.log('buscar chat:', q);
  };

  /** Cuando se cambia el título en el ChatPanel -> reflejar en panel verde */
  const handleChatTitleChange = (id: string, newTitle: string) => {
    setPinned((prev) => prev.map((c) => (c.id === id ? { ...c, title: newTitle } : c)));
    setRecent((prev) => prev.map((c) => (c.id === id ? { ...c, title: newTitle } : c)));
  };

  /** Conversación activa (si hay selección de chat) */
  const activeChat =
    (selection?.type === 'chat' &&
      (pinned.find((c) => c.id === selection.id) || recent.find((c) => c.id === selection.id))) ||
    null;

  // --- User session data ---
  const [user, setUser] = useState<User | null>(null);
  useEffect(() => {
    (async () => {
      try {
        const res = await me();
        setUser(res?.user || null);
      } catch {
        setUser(null);
      }
    })();
  }, []);

  const handleLogout = async () => {
    try { await logout(); } catch {}
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-[#070a14] text-white">
      {/* ----------- ROJO: IconRail ----------- */}
      <IconRail
        active={activeRail}
        onSelect={(key) => setActiveRail(key)}
        onToggleTheme={() => {}}
        avatarUrl={user?.avatar_url || undefined}
        userName={user?.name || 'Tu perfil'}
        onProfile={() => navigate('/profile')}
        onSettings={() => navigate('/settings')}
        onChangePassword={() => navigate('/change-password')}
        onLogout={handleLogout}
      />

      {/* ----------- VERDE: Panel contextual por opción ----------- */}
      {activeRail === 'chats' && (
        <ConversationsPanel
          railWidth={RAIL_W}
          pinned={pinned}
          recent={recent}
          selectedId={selection?.type === 'chat' ? selection.id : undefined}
          onSelect={(id) => setSelection({ type: 'chat', id })}
          onCreate={handleCreateChat}
          onSearch={handleSearchChat}
        />
      )}

      {activeRail === 'group' && (
        <GroupsPanel
          railWidth={RAIL_W}
          myGroups={myGroups}
          explore={explore}
          selectedId={selection?.type === 'group' ? selection.id : undefined}
          onSelect={(id) => setSelection({ type: 'group', id })}
          onCreate={() => console.log('crear grupo')}
          onSearch={(q) => console.log('buscar grupo:', q)}
        />
      )}

      {/* TODO: ProjectPanel / TelemetryPanel igual que arriba */}

      {/* ----------- AZUL: Panel principal (solo cambia por selection) ----------- */}
      <main
        style={{ paddingLeft: mainPaddingLeft }}
        className="pt-6 pr-6 pb-6 h-screen overflow-hidden"
      >
        {/* Si hay un chat seleccionado, mostramos ChatPanel.
            Si no, mantenemos tu MainPanel (placeholder/genérico). */}
        {activeChat ? (
          <ChatPanel conversationId={activeChat.id} onTitleChange={handleChatTitleChange} />
        ) : (
          <MainPanel selection={selection} />
        )}
      </main>
    </div>
  );
}
