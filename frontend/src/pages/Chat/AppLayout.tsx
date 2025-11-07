// AppLayout.tsx
import React, { useEffect, useMemo, useState } from 'react';
import IconRail from './IconRail';
import ConversationsPanel, { Conversation } from './ConversationsPanel';
import GroupsPanel, { Group } from './GroupsPanel';
import MainPanel from './MainPanel';
import ProfilePanel from '../Account/Profile';
import ChatPanel from './ChatPanel';
import { me, logout, User, ensureChatToken } from '../../lib/auth';
import { setChatToken } from '../../lib/chatApi';
import { useNavigate } from 'react-router-dom';
import { listConversations, deleteConversation as deleteChat } from '../../lib/conversations';

type SectionKey = 'chats' | 'group' | 'project' | 'telemetry';

type Selection =
  | { type: 'chat'; id: string }
  | { type: 'group'; id: string }
  | { type: 'project'; id: string }
  | { type: 'telemetry'; view: 'overview' | 'errors' | 'latency' }
  | { type: 'profile' }
  | null;

const RAIL_W = 72;
const SIDE_W = 320;

export default function AppLayout() {
  const navigate = useNavigate();
  const [activeRail, setActiveRail] = useState<SectionKey>('chats');
  const [conversationsOpen, setConversationsOpen] = useState(true);
  const [selection, setSelection] = useState<Selection>(null);
  const [bootStage, setBootStage] = useState<'loading' | 'transition' | 'done'>('loading');

  // Conversations state (no demos)
  const [pinned, setPinned] = useState<Conversation[]>([]);
  const [recent, setRecent] = useState<Conversation[]>([]);

  // Groups (demo)
  const [myGroups] = useState<Group[]>([
    { id: 'g1', name: 'Equipo Backend', members: 12, unread: 3 },
    { id: 'g2', name: 'Diseno UX', members: 7 },
  ]);
  const [explore] = useState<Group[]>([
    { id: 'g3', name: 'IA & LLMs', members: 56 },
    { id: 'g4', name: 'DevOps Lovers', members: 31, unread: 1 },
  ]);

  const mainPaddingLeft = useMemo(() => {
    const side =
      activeRail === 'chats'
        ? conversationsOpen
          ? SIDE_W
          : 0
        : activeRail === 'group'
          ? SIDE_W
          : 0;
    return `${RAIL_W + side}px`;
  }, [activeRail, conversationsOpen]);

  // Create conversation (optimistic; materialized on first send)
  const handleCreateChat = () => {
    const tempId = `tmp-${crypto.randomUUID()}`;
    const tempConv: Conversation = { id: tempId, title: 'Nueva conversación' };
    setRecent((r) => [tempConv, ...r]);
    setSelection({ type: 'chat', id: tempId });
  };

  const handleSearchChat = (q: string) => {
    console.log('buscar chat:', q);
  };

  const normalizeAutoTitle = (t: string): string => {
    try {
      let s = (t || '').replace(/\s+/g, ' ').trim();
      if (!s) return '';

      // quitar signos iniciales tipo ¿¡ y espacios sobrantes
      s = s.replace(/^[¿¡\s]+/u, '').replace(/[\s.?!,;:]+$/u, '');

      const words = s.split(/\s+/u).slice(0, 6);
      let out = words.join(' ');
      if (out) out = out.charAt(0).toUpperCase() + out.slice(1);

      return out;
    } catch {
      return t;
    }
  };

  const handleChatTitleChange = (id: string, newTitle: string) => {
    const normalized = normalizeAutoTitle(newTitle);
    setPinned((prev) => prev.map((c) => (c.id === id ? { ...c, title: normalized } : c)));
    setRecent((prev) => prev.map((c) => (c.id === id ? { ...c, title: normalized } : c)));
  };

  // Pin/unpin with limit 3
  const handleTogglePin = (id: string, nextPinned: boolean) => {
    if (nextPinned) {
      if (pinned.length >= 3) {
        try {
          window.alert('Solo puedes anclar hasta 3 conversaciones');
        } catch {}
        return;
      }
      const item = recent.find((c) => c.id === id);
      if (item && !pinned.find((p) => p.id === id)) {
        setRecent((r) => r.filter((c) => c.id !== id));
        setPinned((p) => [...p, { ...item, pinned: true }]);
      }
    } else {
      const item = pinned.find((c) => c.id === id);
      if (item) {
        setPinned((p) => p.filter((c) => c.id !== id));
        setRecent((r) => [{ ...item, pinned: false }, ...r.filter((c) => c.id !== id)]);
      }
    }
  };

  const activeChat =
    (selection?.type === 'chat' &&
      (pinned.find((c) => c.id === selection.id) || recent.find((c) => c.id === selection.id))) ||
    null;

  const handleDeleteChat = async (id: string) => {
    if (!id) return;
    try {
      const ok = window.confirm('¿Borrar esta conversación? Esta acción no se puede deshacer.');
      if (!ok) return;
    } catch {}
    const wasActive = selection?.type === 'chat' && selection.id === id;
    setPinned((p) => p.filter((c) => c.id !== id));
    setRecent((r) => r.filter((c) => c.id !== id));
    if (wasActive) {
      // pick next available conversation
      const next = pinned.find((c) => c.id !== id) || recent.find((c) => c.id !== id) || null;
      setSelection(next ? { type: 'chat', id: next.id } : null);
    }
    try {
      await deleteChat(id);
    } catch (e) {
      try {
        console.error('Failed to delete conversation remotely:', e);
      } catch {}
      try {
        window.alert('No se pudo borrar en el servidor.');
      } catch {}
    }
  };

  // User session
  const [user, setUser] = useState<User | null>(null);
  useEffect(() => {
    (async () => {
      try {
        const res = await me();
        setUser(res?.user || null);
        // Ensure chat token is bound to this user; mint if missing or mismatched
        try {
          const uid = res?.user?.id;
          const token = await ensureChatToken(uid);
          if (token) setChatToken(token);
        } catch {}
      } catch {
        setUser(null);
      }
    })();
  }, []);

  const handleLogout = async () => {
    try {
      await logout();
    } catch {}
    navigate('/login');
  };

  // Close conversations panel via global event
  useEffect(() => {
    const handler = () => setConversationsOpen(false);
    window.addEventListener('aura:conversations:close', handler as EventListener);
    return () => {
      window.removeEventListener('aura:conversations:close', handler as EventListener);
    };
  }, []);

  // Theme
  useEffect(() => {
    const saved = localStorage.getItem('aura:theme');
    if (saved === 'light') document.documentElement.classList.add('theme-light');
  }, []);

  const handleToggleTheme = () => {
    const el = document.documentElement;
    el.classList.add('theme-xfade');
    const isLight = el.classList.toggle('theme-light');
    localStorage.setItem('aura:theme', isLight ? 'light' : 'dark');
    window.setTimeout(() => el.classList.remove('theme-xfade'), 260);
  };

  // Initial bootstrap: user, token, conversations. Shows loading overlay until ready.
  useEffect(() => {
    (async () => {
      try {
        const res = await me();
        setUser(res?.user || null);
        try {
          const uid = res?.user?.id;
          const token = await ensureChatToken(uid);
          if (token) setChatToken(token);
        } catch {}
        try {
          const items = await listConversations();
          setRecent(items);
          if (!selection && items.length) setSelection({ type: 'chat', id: items[0].id });
        } catch {}
      } catch {
        setUser(null);
      } finally {
        setTimeout(() => setBootStage('transition'), 100);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Safety: if the opacity transition does not fire (browser quirk),
  // auto-complete the transition after 600ms once we enter 'transition'.
  useEffect(() => {
    if (bootStage !== 'transition') return;
    const t = setTimeout(() => setBootStage('done'), 600);
    return () => clearTimeout(t);
  }, [bootStage]);

  // Load conversations when user changes; clear stale state on switch/logout
  useEffect(() => {
    (async () => {
      // Avoid double-running during initial bootstrap transition
      if (bootStage !== 'done') return;
      // Clear UI state to avoid showing previous user's data
      setPinned([]);
      setRecent([]);
      setSelection(null);
      if (!user?.id) return;
      try {
        const items = await listConversations();
        setRecent(items);
        if (items.length) setSelection({ type: 'chat', id: items[0].id });
      } catch {}
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id, bootStage]);

  // Replace temp conversation once backend creates it (event from ChatPanel)
  useEffect(() => {
    const handler = (e: any) => {
      const { tempId, newId, title } = e?.detail || {};
      if (!tempId || !newId) return;
      setPinned((p) =>
        p.map((c) => (c.id === tempId ? { ...c, id: newId, title: title || c.title } : c))
      );
      setRecent((r) => {
        const mapped = r.map((c) =>
          c.id === tempId ? { ...c, id: newId, title: title || c.title } : c
        );
        const seen = new Set<string>();
        return mapped.filter((c) => (seen.has(c.id) ? false : (seen.add(c.id), true)));
      });
      setSelection((sel) =>
        sel?.type === 'chat' && sel.id === tempId ? { type: 'chat', id: newId } : sel
      );
    };
    window.addEventListener('aura:conversation:realized', handler as EventListener);
    return () => window.removeEventListener('aura:conversation:realized', handler as EventListener);
  }, []);

  return (
    <div className="min-h-screen bg-[#070a14] text-white relative overflow-hidden">
      {bootStage !== 'done' && (
        <div
          className={`fixed inset-0 z-[9999] flex items-center justify-center transition-opacity duration-700 ${
            bootStage === 'transition' ? 'opacity-0 pointer-events-none' : 'opacity-100'
          }`}
          aria-hidden="true"
          onTransitionEnd={() => {
            if (bootStage === 'transition') setBootStage('done');
          }}
          style={{
            pointerEvents: bootStage === 'transition' ? ('none' as const) : ('auto' as const),
            background:
              'radial-gradient(circle at 50% 20%, rgba(116,5,180,0.25) 0%, rgba(11,14,25,1) 70%)',
          }}
        >
          <div className="flex flex-col items-center gap-6 text-white">
            {/* LOGO AURA */}
            <div className="relative flex items-center justify-center w-[96px] h-[96px]">
              {/* Glow animado detrás — con ajuste óptico */}
              <div className="absolute -top-[4px] inset-x-0 h-[96px] rounded-[28px] bg-gradient-to-br from-[#CA5CF5]/25 to-[#7405B4]/25 blur-[42px] animate-pulse" />

              {/* Contenedor del logo */}
              <div className="relative w-16 h-16 rounded-2xl bg-gradient-to-br from-[#CA5CF5] to-[#7405B4] flex items-center justify-center shadow-[0_0_50px_rgba(116,5,180,0.45)]">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 48 52"
                  fill="none"
                  className="w-8 h-8 drop-shadow-[0_0_6px_rgba(0,0,0,0.3)]"
                >
                  <path
                    d="M15.5812 25.2787V33.705H41.0345V0H7.0967V16.8525H15.5812V8.42624H32.5501V25.2787H15.5812Z"
                    fill="url(#paint0_linear)"
                  />
                  <defs>
                    <linearGradient
                      id="paint0_linear"
                      x1="0"
                      y1="0"
                      x2="48.9622"
                      y2="0.922365"
                      gradientUnits="userSpaceOnUse"
                    >
                      <stop stopColor="#E19AFB" />
                      <stop offset="1" stopColor="#8B3DFF" />
                    </linearGradient>
                  </defs>
                </svg>
              </div>
            </div>

            {/* TEXTO */}
            <div className="text-center space-y-2">
              <p className="text-[15px] font-medium text-white/90 tracking-wide animate-fade-in">
                Cargando tu espacio de chat…
              </p>
              <p className="text-[13px] text-white/40 animate-fade-in-delay">
                ✦ Aura está preparando tu entorno personal ✦
              </p>
            </div>

            {/* SPINNER */}
            <div className="w-10 h-10 border-4 border-[#ffffff1a] border-t-[#CA5CF5] rounded-full animate-spin shadow-[0_0_20px_rgba(202,92,245,0.3)]" />
          </div>
        </div>
      )}

      <div
        className={`transition-opacity duration-500 ${bootStage !== 'loading' ? 'opacity-100' : 'opacity-0'}`}
      >
        <IconRail
          active={activeRail}
          onSelect={(key) => {
            setActiveRail(key);
            if (key === 'telemetry') {
              setSelection({ type: 'telemetry', view: 'overview' });
              setConversationsOpen(false);
            } else if (key === 'chats') {
              setConversationsOpen(true);
            } else if (key === 'group') {
              setConversationsOpen(true);
            } else {
              setConversationsOpen(false);
            }
          }}
          onToggleTheme={handleToggleTheme}
          avatarUrl={user?.avatar_url || undefined}
          userName={user?.name || 'Tu perfil'}
          onProfile={() => {
            setSelection({ type: 'profile' });
            setConversationsOpen(false);
          }}
          onSettings={() => navigate('/settings')}
          onChangePassword={() => navigate('/change-password')}
          onLogout={handleLogout}
        />

        {activeRail === 'chats' && conversationsOpen && (
          <ConversationsPanel
            railWidth={RAIL_W}
            pinned={pinned}
            recent={recent}
            selectedId={selection?.type === 'chat' ? selection.id : undefined}
            onSelect={(id) => setSelection({ type: 'chat', id })}
            onCreate={handleCreateChat}
            onSearch={handleSearchChat}
            onTogglePin={handleTogglePin}
            onDelete={handleDeleteChat}
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

        <main
          style={{ paddingLeft: mainPaddingLeft }}
          className="pt-6 pr-6 pb-6 h-screen overflow-hidden"
        >
          {selection?.type === 'profile' ? (
            <div className="w-full h-full">
              <ProfilePanel
                name={user?.name || ''}
                email={user?.email || ''}
                id={user?.id || ''}
                avatarUrl={user?.avatar_url || undefined}
              />
            </div>
          ) : activeChat ? (
            <ChatPanel
              conversationId={activeChat.id}
              onTitleChange={handleChatTitleChange}
              userName={user?.name || 'Tú'}
              userAvatar={user?.avatar_url || '/images/avatar_demo.jpg'}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <MainPanel selection={selection} />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
