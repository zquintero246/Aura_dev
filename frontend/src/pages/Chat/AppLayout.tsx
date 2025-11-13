// AppLayout.tsx
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import IconRail from './IconRail';
import ConversationsPanel, { Conversation } from './ConversationsPanel';
import GroupsPanel, { Group } from './GroupsPanel';
import MainPanel from './MainPanel';
import ProfilePanel from '../Account/Profile';
import ChatPanel from './ChatPanel';
import { me, logout, User, ensureChatToken } from '../../lib/auth';
import { setChatToken } from '../../lib/chatApi';
import { useNavigate } from 'react-router-dom';
import {
  listConversations,
  deleteConversation as deleteChat,
  createConversation as createConversationOnServer,
  updateConversationTitle as updateConversationTitleOnServer,
} from '../../lib/conversations';

type SectionKey = 'chats' | 'group' | 'project' | 'telemetry';

type Selection =
  | { type: 'chat'; id: string }
  | { type: 'group'; id: string }
  | { type: 'project'; id: string }
  | { type: 'telemetry'; view: 'overview' | 'errors' | 'latency' }
  | { type: 'profile' }
  | null;

type TitlePromptMode = 'create' | 'rename';
type TitlePromptState =
  | { mode: 'create'; title: string }
  | { mode: 'rename'; title: string; targetId: string };

const RAIL_W = 72;
const SIDE_W = 320;

export default function AppLayout() {
  const navigate = useNavigate();
  const [activeRail, setActiveRail] = useState<SectionKey>('chats');
  const [conversationsOpen, setConversationsOpen] = useState(true);
  const [selection, setSelection] = useState<Selection>(null);
  const [bootStage, setBootStage] = useState<'loading' | 'transition' | 'done'>('loading');

  // Conversations state (no demos)
  const [recent, setRecent] = useState<Conversation[]>([]);
  const [deleteModal, setDeleteModal] = useState<{ id: string; title?: string } | null>(null);
  const [titlePromptState, setTitlePromptState] = useState<TitlePromptState | null>(null);
  const [titlePromptBusy, setTitlePromptBusy] = useState(false);

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

  // Create conversation (shows prompt before persisting)
  const handleCreateChat = () => {
    setTitlePromptState({ mode: 'create', title: '' });
  };

  const [messageSearchQuery, setMessageSearchQuery] = useState('');
  const handleSearchChat = (query: string) => {
    setMessageSearchQuery(query);
  };

  const handleRequestRename = (id: string) => {
    if (titlePromptBusy) return;
    const candidate = recent.find((c) => c.id === id);
    setTitlePromptState({
      mode: 'rename',
      targetId: id,
      title: candidate?.title || '',
    });
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
    setRecent((prev) => prev.map((c) => (c.id === id ? { ...c, title: normalized } : c)));
  };

  const updateTitlePromptValue = (value: string) => {
    setTitlePromptState((prev) => (prev ? { ...prev, title: value } : prev));
  };

  const closeTitlePrompt = () => {
    if (titlePromptBusy) return;
    setTitlePromptState(null);
  };

  const handleTitlePromptConfirm = async () => {
    if (!titlePromptState || titlePromptBusy) return;
    const raw = titlePromptState.title || '';
    let trimmed = raw.trim();
    if (!trimmed) {
      if (titlePromptState.mode === 'create') {
        trimmed = 'Nueva conversación';
      } else {
        try {
          window.alert('Escribe un nombre válido para la conversación.');
        } catch {}
        return;
      }
    }

    setTitlePromptBusy(true);
    let success = false;
    try {
      if (titlePromptState.mode === 'create') {
        const conv = await createConversationOnServer(trimmed);
        setRecent((prev) => {
          const filtered = prev.filter((c) => c.id !== conv.id);
          return [conv, ...filtered];
        });
        setSelection({ type: 'chat', id: conv.id });
        success = true;
      } else if (titlePromptState.mode === 'rename' && titlePromptState.targetId) {
        await updateConversationTitleOnServer(titlePromptState.targetId, trimmed);
        handleChatTitleChange(titlePromptState.targetId, trimmed);
        success = true;
      }
    } catch (err) {
      console.error('Failed to save conversation title', err);
      try {
        window.alert('No se pudo guardar el título. Intenta nuevamente.');
      } catch {}
    } finally {
      setTitlePromptBusy(false);
      if (success) setTitlePromptState(null);
    }
  };

  const activeChat =
    (selection?.type === 'chat' &&
      recent.find((c) => c.id === selection.id)) ||
    null;

  const removeConversationLocally = useCallback(
    (id: string) => {
      if (!id) return;
      const wasActive = selection?.type === 'chat' && selection.id === id;
      setRecent((r) => {
        const updated = r.filter((c) => c.id !== id);
        if (wasActive) {
          const next = updated[0] || null;
          setSelection(next ? { type: 'chat', id: next.id } : null);
        }
        return updated;
      });
      if (!wasActive) {
        // Ensure selection doesn't point to deleted ID when not active
        setSelection((sel) => (sel?.type === 'chat' && sel.id === id ? null : sel));
      }
    },
    [selection]
  );

  const handleDeleteChat = (id: string) => {
    if (!id) return;
    const candidate = recent.find((c) => c.id === id);
    setDeleteModal({ id, title: candidate?.title });
  };

  const executeDeleteChat = async () => {
    if (!deleteModal?.id) return;
    const id = deleteModal.id;
    setDeleteModal(null);
    removeConversationLocally(id);
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

  useEffect(() => {
    const handler = (e: any) => {
      const id = e?.detail?.id;
      if (!id) return;
      removeConversationLocally(id);
    };
    window.addEventListener('aura:conversation:deleted', handler as EventListener);
    return () => window.removeEventListener('aura:conversation:deleted', handler as EventListener);
  }, [removeConversationLocally]);

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
          onLogout={handleLogout}
        />

        {activeRail === 'chats' && conversationsOpen && (
          <ConversationsPanel
            railWidth={RAIL_W}
            recent={recent}
            selectedId={selection?.type === 'chat' ? selection.id : undefined}
            onSelect={(id) => setSelection({ type: 'chat', id })}
            onCreate={handleCreateChat}
            onSearch={handleSearchChat}
            onDelete={handleDeleteChat}
            onMenuRename={handleRequestRename}
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
          className="h-screen overflow-hidden"
        >
          {selection?.type === 'profile' ? (
            <div className="w-full h-full">
              <ProfilePanel
                name={user?.name || ''}
                email={user?.email || ''}
                id={user?.id || ''}
                avatarUrl={user?.avatar_url || undefined}
                onProfileChange={(updated) => setUser(updated)}
              />
            </div>
          ) : activeChat ? (
          <ChatPanel
            conversationId={activeChat.id}
            conversation={activeChat}
            onTitleChange={handleChatTitleChange}
            conversationTitle={activeChat.title}
            userName={user?.name || 'Tú'}
            userAvatar={user?.avatar_url || '/images/avatar_demo.jpg'}
            searchQuery={messageSearchQuery}
            onClearSearch={() => setMessageSearchQuery('')}
          />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <MainPanel selection={selection} />
            </div>
          )}
        </main>
      </div>
      <TitlePromptOverlay
        state={titlePromptState}
        busy={titlePromptBusy}
        onChange={updateTitlePromptValue}
        onConfirm={handleTitlePromptConfirm}
        onCancel={closeTitlePrompt}
      />
      {deleteModal && (
        <div className="fixed inset-0 z-[10000] flex items-center justify-center px-4">
          <div
            className="absolute inset-0 bg-black/70 backdrop-blur-sm"
            onClick={() => setDeleteModal(null)}
          />
          <div className="relative w-full max-w-md rounded-[28px] border border-white/15 bg-[#0b101d] p-6 text-white shadow-[0_30px_70px_rgba(0,0,0,0.8)]">
            <p className="text-[11px] uppercase tracking-[0.35em] text-white/50">Confirmar</p>
            <h3 className="mt-2 text-2xl font-semibold">¿Borrar esta conversación?</h3>
            <p className="mt-1 text-sm text-white/60">Esta acción no se puede deshacer.</p>
            {deleteModal.title && (
              <p className="mt-2 text-sm text-white/40 italic">“{deleteModal.title}”</p>
            )}
            <div className="mt-6 flex justify-end gap-3 text-sm">
              <button
                type="button"
                onClick={() => setDeleteModal(null)}
                className="rounded-full border border-white/25 px-4 py-2 text-white/80 transition hover:border-white/40 hover:text-white"
              >
                Cancelar
              </button>
              <button
                type="button"
                onClick={executeDeleteChat}
                className="rounded-full bg-gradient-to-r from-[#8B3DFF] to-[#6F6AF0] px-4 py-2 font-semibold text-[#05040b] shadow-[0_12px_30px_rgba(127,157,255,0.45)] transition hover:opacity-90"
              >
                Aceptar
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

type TitlePromptOverlayProps = {
  state: TitlePromptState | null;
  busy: boolean;
  onChange: (value: string) => void;
  onConfirm: () => void;
  onCancel: () => void;
};

function TitlePromptOverlay({
  state,
  busy,
  onChange,
  onConfirm,
  onCancel,
}: TitlePromptOverlayProps) {
  if (!state) return null;
  const handleCancel = () => {
    if (busy) return;
    onCancel();
  };
  const isRename = state.mode === 'rename';
  const confirmDisabled = busy || (isRename && !state.title.trim());
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={handleCancel} />
      <div className="relative w-full max-w-md rounded-[32px] border border-white/10 bg-[#0b1220]/90 p-6 shadow-[0_35px_120px_rgba(0,0,0,0.9)] overflow-hidden">
        <div className="absolute inset-0 -z-10 bg-gradient-to-br from-[#8B3DFF]/30 via-transparent to-[#3C6EF5]/20 opacity-80 blur-3xl animate-[pulse_3s_ease-in-out_infinite]" />
        <div className="relative flex flex-col gap-3">
          <span className="text-[11px] uppercase tracking-[0.35em] text-white/50">
            {state.mode === 'create' ? 'Nueva conversación' : 'Configuración'}
          </span>
          <h3 className="text-2xl font-semibold text-white">
            {state.mode === 'create'
              ? '¿Cómo quieres llamar a esta conversación?'
              : 'Dale un nuevo nombre'}
          </h3>
          <p className="text-sm text-white/60">
            {state.mode === 'create'
              ? 'El título que escribas se guardará en tu historial antes de empezar.'
              : 'Actualizamos el nombre en tu historial y en los tres puntos.'}
          </p>
          <div className="h-px bg-white/10" />
          <input
            autoFocus
            value={state.title}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                onConfirm();
              } else if (e.key === 'Escape') {
                e.preventDefault();
                handleCancel();
              }
            }}
            placeholder="Título de la conversación"
            className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-lg font-medium text-white placeholder:text-white/45 outline-none transition focus:border-[#7B2FE3] focus:bg-white/10"
          />
          <div className="flex gap-3 text-sm">
            <button
              type="button"
              onClick={handleCancel}
              disabled={busy}
              className="flex-1 rounded-2xl border border-white/20 px-4 py-3 text-white/70 transition hover:border-white/40 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Cancelar
            </button>
            <button
              type="button"
              onClick={onConfirm}
              disabled={confirmDisabled}
              className="flex-1 rounded-2xl bg-gradient-to-r from-[#8B3DFF] to-[#6B2FD8] px-4 py-3 font-semibold text-white shadow-[0_15px_40px_rgba(139,61,255,0.45)] transition disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {busy
                ? 'Guardando…'
                : state.mode === 'create'
                  ? 'Crear conversación'
                  : 'Guardar nombre'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
