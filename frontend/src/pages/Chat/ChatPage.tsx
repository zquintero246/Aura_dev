// ChatPage.tsx
import React, { useEffect, useState } from 'react';
import ChatPanel from './ChatPanel';
import ConversationsPanel, { Conversation } from './ConversationsPanel';
import { listConversations } from '../../lib/conversations';

export default function ChatPage() {
  // Estado global de las conversaciones
  const [pinned, setPinned] = useState<Conversation[]>([]);
  const [recent, setRecent] = useState<Conversation[]>([]);

  const [selectedId, setSelectedId] = useState<string>('');

  // Buscar la conversación activa (pinned o reciente)
  const activeConv =
    pinned.find((c) => c.id === selectedId) || recent.find((c) => c.id === selectedId);

  // Función para actualizar el título desde el ChatPanel
  const handleTitleChange = (id: string, newTitle: string) => {
    setPinned((prev) => prev.map((c) => (c.id === id ? { ...c, title: newTitle } : c)));
    setRecent((prev) => prev.map((c) => (c.id === id ? { ...c, title: newTitle } : c)));
  };

  const handleTogglePin = (id: string, nextPinned: boolean) => {
    if (nextPinned) {
      if (pinned.length >= 3) {
        try { window.alert('Solo puedes anclar hasta 3 conversaciones'); } catch {}
        return;
      }
      // mover de recientes -> anclados
      const item = recent.find((c) => c.id === id);
      if (item && !pinned.find((p) => p.id === id)) {
        setRecent((r) => r.filter((c) => c.id !== id));
        setPinned((p) => [...p, { ...item, pinned: true }]);
      }
    } else {
      // desanclar -> regresar a recientes (al inicio)
      const item = pinned.find((c) => c.id === id);
      if (item) {
        setPinned((p) => p.filter((c) => c.id !== id));
        setRecent((r) => [{ ...item, pinned: false }, ...r.filter((c) => c.id !== id)]);
      }
    }
  };

  // Crear una nueva conversación (persistente en backend Mongo)
  const handleCreate = () => {
    const tempId = `tmp-${crypto.randomUUID()}`;
    const tempConv = { id: tempId, title: 'Nueva conversación' } as Conversation;
    setRecent((r) => [tempConv, ...r]);
    setSelectedId(tempId);
  };

  // Visibilidad del panel de conversaciones (demo)
  const [conversationsOpen, setConversationsOpen] = useState(true);
  useEffect(() => {
    const handler = () => setConversationsOpen(false);
    window.addEventListener('aura:conversations:close', handler as EventListener);
    return () => {
      window.removeEventListener('aura:conversations:close', handler as EventListener);
    };
  }, []);

  // Cargar conversaciones del usuario desde backend (Mongo)
  useEffect(() => {
    (async () => {
      try {
        const items = await listConversations();
        setRecent(items);
        if (items.length && !selectedId) setSelectedId(items[0].id);
      } catch {
        // ignore; quedará vacío hasta que se cree una
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reemplazar conversación temporal cuando el ChatPanel la cree en backend
  useEffect(() => {
    const handler = (e: any) => {
      const { tempId, newId, title } = e?.detail || {};
      if (!tempId || !newId) return;
      setPinned((p) => p.map((c) => (c.id === tempId ? { ...c, id: newId, title: title || c.title } : c)));
      setRecent((r) => {
        const mapped = r.map((c) => (c.id === tempId ? { ...c, id: newId, title: title || c.title } : c));
        return mapped.some((c) => c.id === newId) ? mapped.filter((c, i, arr) => arr.findIndex(x => x.id === c.id) === i) : mapped;
      });
      setSelectedId((id) => (id === tempId ? newId : id));
    };
    window.addEventListener('aura:conversation:realized', handler as EventListener);
    return () => window.removeEventListener('aura:conversation:realized', handler as EventListener);
  }, []);

  return (
    <div className="h-screen flex">
      {/* --- ICON RAIL --- */}
      <div className="w-[72px] bg-[#05080f] border-r border-white/10" />

      {/* --- PANEL IZQUIERDO (conversaciones) --- */}
      {conversationsOpen && (
        <ConversationsPanel
          pinned={pinned}
          recent={recent}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onCreate={handleCreate}
          railWidth={72}
          onTogglePin={handleTogglePin}
        />
      )}

      {/* --- PANEL PRINCIPAL (chat) --- */}
      {activeConv && (
        <main className="flex-1 bg-[#070a14]" style={{ paddingLeft: conversationsOpen ? 320 : 0 }}>
          <ChatPanel conversationId={activeConv.id} onTitleChange={handleTitleChange} />
        </main>
      )}
    </div>
  );
}
