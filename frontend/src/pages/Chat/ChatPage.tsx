// ChatPage.tsx
import React, { useEffect, useState } from 'react';
import ChatPanel from './ChatPanel';
import ConversationsPanel, { Conversation } from './ConversationsPanel';
import { listConversations } from '../../lib/conversations';

export default function ChatPage() {
  const [recent, setRecent] = useState<Conversation[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');

  const activeConv = recent.find((c) => c.id === selectedId);

  const handleTitleChange = (id: string, newTitle: string) => {
    setRecent((prev) => prev.map((c) => (c.id === id ? { ...c, title: newTitle } : c)));
  };

  const handleCreate = () => {
    const tempId = `tmp-${crypto.randomUUID()}`;
    const tempConv = { id: tempId, title: 'Nueva conversación' } as Conversation;
    setRecent((r) => [tempConv, ...r]);
    setSelectedId(tempId);
  };

  const [conversationsOpen, setConversationsOpen] = useState(true);
  useEffect(() => {
    const handler = () => setConversationsOpen(false);
    window.addEventListener('aura:conversations:close', handler as EventListener);
    return () => window.removeEventListener('aura:conversations:close', handler as EventListener);
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const items = await listConversations();
        setRecent(items);
        if (items.length && !selectedId) setSelectedId(items[0].id);
      } catch {
        // ignore
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const handler = (e: any) => {
      const { tempId, newId, title } = e?.detail || {};
      if (!tempId || !newId) return;
      setRecent((r) => {
        const mapped = r.map((c) => (c.id === tempId ? { ...c, id: newId, title: title || c.title } : c));
        return mapped.some((c) => c.id === newId) ? mapped.filter((c, i, arr) => arr.findIndex((x) => x.id === c.id) === i) : mapped;
      });
      setSelectedId((id) => (id === tempId ? newId : id));
    };
    window.addEventListener('aura:conversation:realized', handler as EventListener);
    return () => window.removeEventListener('aura:conversation:realized', handler as EventListener);
  }, []);

  return (
    <div className="h-screen flex">
      <div className="w-[72px] bg-[#05080f] border-r border-white/10" />

      {conversationsOpen && (
        <ConversationsPanel
          recent={recent}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onCreate={handleCreate}
          railWidth={72}
        />
      )}

      {activeConv && (
        <main className="flex-1 bg-[#070a14]" style={{ paddingLeft: conversationsOpen ? 320 : 0 }}>
          <ChatPanel conversationId={activeConv.id} onTitleChange={handleTitleChange} />
        </main>
      )}
    </div>
  );
}
