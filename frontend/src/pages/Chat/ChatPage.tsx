// ChatPage.tsx
import React, { useState } from 'react';
import ChatPanel from './ChatPanel';
import ConversationsPanel, { Conversation } from './ConversationsPanel';

export default function ChatPage() {
  // Estado global de las conversaciones
  const [pinned, setPinned] = useState<Conversation[]>([
    { id: 'c1', title: 'Conversación de ejemplo', pinned: true },
  ]);
  const [recent, setRecent] = useState<Conversation[]>([{ id: 'c2', title: 'Otra conversación' }]);

  const [selectedId, setSelectedId] = useState<string>('c1');

  // Buscar la conversación activa (pinned o reciente)
  const activeConv =
    pinned.find((c) => c.id === selectedId) || recent.find((c) => c.id === selectedId);

  // Función para actualizar el título desde el ChatPanel
  const handleTitleChange = (id: string, newTitle: string) => {
    setPinned((prev) => prev.map((c) => (c.id === id ? { ...c, title: newTitle } : c)));
    setRecent((prev) => prev.map((c) => (c.id === id ? { ...c, title: newTitle } : c)));
  };

  // Crear una nueva conversación
  const handleCreate = () => {
    const newId = crypto.randomUUID();
    const newConv = { id: newId, title: 'Nueva conversación' };
    setRecent((r) => [newConv, ...r]);
    setSelectedId(newId);
  };

  return (
    <div className="h-screen flex">
      {/* --- ICON RAIL --- */}
      <div className="w-[72px] bg-[#05080f] border-r border-white/10" />

      {/* --- PANEL IZQUIERDO (conversaciones) --- */}
      <ConversationsPanel
        pinned={pinned}
        recent={recent}
        selectedId={selectedId}
        onSelect={setSelectedId}
        onCreate={handleCreate}
        railWidth={72}
      />

      {/* --- PANEL PRINCIPAL (chat) --- */}
      {activeConv && (
        <main className="flex-1 bg-[#070a14]">
          <ChatPanel
            conversationId={activeConv.id}
            onTitleChange={handleTitleChange} // 🔥 conexión
          />
        </main>
      )}
    </div>
  );
}
