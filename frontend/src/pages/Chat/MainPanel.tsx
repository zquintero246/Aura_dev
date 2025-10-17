// MainPanel.tsx
import React from 'react';
import ChatPanel from './ChatPanel';

type Selection =
  | { type: 'chat'; id: string }
  | { type: 'group'; id: string }
  | { type: 'project'; id: string }
  | { type: 'telemetry'; view: 'overview' | 'errors' | 'latency' }
  | null;

export default function MainPanel({ selection }: { selection: Selection }) {
  if (!selection) {
    // Estado vacío: aún no entras a nada
    return (
      <div className="min-h-[70vh] grid place-items-center">
        <div className="text-center max-w-[680px]">
          <h1 className="text-[22px] md:text-[24px] font-semibold text-white/90">
            Agregar título…
          </h1>
          <p className="mt-2 text-white/55">
            Selecciona un chat, un grupo o un proyecto desde el panel izquierdo para empezar.
          </p>
        </div>
      </div>
    );
  }

  // Distintas vistas según lo que abriste
  if (selection?.type === 'chat') {
    return (
      <ChatPanel
        conversationId={selection.id}
        userName="Santiago Arias"
        userAvatar="/images/tu_foto.jpg"
      />
    );
  }
  if (selection.type === 'group') {
    return <GroupView groupId={selection.id} />;
  }
  if (selection.type === 'project') {
    return <ProjectView projectId={selection.id} />;
  }
  if (selection.type === 'telemetry') {
    return <TelemetryView view={selection.view} />;
  }

  return null;
}

/* ---- Ejemplos de vistas (mock) ---- */

function ChatView({ conversationId }: { conversationId: string }) {
  return (
    <div className="mx-auto max-w-[980px]">
      <header className="py-3 text-white/80">Chat · {conversationId}</header>
      <div className="mt-4 h-[60vh] rounded-2xl bg-white/5 ring-1 ring-white/10" />
      <footer className="mt-6 rounded-2xl bg-white/5 ring-1 ring-white/10 h-14" />
    </div>
  );
}

function GroupView({ groupId }: { groupId: string }) {
  return (
    <div className="mx-auto max-w-[980px]">
      <header className="py-3 text-white/80">Grupo · {groupId}</header>
      <div className="mt-4 h-[60vh] rounded-2xl bg-white/5 ring-1 ring-white/10" />
      <footer className="mt-6 rounded-2xl bg-white/5 ring-1 ring-white/10 h-14" />
    </div>
  );
}

function ProjectView({ projectId }: { projectId: string }) {
  return (
    <div className="mx-auto max-w-[980px]">
      <header className="py-3 text-white/80">Proyecto · {projectId}</header>
      <div className="mt-4 h-[60vh] rounded-2xl bg-white/5 ring-1 ring-white/10" />
    </div>
  );
}

function TelemetryView({ view }: { view: 'overview' | 'errors' | 'latency' }) {
  return (
    <div className="mx-auto max-w-[1100px]">
      <header className="py-3 text-white/80">Telemetría · {view}</header>
      <div className="mt-4 grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl bg-white/5 ring-1 ring-white/10 h-40" />
        <div className="rounded-2xl bg-white/5 ring-1 ring-white/10 h-40" />
        <div className="rounded-2xl bg-white/5 ring-1 ring-white/10 h-40" />
      </div>
    </div>
  );
}
