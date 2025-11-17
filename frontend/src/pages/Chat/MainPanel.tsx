import React from 'react';
import ChatPanel from './ChatPanel';
import HomeAssistantPanel from './HomeAssistantPanel';

type Selection =
  | { type: 'chat'; id: string }
  | { type: 'group'; id: string }
  | { type: 'project'; id: string }
  | { type: 'telemetry'; view: 'overview' | 'errors' | 'latency' }
  | null;

export default function MainPanel({ selection }: { selection: Selection }) {
  if (!selection) {
    // Estado vacío visual estilo Aura
    return (
      <div className="h-[75vh] flex flex-col items-center justify-center text-center select-none">
        <h1 className="text-[22px] md:text-[26px] font-semibold bg-gradient-to-r from-[#CA5CF5] to-[#7405B4] bg-clip-text text-transparent mb-2">
          ¿Cómo empezamos?
        </h1>
        <p className="text-[15px] text-white/60">
          Selecciona un chat, un grupo o un proyecto desde el panel izquierdo para empezar
        </p>
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
    return <HomeAssistantPanel />;
  }

  return null;
}

/* ---- Ejemplos de vistas (mock) ---- */

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
