import React, { useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { logout as apiLogout } from '@/lib/auth';

/** Tipos opcionales para tus listas (mock por ahora) */
export type ChatItem = { id: string; title: string };

type SidebarProps = {
  isOpen: boolean;
  onToggle: () => void;
  saved: ChatItem[];
  recent: ChatItem[];
  user: { name: string; avatarUrl?: string };
  onNewChat?: () => void;
  onSearch?: () => void;
  onOpenSaved?: (id: string) => void;
  onOpenRecent?: (id: string) => void;
  onProfile?: () => void;
  onSettings?: () => void;
  onHelp?: () => void;
  onTelemetry?: () => void;
  onLogout?: () => void;
};

const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onToggle,
  saved,
  recent,
  user,
  onNewChat,
  onSearch,
  onOpenSaved,
  onOpenRecent,
  onProfile,
  onSettings,
  onHelp,
  onTelemetry,
  onLogout,
}) => {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const handleProfile = onProfile ?? (() => navigate('/profile'));
  const handleSettings = onSettings ?? (() => navigate('/settings'));
  const handleHelp = onHelp ?? (() => navigate('/help'));
  const handleTelemetry = onTelemetry ?? (() => navigate('/telemetry'));
  const handleLogout =
    onLogout ??
    (async () => {
      try {
        await apiLogout();
      } catch {}
      navigate('/login');
    });

  // Cerrar dropdown al hacer click afuera o presionar Esc
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (!menuRef.current) return;
      if (!menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMenuOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, []);

  return (
    <>
      {/* Sidebar */}
      <aside
        className={`
          fixed md:static z-30 top-0 left-0 h-full md:h-auto
          w-[86vw] max-w-[476px] md:w-[250px]
          bg-[#151515]/95 backdrop-blur
          border-r md:border-r border-white/10
          text-white
          transition-transform duration-300
          ${isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
          md:rounded-none
          rounded-tr-[36px] rounded-br-[36px] md:rounded-tr-none md:rounded-br-none
        `}
        aria-label="Sidebar de navegaciÃ³n"
      >
        {/* Header con logo + botÃ³n de colapso a la derecha */}
        <div className="h-[80px] flex items-center justify-between px-5">
          <div className="flex items-center gap-3">
            <img src="/images/logo_aura.svg" alt="Aura" className="w-[105px] h-auto" />
          </div>

          <button
            onClick={onToggle}
            className="grid place-items-center w-[34px] h-[34px] rounded-lg bg-white/5 hover:bg-white/10 ring-1 ring-white/10 transition"
            title={isOpen ? 'Ocultar panel' : 'Mostrar panel'}
          >
            {isOpen ? <ChevronLeftIcon className="w-5 h-5" /> : <MenuIcon className="w-5 h-5" />}
          </button>
        </div>

        <div className="mx-5 h-px bg-white/10" />

        {/* NavegaciÃ³n */}
        <nav className="px-2 py-4 text-sm">
          <div className="space-y-2 px-2">
            <SidebarItem
              icon={
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 20 20"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M10.0141 0C8.7027 0 7.40417 0.258646 6.19262 0.761171C4.98108 1.2637 3.88024 2.00026 2.95296 2.9288C1.08023 4.80408 0.0281467 7.34751 0.0281467 9.99956C0.0194168 12.3086 0.817826 14.5479 2.28497 16.3293L0.287781 18.3292C0.149218 18.4698 0.0553545 18.6484 0.0180329 18.8424C-0.0192886 19.0363 0.00160464 19.237 0.0780763 19.4191C0.161017 19.5991 0.295474 19.7502 0.464379 19.8535C0.633283 19.9568 0.829019 20.0075 1.02674 19.9991H10.0141C12.6625 19.9991 15.2025 18.9456 17.0752 17.0703C18.9479 15.195 20 12.6516 20 9.99956C20 7.34751 18.9479 4.80408 17.0752 2.9288C15.2025 1.05352 12.6625 0 10.0141 0ZM10.0141 17.9992H3.43335L4.36204 17.0692C4.54803 16.8819 4.65242 16.6285 4.65242 16.3643C4.65242 16.1001 4.54803 15.8467 4.36204 15.6593C3.05447 14.3514 2.2402 12.63 2.05798 10.7883C1.87575 8.94665 2.33684 7.09869 3.36267 5.55928C4.38851 4.01986 5.91564 2.88423 7.68387 2.34587C9.4521 1.80751 11.352 1.89972 13.06 2.6068C14.7679 3.31388 16.1782 4.59207 17.0506 6.22362C17.923 7.85516 18.2034 9.73911 17.8442 11.5545C17.4849 13.3699 16.5082 15.0044 15.0805 16.1795C13.6527 17.3547 11.8622 17.9977 10.0141 17.9992ZM13.0099 8.9996H11.0127V6.99969C11.0127 6.73449 10.9075 6.48014 10.7202 6.29261C10.5329 6.10509 10.2789 5.99973 10.0141 5.99973C9.74923 5.99973 9.49523 6.10509 9.30796 6.29261C9.12069 6.48014 9.01548 6.73449 9.01548 6.99969V8.9996H7.01829C6.75345 8.9996 6.49946 9.10495 6.31218 9.29248C6.12491 9.48001 6.0197 9.73435 6.0197 9.99956C6.0197 10.2648 6.12491 10.5191 6.31218 10.7066C6.49946 10.8942 6.75345 10.9995 7.01829 10.9995H9.01548V12.9994C9.01548 13.2646 9.12069 13.519 9.30796 13.7065C9.49523 13.894 9.74923 13.9994 10.0141 13.9994C10.2789 13.9994 10.5329 13.894 10.7202 13.7065C10.9075 13.519 11.0127 13.2646 11.0127 12.9994V10.9995H13.0099C13.2747 10.9995 13.5287 10.8942 13.716 10.7066C13.9032 10.5191 14.0084 10.2648 14.0084 9.99956C14.0084 9.73435 13.9032 9.48001 13.716 9.29248C13.5287 9.10495 13.2747 8.9996 13.0099 8.9996Z"
                    fill="url(#paint0_linear_36_53)"
                  />
                  <defs>
                    <linearGradient
                      id="paint0_linear_36_53"
                      x1="0"
                      y1="0"
                      x2="20.3997"
                      y2="0.41632"
                      gradientUnits="userSpaceOnUse"
                    >
                      <stop stopColor="#CA5CF5" />
                      <stop offset="1" stopColor="#7405B4" />
                    </linearGradient>
                  </defs>
                </svg>
              }
              label={
                <span className="bg-[linear-gradient(180deg,#b77cff_0%,#7B2FE3_100%)] bg-clip-text text-transparent select-none">
                  Iniciar nuevo chat
                </span>
              }
              onClick={onNewChat}
              active
            />
            <SidebarItem icon={<SearchLgIcon />} label="Buscar Chat" onClick={onSearch} />
            <SidebarItem icon={<BookmarkLgIcon />} label="Chats guardados" />
          </div>

          {/* Guardados */}
          {saved?.length > 0 && (
            <>
              <SectionTitle>Guardados</SectionTitle>
              <div className="space-y-1.5">
                {saved.map((c) => (
                  <Row key={c.id} onClick={() => onOpenSaved?.(c.id)} label={c.title} />
                ))}
              </div>
            </>
          )}

          {/* Recientes */}
          {recent?.length > 0 && (
            <>
              <SectionTitle className="mt-5">Recientes</SectionTitle>
              <div className="space-y-1.5">
                {recent.map((c) => (
                  <Row key={c.id} onClick={() => onOpenRecent?.(c.id)} label={c.title} />
                ))}
              </div>
            </>
          )}
        </nav>

        {/* Footer usuario */}
        <div className="mt-auto">
          <div className="mx-5 h-px bg-white/10" />
          <div className="p-4">
            <button
              onClick={() => setMenuOpen((s) => !s)}
              className="w-full flex items-center gap-3 px-3 py-3 rounded-xl hover:bg-white/5 transition text-left"
            >
              <Avatar src={user.avatarUrl} name={user.name} />
              <span className="truncate text-[16px] text-white/90">{user.name}</span>
            </button>

            {/* Dropdown perfil */}
            <div ref={menuRef} className={`relative ${menuOpen ? 'block' : 'hidden'}`}>
              <div
                className="absolute bottom-16 left-3 w-[240px] rounded-2xl border border-white/15 bg-[#1B1B1B]/95 backdrop-blur shadow-[0_10px_40px_-10px_rgba(0,0,0,0.6)] p-2"
                role="menu"
              >
                <DropdownItem icon={<UserIcon />} label="Perfil" onClick={handleProfile} />
                <DropdownItem icon={<GearIcon />} label="Configuraciï¿½n" onClick={handleSettings} />
                <div className="h-px bg-white/10 my-1" />
                <DropdownItem icon={<HelpIcon />} label="Centro de ayuda" onClick={handleHelp} />
                <DropdownItem
                  icon={<TelemetryIcon />}
                  label="Telemetrï¿½a"
                  onClick={handleTelemetry}
                />
                <DropdownItem icon={<LogoutIcon />} label="Cerrar sesiï¿½n" onClick={handleLogout} />
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* Capa para cerrar en mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/40 backdrop-blur-sm md:hidden"
          onClick={onToggle}
        />
      )}
    </>
  );
};

export default Sidebar;

/* ================= Subcomponentes ================= */

function SidebarItem({
  icon,
  label,
  onClick,
  active = false,
}: {
  icon: React.ReactNode;
  label: React.ReactNode;
  onClick?: () => void;
  active?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      className={`
        w-full flex items-center gap-3
        px-4 py-3 rounded-xl transition
        ${active ? 'bg-white/10 ring-1 ring-white/15' : 'hover:bg-white/5'}
        text-[18px] leading-[22px]
      `}
    >
      <span className="w-6 h-6 grid place-items-center">{icon}</span>
      <span className="truncate">{label}</span>
    </button>
  );
}

function SectionTitle({
  children,
  className = '',
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`px-5 mt-5 mb-2 text-[13px] tracking-wide text-white/50 ${className}`}>
      {children}
    </div>
  );
}

function Row({ label, onClick }: { label: React.ReactNode; onClick?: () => void }) {
  return (
    <button
      onClick={onClick}
      className="w-full flex items-center gap-3 px-4 py-3 mx-3 rounded-xl hover:bg-white/5 transition text-[16px] text-white/90"
    >
      <DocIcon className="w-5 h-5 text-white/70" />
      <span className="truncate">{label}</span>
    </button>
  );
}

function Avatar({ src, name }: { src?: string; name: string }) {
  if (src) {
    return <img src={src} alt={name} className="w-9 h-9 rounded-full object-cover" />;
  }
  // Iniciales
  const initials =
    name
      .split(' ')
      .map((n) => n[0]?.toUpperCase())
      .slice(0, 2)
      .join('') || '?';
  return (
    <div className="w-9 h-9 rounded-full bg-white/20 grid place-items-center text-sm font-semibold">
      {initials}
    </div>
  );
}

/* Dropdown item used in the profile menu */
function DropdownItem({
  icon,
  label,
  onClick,
}: {
  icon: React.ReactNode;
  label: React.ReactNode;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      role="menuitem"
      className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-white/5 transition text-white/90 text-sm"
    >
      <span className="w-5 grid place-items-center">{icon}</span>
      <span className="truncate">{label}</span>
    </button>
  );
}

/* ================= Ãconos (SVG inline) ================= */
function MenuIcon(props: any) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M3 6h18v2H3zm0 5h18v2H3zm0 5h18v2H3z" />
    </svg>
  );
}
function ChevronLeftIcon(props: any) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M15.41 7.41 14 6l-6 6 6 6 1.41-1.41L10.83 12z" />
    </svg>
  );
}

/* Ãconos grandes para items principales */
function PlusCircleLgIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-6 h-6 text-[#8B3DFF]" fill="currentColor">
      <path d="M12 2a10 10 0 1010 10A10 10 0 0012 2zm1 11h3v-2h-3V8h-2v3H8v2h3v3h2z" />
    </svg>
  );
}
function SearchLgIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-6 h-6 text-white/90" fill="currentColor">
      <path d="M10 3a7 7 0 105.29 12.29l4.21 4.2 1.4-1.4-4.2-4.21A7 7 0 0010 3zm0 2a5 5 0 110 10 5 5 0 010-10z" />
    </svg>
  );
}
function BookmarkLgIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-6 h-6 text-white/90" fill="currentColor">
      <path d="M6 2h12a1 1 0 011 1v18l-7-3-7 3V3a1 1 0 011-1z" />
    </svg>
  );
}

/* Ãconos auxiliares */
function DocIcon(props: any) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M6 2h8l4 4v14a2 2 0 01-2 2H6a2 2 0 01-2-2V4a2 2 0 012-2zm7 1.5V8h4.5L13 3.5z" />
    </svg>
  );
}
function UserIcon(props: any) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M12 12a5 5 0 100-10 5 5 0 000 10zm-9 9a9 9 0 1118 0H3z" />
    </svg>
  );
}
function GearIcon(props: any) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M19.14 12.94a7.49 7.49 0 000-1.88l2.03-1.58a.5.5 0 00.12-.64l-1.92-3.32a.5.5 0 00-.6-.22l-2.39.96a7.48 7.48 0 00-1.62-.94l-.36-2.54a.5.5 0 00-.5-.42h-3.84a.5.5 0 00-.5.42l-.36 2.54a7.48 7.48 0 00-1.62.94l-2.39-.96a.5.5 0 00-.6.22L2.7 8.84a.5.5 0 00.12.64l2.03 1.58a7.49 7.49 0 000 1.88L2.82 14.5a.5.5 0 00-.12.64l1.92 3.32a.5.5 0 00.6.22l2.39-.96c.5.38 1.05.69 1.62.94l.36 2.54a.5.5 0 00.5.42h3.84a.5.5 0 00.5-.42l.36-2.54c.57-.25 1.12-.56 1.62-.94l2.39.96a.5.5 0 00.6-.22l1.92-3.32a.5.5 0 00-.12-.64l-2.03-1.56zM12 15.5A3.5 3.5 0 1115.5 12 3.5 3.5 0 0112 15.5z" />
    </svg>
  );
}
function HelpIcon(props: any) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M12 2a10 10 0 1010 10A10 10 0 0012 2zm1 15h-2v-2h2zm2.07-7.75l-.9.92A3.49 3.49 0 0013 13h-2v-.5a4.49 4.49 0 011.32-3.18l1.24-1.26a1.75 1.75 0 10-2.48-2.48 1.74 1.74 0 00-.51 1.24H8a3.75 3.75 0 116.07 2.5z" />
    </svg>
  );
}
function TelemetryIcon(props: any) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M3 3h18v2H3zM7 9h2v12H7zm4-4h2v16h-2zm4 8h2v8h-2z" />
    </svg>
  );
}
function LogoutIcon(props: any) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M16 13v-2H7V8l-5 4 5 4v-3zM20 3h-8v2h8v14h-8v2h8a2 2 0 002-2V5a2 2 0 00-2-2z" />
    </svg>
  );
}
