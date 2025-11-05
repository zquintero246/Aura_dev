import React, { useEffect, useState } from 'react';
import { getUserHome } from '../../lib/homeApi';
import { me } from '../../lib/auth';
import HomeRegistration from './HomeRegistration';

export default function HomeAssistantPanel() {
  const [needsRegistration, setNeedsRegistration] = useState<boolean | null>(null);
  const [userId, setUserId] = useState<string>('');

  useEffect(() => {
    (async () => {
      try {
        const res = await me();
        const id = res?.user?.id ?? '';
        setUserId(String(id));
        let required = true;
        if (id) {
          const home = await getUserHome(String(id));
          required = !home;
        } else {
          // fallback local
          required = localStorage.getItem('home:registered') !== '1';
        }
        setNeedsRegistration(required);
      } catch {
        setNeedsRegistration(localStorage.getItem('home:registered') !== '1');
      }
    })();
  }, []);

  if (needsRegistration === null) {
    return (
      <div className="w-full h-full grid place-items-center text-white/60">Cargando…</div>
    );
  }

  if (needsRegistration) {
    return <HomeRegistration userId={userId} onDone={() => setNeedsRegistration(false)} />;
  }

  return (
    <div className="w-full h-full flex flex-col select-none">
      <header className="px-2 md:px-4">
        <div className="mt-1 h-12 md:h-14 rounded-2xl bg-white/5 ring-1 ring-white/10 flex items-center justify-between px-4 text-white/80" role="region" aria-label="Resumen del hogar">
          <div className="font-medium tracking-wide">Mi casa</div>
          <div className="flex items-center gap-3">
            <span className="hidden sm:inline text-sm text-white/70">Bucaramanga</span>
            <span className="w-6 h-6 rounded-full bg-white/10" aria-hidden />
            <span className="w-6 h-6 rounded-full bg-white/10" aria-hidden />
            <span className="w-6 h-6 rounded-full bg-white/10" aria-hidden />
          </div>
        </div>
      </header>
      <section className="flex-1 min-h-0 px-2 md:px-4 py-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-[160px] md:h-[220px] rounded-2xl bg-white/5 ring-1 ring-white/10" />
          ))}
        </div>
      </section>
      <footer className="px-2 md:px-4 pb-3">
        <div className="relative mx-auto max-w-[820px]">
          <div className="h-10 rounded-full bg-gradient-to-r from-white/5 via-white/8 to-white/5 ring-1 ring-white/10" />
          <button className="absolute left-1/2 -translate-x-1/2 -top-4 w-12 h-12 rounded-full grid place-items-center bg-[#8B3DFF] text-white shadow-[0_10px_40px_-10px_rgba(139,61,255,0.7)] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#8B3DFF]/60 focus:ring-offset-[#070a14]" aria-label="Acción principal">
            <SparkleIcon />
          </button>
        </div>
      </footer>
    </div>
  );
}

function SparkleIcon() {
  return (
    <img
      src="/images/img_1.svg"
      alt="Arrow icon"
      className="w-[10px] sm:w-[14px] md:w-[17px] lg:w-[20px] h-[10px] sm:h-[14px] md:h-[17px] lg:h-[20px] absolute left-[2px] sm:left-[3px] md:left-[3px] lg:left-spacing-xs"
    />
  );
}
