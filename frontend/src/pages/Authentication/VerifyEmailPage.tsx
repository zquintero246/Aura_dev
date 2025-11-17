// VerifyEmailPage.tsx
// Página SPA para mostrar el estado de verificación y permitir reenviar el correo.
// Nota: El backend expone `/api/auth/email/resend` que reenvía la notificación.

import React, { useEffect, useState } from 'react';
import { Helmet } from 'react-helmet';
import Button from './components/ui/Button';
import Particles from './components/ui/Particles';
import { resendVerification, me } from '../../lib/auth';
import { useNavigate } from 'react-router-dom';

export default function VerifyEmailPage() {
  // === Estados base tuyos ===
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const navigate = useNavigate();

  // === Estados nuevos para UX mejorada ===
  const [cooldown, setCooldown] = useState(0);
  const [cooldownStep, setCooldownStep] = useState(15); // aumenta 15s, máx 60
  const [toast, setToast] = useState<{ msg: string; show: boolean }>({
    msg: '',
    show: false,
  });

  // Toast helper
  const showToast = (text: string) => {
    setToast({ msg: text, show: true });
    setTimeout(() => {
      setToast((t) => ({ ...t, show: false }));
    }, 3000);
  };

  // Calienta sesión / detecta auth
  useEffect(() => {
    (async () => {
      try {
        await me();
        setReady(true);
      } catch {
        // incluso si no hay sesión válida, igual mostramos la UI
        setReady(true);
      }
    })();
  }, []);

  // Redirección automática a /chat cuando el correo queda verificado
  useEffect(() => {
    let cancelled = false;
    let id: number | undefined;
    const check = async () => {
      try {
        const res = await me();
        if (!cancelled && res?.user?.email_verified_at) {
          navigate('/chat');
        }
      } catch (_) {
        /* ignore */
      }
    };
    // primera comprobación y luego cada 1.5s
    check();
    id = window.setInterval(check, 1500);
    return () => {
      cancelled = true;
      if (id) window.clearInterval(id);
    };
  }, [navigate]);

  // Mensaje desde la ventana de verificación (postMessage)
  useEffect(() => {
    const onMessage = (e: MessageEvent) => {
      const data: any = (e && (e as any).data) || {};
      if (
        data &&
        (data.type === 'aura-email-verified' || data.action === 'email-verified') &&
        data.status === 'ok'
      ) {
        navigate('/chat');
      }
    };
    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, [navigate]);

  // Re-comprobar al volver a la pestaña
  useEffect(() => {
    const onFocus = async () => {
      try {
        const res = await me();
        if (res?.user?.email_verified_at) navigate('/chat');
      } catch (_) {}
    };
    window.addEventListener('focus', onFocus);
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) onFocus();
    });
    return () => {
      window.removeEventListener('focus', onFocus);
      const handleVisibilityChange = () => {};
      document.removeEventListener('visibilitychange', handleVisibilityChange as any);
    };
  }, [navigate]);

  // Botón reenviar
  const handleResend = async () => {
    // bloqueos: cargando, cooldown o no listo
    if (loading || cooldown > 0 || !ready) return;

    setLoading(true);
    setMsg(null);
    setErr(null);

    try {
      const res = await resendVerification();
      const successMessage = res?.message || 'Correo de verificación reenviado';

      // guardamos en tus estados originales (por si los lees en otra parte)
      setMsg(successMessage);

      // toast bonito
      showToast(successMessage);

      // cooldown progresivo
      setCooldown(cooldownStep);
      setCooldownStep((prev) => Math.min(prev + 15, 60));
    } catch (e: any) {
      const errorMessage =
        e?.response?.data?.message || 'No se pudo reenviar el correo. Intenta de nuevo.';

      setErr(errorMessage);
      showToast(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // temporizador cooldown
  useEffect(() => {
    if (cooldown <= 0) return;
    const timer = setInterval(() => {
      setCooldown((c) => c - 1);
    }, 1000);
    return () => clearInterval(timer);
  }, [cooldown]);

  return (
    <>
      <Helmet>
        <title>Verifica tu correo</title>
      </Helmet>

      {/* Fondo gradiente + partículas (solo desktop para performance) */}
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-[#141C34] to-[#0a0a0a] -z-10 overflow-hidden hidden lg:block">
        <Particles
          particleColors={['#ffffff', '#7405B4']}
          particleCount={200}
          particleSpread={10}
          speed={0.1}
          particleBaseSize={100}
          moveParticlesOnHover={true}
          alphaParticles={false}
          disableRotation={false}
        />
      </div>

      {/* Main */}
      <main className="min-h-screen flex flex-col items-center justify-center px-4 sm:px-6 py-12 text-white bg-[#0a0a0a] lg:bg-transparent">
        {/* Logo vidrio */}
        <header className="mb-10 flex flex-col items-center">
          <div className="w-[90px] h-[90px] rounded-full bg-[rgba(29,29,29,0.25)] ring-1 ring-white/15 flex items-center justify-center backdrop-blur-md shadow-[0_0_25px_rgba(0,0,0,0.50)]">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-10 w-10 text-[#8B3DFF]"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.8}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M3 8l9 6 9-6M4 6h16a2 2 0 012 2v10a2 2 0 01-2 2H4a2 2 0 01-2-2V8a2 2 0 012-2z"
              />
            </svg>
          </div>
        </header>

        {/* Card principal */}
        <section className="w-full flex justify-center">
          <div className="relative w-full max-w-[540px]">
            {/* capa glass blur detrás */}
            <div className="absolute inset-0 rounded-[24px] bg-[rgba(29,29,29,0.25)]" />

            <div className="relative rounded-[24px] overflow-hidden border border-white/5 shadow-[0_10px_40px_-10px_rgba(0,0,0,0.5)] backdrop-blur-sm text-center px-6 sm:px-10 py-10 bg-[#1a1a1a]/30">
              <h1 className="text-[22px] sm:text-[24px] font-bold text-white">
                Verifica tu correo electrónico
              </h1>

              <p className="mt-3 text-[14px] text-[#686868] leading-relaxed max-w-[420px] mx-auto">
                Te enviamos un enlace de verificación a tu email. Por favor revisa tu bandeja de
                entrada o la carpeta de spam para continuar con el proceso de registro.
              </p>

              {/* Botón Reenviar */}
              <div className="mt-10 flex justify-center">
                <Button
                  onClick={handleResend}
                  disabled={loading || cooldown > 0 || !ready}
                  className={`w-[230px] h-11 rounded-full transition text-white ${
                    loading || cooldown > 0 || !ready
                      ? 'bg-[#5b2cb0]/60 cursor-not-allowed'
                      : 'bg-[#7B2FE3] hover:bg-[#6c29c9] active:scale-[0.98]'
                  }`}
                  border_border="transparent"
                  layout_width="auto"
                  padding="sm"
                  position="relative"
                  layout_gap="md"
                  variant="primary"
                  size="medium"
                >
                  {loading
                    ? 'Reenviando...'
                    : cooldown > 0
                      ? `Reenviar en ${cooldown}s`
                      : 'Reenviar correo de verificación'}
                </Button>
              </div>

              {/* Nota secundaria */}
              <p className="mt-5 text-[13px] text-[#686868]">
                Si no ves el mensaje, revisa tu carpeta de spam o correo no deseado.
              </p>

              {/* Mensajes de estado internos, preservados de tu versión original */}
              {msg && <p className="text-sm text-emerald-400 mt-4">{msg}</p>}

              {err && <p className="text-sm text-rose-400 mt-4">{err}</p>}

              {/* Comentario original: esta ruta `/verify-email` es completamente del frontend (React Router). */}
              {/* MANTENER (muy importante para que nadie rompa el flujo):
                 Esta ruta `/verify-email` es completamente del frontend (React Router).
                 El backend NO debe servir esta vista directamente.
                 El backend solamente envía el correo y maneja `/email/verify/{id}/{hash}`.
                 Además expone `/api/auth/email/resend` que reenvía la notificación.
              */}
            </div>
          </div>
        </section>

        {/* Toast flotante con transición */}
        <div
          aria-live="polite"
          aria-atomic="true"
          className={`fixed top-6 right-6 z-50 transition-all duration-300 ${
            toast.show
              ? 'opacity-100 translate-y-0'
              : 'opacity-0 -translate-y-2 pointer-events-none'
          }`}
        >
          <div className="flex items-start gap-3 rounded-2xl border border-white/10 bg-[#1D1D1D]/90 backdrop-blur px-4 py-3 shadow-[0_10px_40px_-10px_rgba(0,0,0,0.6)]">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5 mt-0.5 text-[#8B3DFF]"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path d="M16.707 5.293a1 1 0 00-1.414 0L8 12.586 4.707 9.293a1 1 0 10-1.414 1.414l4 4a1 1 0 001.414 0l8-8a1 1 0 000-1.414z" />
            </svg>

            <div className="text-sm text-white">{toast.msg}</div>

            <button
              onClick={() => setToast((t) => ({ ...t, show: false }))}
              className="ml-2 text-white/70 hover:text-white transition"
              aria-label="Cerrar"
            >
              ✕
            </button>
          </div>
        </div>
      </main>
    </>
  );
}
