import React, { useState } from 'react';
import { Helmet } from 'react-helmet';
import { Link } from 'react-router-dom';
import Button from './components/ui/Button';
import Particles from './components/ui/Particles';

const VerificationCode = () => {
  const [code, setCode] = useState(['', '', '', '', '', '']);
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (index: number, value: string) => {
    if (!/^[0-9]?$/.test(value)) return;
    const newCode = [...code];
    newCode[index] = value;
    setCode(newCode);

    // Avanzar automáticamente al siguiente input
    if (value && index < code.length - 1) {
      const nextInput = document.getElementById(`code-${index + 1}`);
      nextInput?.focus();
    }
  };

  const handleVerify = async () => {
    setIsLoading(true);
    await new Promise((r) => setTimeout(r, 1000));
    setIsLoading(false);
  };

  const handleResend = () => {
    showToast('Código reenviado al correo electrónico.');
  };

  // Toast
  const [toast, setToast] = useState<{ msg: string; show: boolean }>({ msg: '', show: false });
  const showToast = (msg: string) => {
    setToast({ msg, show: true });
    setTimeout(() => setToast((t) => ({ ...t, show: false })), 3000);
  };

  return (
    <>
      <Helmet>
        <title>Verificación de código</title>
      </Helmet>

      {/* Fondo */}
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-[#141C34] to-[#0a0a0a] -z-10 overflow-hidden display-none lg:block">
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
      <main className="min-h-screen flex flex-col items-center justify-center px-4 sm:px-6 py-12">
        {/* Logo */}
        <header className="mb-8 flex flex-col items-center">
          <img
            src="/images/logo_debajo_aura.svg"
            alt="Aura"
            className="w-[140px] sm:w-[160px] md:w-[100px] h-auto"
          />
        </header>

        {/* Card */}
        <section className="w-full flex justify-center">
          <div className="relative w-full max-w-[540px]">
            <div className="absolute inset-0 rounded-[24px] bg-[rgba(29,29,29,0.25)]" />
            <div className="relative rounded-[24px] overflow-hidden border border-white/5 shadow-[0_10px_40px_-10px_rgba(0,0,0,0.5)] backdrop-blur-sm text-center px-6 sm:px-10 py-10">
              <h2 className="text-[22px] sm:text-[24px] font-bold text-white">
                Introduce el código
              </h2>
              <p className="mt-2 text-[14px] text-[#686868]">
                Por favor ingrese el código que acabamos de enviarle.
              </p>

              {/* Inputs de código */}
              <div className="flex justify-center gap-3 sm:gap-4 mt-8">
                {code.map((digit, index) => (
                  <input
                    key={index}
                    id={`code-${index}`}
                    type="text"
                    value={digit}
                    maxLength={1}
                    onChange={(e) => handleChange(index, e.target.value)}
                    className="w-10 h-12 sm:w-12 sm:h-14 text-center text-lg font-semibold rounded-[10px] bg-white/5 text-white placeholder:text-white/40 ring-1 ring-white/10 focus:ring-2 focus:ring-[#8B3DFF6e] transition-shadow"
                  />
                ))}
              </div>

              {/* Botón */}
              <div className="mt-10 flex justify-center">
                <Button
                  onClick={handleVerify}
                  disabled={isLoading || code.some((c) => !c)}
                  className="w-[230px] h-11 rounded-full bg-[#7B2FE3] hover:bg-[#6c29c9] active:scale-[0.98] transition text-white"
                  border_border="transparent"
                  layout_width="auto"
                  padding="sm"
                  position="relative"
                  layout_gap="md"
                  variant="primary"
                  size="medium"
                >
                  {isLoading ? 'Verificando...' : 'Registrarse'}
                </Button>
              </div>

              {/* Reenviar código */}
              <div className="mt-5">
                <button
                  onClick={handleResend}
                  className="text-[14px] text-[#8B3DFF] hover:underline"
                >
                  Reenviar código.
                </button>
              </div>
            </div>
          </div>
        </section>

        {/* Toast flotante */}
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
};

export default VerificationCode;
