import React, { useState } from 'react';
import { Helmet } from 'react-helmet';
import { Link, useNavigate } from 'react-router-dom';
import Button from '@/pages/Authentication/components/ui/Button';
import EditText from '@/pages/Authentication/components/ui/EditText';
import Particles from '@/pages/Authentication/components/ui/Particles';
import { login as apiLogin } from '@/lib/auth';
import { socialLogin, SocialAuthPayload } from '@/lib/social';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  // NOTE: The postMessage listener is managed inside socialLogin().
  // We only handle its returned payload here to store token and navigate.

  const handleLogin = async () => {
    try {
      setIsLoading(true);
      const { user } = await apiLogin(email, password);
      setIsLoading(false);
      if (!user.email_verified_at) {
        navigate('/verify-email');
      } else {
        navigate('/chat');
      }
    } catch (err: any) {
      setIsLoading(false);
      alert(err?.response?.data?.message || 'Error al iniciar sesión');
    }
  };

  const handleGithubLogin = async () => {
    const payload: SocialAuthPayload | null = await socialLogin('github');
    if (payload && payload.status === 'ok' && payload.action === 'auth-complete') {
      // Guardar token si vino en el payload (ajusta a tu store real)
      if (payload.token) {
        try { localStorage.setItem('auth_token', String(payload.token)); } catch {}
      } else {
        // TODO: usar cookie de sesión / Sanctum si aplica
      }
      const target = payload.redirect || '/verify-email';
      if (/^https?:\/\//i.test(target)) {
        window.location.assign(target);
      } else {
        navigate(target);
      }
    } else {
      // Solo alert si NO llegó postMessage válido y el popup se cerró
      alert('No se pudo completar el inicio con GitHub.');
    }
  };
  const handleGoogleLogin = async () => {
    const payload: SocialAuthPayload | null = await socialLogin('google');
    if (payload && payload.status === 'ok' && payload.action === 'auth-complete') {
      if (payload.token) {
        try { localStorage.setItem('auth_token', String(payload.token)); } catch {}
      } else {
        // TODO: usar cookie de sesión / Sanctum si aplica
      }
      const target = payload.redirect || '/verify-email';
      if (/^https?:\/\//i.test(target)) {
        window.location.assign(target);
      } else {
        navigate(target);
      }
    } else {
      alert('No se pudo completar el inicio con Google.');
    }
  };

  return (
    <>
      <Helmet>
        <title>Ingresar</title>
      </Helmet>

      {/* Fondo (idéntico) */}
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-[#141C34] to-[#0a0a0a] -z-10 overflow-hidden display-none lg:block">
        <Particles
          particleColors={['#ffffff', '#7405B4']}
          particleCount={200}
          particleSpread={10}
          speed={0.1}
          particleBaseSize={100}
          moveParticlesOnHover
          alphaParticles={false}
          disableRotation={false}
        />
      </div>

      <main className="min-h-screen flex flex-col items-center justify-center px-4 sm:px-6 py-12">
        {/* Logo centrado */}
        <header className="mb-8 flex flex-col items-center">
          <Link to="/">
            <img
              src="/images/logo_debajo_aura.svg"
              alt="Aura"
              className="w-[140px] sm:w-[160px] md:w-[100px] h-auto"
            />
          </Link>
        </header>

        {/* Card */}
        <section className="w-full flex justify-center">
          <div className="relative w-full max-w-[540px]">
            <div className="absolute inset-0 rounded-[24px] bg-[rgba(29,29,29,0.25)]" />
            <div className="relative rounded-[24px] overflow-hidden border border-white/5 shadow-[0_10px_40px_-10px_rgba(0,0,0,0.5)] backdrop-blur-sm">
              {/* Header del card */}
              <div className="px-6 sm:px-8 md:px-10 pt-7 pb-6 text-left sm:text-center">
                <h2 className="text-[22px] sm:text-[24px] font-bold leading-tight text-white">
                  Ingresar
                </h2>
                <p className="mt-2 text-[14px] leading-6 text-[#686868]">
                  Ingresa tus credenciales para acceder a su cuenta
                </p>
              </div>

              <div className="h-px bg-white/10" />

              {/* Formulario */}
              <div className="px-6 sm:px-8 md:px-10 py-8">
                <div className="max-w-[460px] mx-auto">
                  <label
                    htmlFor="email"
                    className="block text-[15px] font-semibold text-white mb-2"
                  >
                    Email
                  </label>
                  <EditText
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e: any) => setEmail(e?.target?.value)}
                    className="w-full h-12 rounded-[10px] bg-white/5 text-white placeholder:text-white/40 ring-1 ring-white/10 focus:ring-2 focus:ring-[#8B3DFF6e] transition-shadow px-5 py-3"
                    padding="16px"
                  />
                </div>

                {/* Contraseña */}
                <div className="max-w-[460px] mx-auto mt-7 relative">
                  <label
                    htmlFor="password"
                    className="block text-[15px] font-semibold text-white mb-2"
                  >
                    Contraseña
                  </label>
                  <div className="relative">
                    <EditText
                      id="password"
                      type={showPassword ? 'text' : 'password'}
                      value={password}
                      onChange={(e: any) => setPassword(e?.target?.value)}
                      className="w-full h-12 rounded-[10px] bg-white/5 text-white placeholder:text-white/40 ring-1 ring-white/10 focus:ring-2 focus:ring-[#8B3DFF6e] transition-shadow px-5 pr-12 py-3"
                      padding="16px"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute inset-y-0 right-4 flex items-center text-white/70 hover:text-white transition-colors"
                      aria-label={showPassword ? 'Ocultar contraseña' : 'Mostrar contraseña'}
                    >
                      {showPassword ? (
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          strokeWidth={1.8}
                          stroke="currentColor"
                          className="w-5 h-5"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M2.036 12.322a1.012 1.012 0 010-.644C3.423 7.51 7.283 4.5 12 4.5c4.717 0 8.577 3.01 9.964 7.178.07.2.07.434 0 .644C20.577 16.49 16.717 19.5 12 19.5c-4.717 0-8.577-3.01-9.964-7.178z"
                          />
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                          />
                        </svg>
                      ) : (
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          strokeWidth={1.8}
                          stroke="currentColor"
                          className="w-5 h-5"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.21 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.79 0 8.774 3.162 10.066 7.5a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65"
                          />
                        </svg>
                      )}
                    </button>
                  </div>
                </div>

                {/* Olvidaste tu contraseña */}
                <div className="max-w-[460px] mx-auto mt-3">
                  <button
                    type="button"
                    className="text-[14px] text-[#686868] hover:text-white/80 transition-colors"
                  >
                    ¿Olvidaste tu contraseña?
                  </button>
                </div>

                {/* Botón */}
                <div className="mt-8 flex justify-center">
                  <Button
                    onClick={handleLogin}
                    disabled={isLoading || !email || !password}
                    className="w-[230px] h-11 rounded-full bg-[#7B2FE3] hover:bg-[#6c29c9] active:scale-[0.98] transition text-white"
                    border_border="transparent"
                    layout_width="auto"
                    padding="sm"
                    position="relative"
                    layout_gap="md"
                    variant="primary"
                    size="medium"
                  >
                    {isLoading ? 'Ingresando...' : 'Ingresar'}
                  </Button>
                </div>

                {/* Register link */}
                <div className="max-w-[460px] mx-auto mt-8 text-[14px] text-[#686868]">
                  ¿No tienes una cuenta?{' '}
                  <Link
                    to="/register"
                    className="font-semibold text-white hover:opacity-80 transition-opacity"
                  >
                    Regístrate
                  </Link>
                </div>
              </div>

              <div className="h-px bg-white/10" />

              {/* Social */}
              <div className="px-6 sm:px-8 md:px-10 py-8">
                <div className="max-w-[400px] mx-auto grid gap-5">
                  <button
                    onClick={handleGithubLogin}
                    className="h-12 w-full rounded-[12px] bg-white text-black border border-white/10 flex items-center justify-center gap-2 hover:translate-y-[-1px] hover:shadow transition-all focus:outline-none focus:ring-2 focus:ring-white/30"
                  >
                    <img src="/images/github.svg" alt="" className="w-5 h-5" />
                    <span className="text-[15px] font-semibold">Github</span>
                  </button>

                  <button
                    onClick={handleGoogleLogin}
                    className="h-12 w-full rounded-[12px] bg-white text-black border border-white/10 flex items-center justify-center gap-2 hover:translate-y-[-1px] hover:shadow transition-all focus:outline-none focus:ring-2 focus:ring-white/30"
                  >
                    <img src="/images/google.svg" alt="" className="w-5 h-5" />
                    <span className="text-[15px] font-semibold">Google</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </>
  );
};

export default Login;
