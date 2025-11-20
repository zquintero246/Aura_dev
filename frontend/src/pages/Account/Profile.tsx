import React, { ChangeEvent, useCallback, useEffect, useRef, useState } from 'react';
import { Check } from 'lucide-react';
import { Camera, Lock, Mail, Plus } from 'lucide-react';
import { getMyLocation } from '@/lib/location';
import type { User } from '@/lib/auth';
import { updateProfile } from '@/lib/profile';

type Props = {
  name?: string;
  email?: string;
  id?: string | number;
  avatarUrl?: string;
  onProfileChange?: (user: User) => void;
};

type ProfileForm = {
  fullName: string;
  country: string;
  language: string;
  timezone: string;
};

type FieldDefinition = {
  key: keyof ProfileForm;
  label: string;
  placeholder: string;
  type: 'input' | 'select';
  options?: { value: string; label: string }[];
  disabled?: boolean;
};

const COUNTRY_OPTIONS = [
  { value: 'colombia', label: 'Colombia' },
  { value: 'mexico', label: 'México' },
  { value: 'argentina', label: 'Argentina' },
  { value: 'chile', label: 'Chile' },
  { value: 'usa', label: 'Estados Unidos' },
];

const COUNTRY_TIMEZONES: Record<string, string> = {
  colombia: 'GMT-5 · Bogotá',
  mexico: 'GMT-6 · Ciudad de México',
  argentina: 'GMT-3 · Buenos Aires',
  chile: 'GMT-4 · Santiago',
  usa: 'GMT-5 · Washington D.C.',
};

const LANGUAGE_OPTIONS = [{ value: 'es', label: 'Español' }];

const TIMEZONE_OPTIONS = [
  { value: 'GMT-5 · Bogotá', label: 'GMT-5 · Bogotá' },
  { value: 'GMT-6 · Ciudad de México', label: 'GMT-6 · Ciudad de México' },
  { value: 'GMT-3 · Buenos Aires', label: 'GMT-3 · Buenos Aires' },
  { value: 'GMT-4 · Santiago', label: 'GMT-4 · Santiago' },
  { value: 'GMT-5 · Washington D.C.', label: 'GMT-5 · Washington D.C.' },
];

const FIELD_DEFS: FieldDefinition[] = [
  { key: 'fullName', label: 'Nombre completo', placeholder: 'Tu nombre completo', type: 'input' },
  {
    key: 'country',
    label: 'País',
    placeholder: 'Registra tu país en "HomeAssistant"',
    type: 'select',
    options: COUNTRY_OPTIONS,
    disabled: true,
  },
  {
    key: 'language',
    label: 'Idioma',
    placeholder: 'Español',
    type: 'select',
    options: LANGUAGE_OPTIONS,
    disabled: true,
  },
  {
    key: 'timezone',
    label: 'Zona horaria',
    placeholder: '--',
    type: 'select',
    options: TIMEZONE_OPTIONS,
    disabled: true,
  },
];

const INITIAL_FORM: ProfileForm = {
  fullName: '',
  country: '',
  language: 'es',
  timezone: '',
};

export default function ProfilePanel({ name, email, id, avatarUrl, onProfileChange }: Props) {
  const [form, setForm] = useState<ProfileForm>(INITIAL_FORM);
  const [status, setStatus] = useState('');
  const [showChangePassword, setShowChangePassword] = useState(false);
  const [currentPass, setCurrentPass] = useState('');
  const [newPass, setNewPass] = useState('');
  const [confirmPass, setConfirmPass] = useState('');
  const [copied, setCopied] = useState(false);
  const statusTimer = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const primaryEmail = email || 'alexarawles@gmail.com';

  const scheduleStatus = useCallback((message: string, duration = 3200) => {
    setStatus(message);
    if (statusTimer.current && typeof window !== 'undefined') {
      window.clearTimeout(statusTimer.current);
    }
    if (typeof window !== 'undefined') {
      statusTimer.current = window.setTimeout(() => {
        setStatus('');
        statusTimer.current = null;
      }, duration);
    }
  }, []);

  const saveProfile = useCallback(async () => {
    const trimmed = form.fullName.trim();
    if (!trimmed) {
      throw new Error('empty-name');
    }
    const formData = new FormData();
    formData.append('name', trimmed);
    const updatedUser = await updateProfile(formData);
    if (!updatedUser) {
      throw new Error('no-user');
    }
    onProfileChange?.(updatedUser);
    setForm((prev) => ({ ...prev, fullName: trimmed }));
    return updatedUser;
  }, [form.fullName, onProfileChange]);

  const handleAvatarButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleAvatarSelected = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;
      const payload = new FormData();
      payload.append('avatar', file);
      try {
        const updatedUser = await updateProfile(payload);
        if (!updatedUser) throw new Error('no-user');
        onProfileChange?.(updatedUser);
        scheduleStatus('Foto actualizada');
      } catch {
        scheduleStatus('No se pudo subir la imagen');
      } finally {
        event.target.value = '';
      }
    },
    [onProfileChange, scheduleStatus]
  );

  useEffect(() => {
    setForm((prev) => ({
      ...prev,
      fullName: name || prev.fullName,
    }));
  }, [name]);

  useEffect(() => {
    (async () => {
      try {
        const location = await getMyLocation();
        const normalized = location?.country?.toLowerCase();
        if (normalized && COUNTRY_TIMEZONES[normalized]) {
          setForm((prev) => ({
            ...prev,
            country: normalized,
            timezone: COUNTRY_TIMEZONES[normalized],
          }));
        }
      } catch (_err) {
        // ignore
      }
    })();

    return () => {
      if (statusTimer.current && typeof window !== 'undefined') {
        window.clearTimeout(statusTimer.current);
      }
    };
  }, []);

  const handleChangeField = (key: keyof ProfileForm, value: string) => {
    setForm((prev) => {
      const next = { ...prev, [key]: value };
      if (key === 'country') {
        next.timezone = COUNTRY_TIMEZONES[value] || '';
      }
      return next;
    });
  };

  const handleFullNameBlur = useCallback(async () => {
    try {
      await saveProfile();
      scheduleStatus('Nombre guardado');
    } catch {
      scheduleStatus('No se pudo actualizar el nombre');
    }
  }, [saveProfile, scheduleStatus]);

  const handleChangePasswordSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (newPass !== confirmPass) {
      scheduleStatus('Las contraseñas no coinciden');
      return;
    }
    scheduleStatus('Solicitud de cambio enviada');
    setShowChangePassword(false);
    setCurrentPass('');
    setNewPass('');
    setConfirmPass('');
  };

  const handleAddEmailAddress = () => {
    if (typeof window !== 'undefined') {
      window.alert('Agregar una dirección de correo está en desarrollo.');
    }
  };

  const displayName = form.fullName || name || (id ? `Usuario ${id}` : 'Usuario de Aura');
  const displayAvatar = avatarUrl || '/images/avatar_demo.jpg';

  const handleCopyId = () => {
    if (!id) return;
    navigator.clipboard.writeText(String(id));
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <>
      <div className="h-full w-full bg-[#030712] text-white">
        <div className="h-full w-full">
          <div className="h-full min-h-full w-full bg-[#0b0f1f] shadow-[0_40px_80px_rgba(0,0,0,0.6)] border border-white/5">
            <div className="h-36 w-full bg-gradient-to-r from-[#0e1a3b] via-[#201643] to-[#2c1b4b]" />
            <div className="px-6 pb-10 pt-2 md:px-10">
              <div className="flex flex-col items-start gap-4 md:flex-row md:items-end md:justify-between mt-2">
                <div className="flex items-center gap-6">
                  <div className="relative h-28 w-28 rounded-full border-4 border-[#0b0f1f] bg-[#0b0f1f] shadow-lg -mt-10 md:-mt-14">
                    <img
                      src={displayAvatar}
                      alt={displayName}
                      className="h-full w-full rounded-full object-cover"
                    />
                    <button
                      type="button"
                      onClick={handleAvatarButtonClick}
                      className="absolute bottom-0 right-0 grid h-10 w-10 place-items-center rounded-full bg-white text-[#0f172a] shadow-md transition hover:scale-105"
                      aria-label="Cambiar foto de perfil"
                    >
                      <Camera className="h-4 w-4" />
                    </button>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={handleAvatarSelected}
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <div className="flex items-center gap-3">
                      <p className="text-2xl font-semibold leading-tight text-white">
                        {displayName}
                      </p>
                      {id != null && (
                        <div className="relative flex items-center">
                          <div
                            onClick={handleCopyId}
                            className="flex cursor-pointer select-none items-center rounded-full bg-[#0f172a] px-3 py-1 text-left text-[11px] font-semibold tracking-widest text-[#5c41d6] transition active:scale-95"
                            title="Copiar ID"
                          >
                            <span className="text-[9px] font-normal uppercase tracking-[0.3em] text-[#5c41d6] opacity-70">
                              ID:
                            </span>
                            <span className="ml-1 text-[9px] font-semibold text-[#5c41d6]">
                              {id}
                            </span>
                          </div>
                          {copied && (
                            <span className="absolute -right-20 top-1/2 -translate-y-1/2 flex items-center gap-1 text-[10px] font-medium text-green-600 animate-fade-out">
                              <Check className="h-3 w-3" strokeWidth={3} />
                              Copiado
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                    <p className="text-sm text-[#475467]">{primaryEmail}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3 md:flex-col md:items-end">
                  <button
                    type="button"
                    onClick={() => setShowChangePassword(true)}
                    className="flex items-center gap-2 rounded-full bg-[#0f172a] px-5 py-2 text-sm font-semibold text-white transition hover:bg-[#111827]"
                  >
                    <Lock className="h-4 w-4" />
                    Cambiar contraseña
                  </button>
                  {status && <span className="text-xs font-medium text-[#10b981]">{status}</span>}
                </div>
              </div>

              <div className="mt-8 grid gap-4 md:grid-cols-2">
                {FIELD_DEFS.map((field) => {
                  const value = form[field.key];
                  const isSelect = field.type === 'select';
                  const controlClasses =
                    'mt-2 block w-full rounded-2xl border border-white/10 bg-[#050b1e] px-4 py-3 text-sm font-medium text-white placeholder:text-white/40 focus:border-[#8B3DFF] focus:outline-none focus:ring-1 focus:ring-[#8B3DFF]/30 transition';
                  return (
                    <div key={field.key} className="space-y-1">
                      <div className="text-xs font-semibold uppercase tracking-[0.3em] text-[#8b93b1]">
                        {field.label}
                      </div>
                      {isSelect ? (
                        <select
                          value={value}
                          onChange={(event) => handleChangeField(field.key, event.target.value)}
                          disabled={field.disabled}
                          className={`${controlClasses} appearance-none bg-no-repeat pr-4 ${field.disabled ? 'opacity-80' : ''}`}
                          style={{ backgroundImage: 'none' }}
                        >
                          {field.placeholder && (
                            <option value="" disabled>
                              {field.placeholder}
                            </option>
                          )}
                          {field.options?.map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <input
                          type="text"
                          value={value}
                          onChange={(event) => handleChangeField(field.key, event.target.value)}
                          placeholder={field.placeholder}
                          className={controlClasses}
                          onBlur={field.key === 'fullName' ? handleFullNameBlur : undefined}
                        />
                      )}
                    </div>
                  );
                })}
              </div>

              <div className="mt-10 rounded-[26px] border border-white/10 bg-[#0b1326] p-6 shadow-[0_20px_60px_rgba(0,0,0,0.55)]">
                <p className="text-sm font-semibold text-white/80">My email address</p>
                <div className="mt-4 flex flex-col gap-4 md:flex-row md:items-center">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#11172c]">
                    <Mail className="h-5 w-5 text-[#7b9cff]" />
                  </div>
                  <div className="flex-1 space-y-1">
                    <p className="text-sm font-semibold text-white">{primaryEmail}</p>
                    <p className="text-xs text-white/50">1 month ago</p>
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    <button
                      type="button"
                      onClick={handleAddEmailAddress}
                      className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-gradient-to-r from-[#3c1d91] via-[#7b2fe3] to-[#b97cff] px-4 py-2 text-[13px] font-semibold text-white shadow-lg shadow-[#7B2FE3]/30 transition hover:brightness-110"
                    >
                      <Plus className="h-4 w-4" />
                      Add Email Address
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      {showChangePassword && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4">
          <div className="w-full max-w-md rounded-2xl bg-white p-6 shadow-2xl">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-[#0f172a]">Cambiar contraseña</h3>
              <button
                type="button"
                onClick={() => setShowChangePassword(false)}
                className="text-sm text-[#475467] hover:text-[#0f172a]"
              >
                Cerrar
              </button>
            </div>
            <form className="mt-4 space-y-3" onSubmit={handleChangePasswordSubmit}>
              <input
                type="password"
                value={currentPass}
                onChange={(event) => setCurrentPass(event.target.value)}
                placeholder="Contraseña actual"
                className="w-full rounded-lg border border-[#e5e7eb] px-3 py-2"
                required
              />
              <input
                type="password"
                value={newPass}
                onChange={(event) => setNewPass(event.target.value)}
                placeholder="Nueva contraseña"
                className="w-full rounded-lg border border-[#e5e7eb] px-3 py-2"
                required
              />
              <input
                type="password"
                value={confirmPass}
                onChange={(event) => setConfirmPass(event.target.value)}
                placeholder="Confirmar contraseña"
                className="w-full rounded-lg border border-[#e5e7eb] px-3 py-2"
                required
              />
              <button
                type="submit"
                className="w-full rounded-full bg-[#8B3DFF] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[#7A2CF0]"
              >
                Actualizar contraseña
              </button>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
