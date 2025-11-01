import React, { useState } from 'react';
import { Camera, Pencil } from 'lucide-react';

type Props = {
  name?: string;
  email?: string;
  id?: string | number;
  avatarUrl?: string;
};

export default function ProfilePanel({ name, email, id, avatarUrl }: Props) {
  const [isEditing, setIsEditing] = useState(false);

  return (
    <div className="min-h-screen bg-[#070A14] text-white flex flex-col items-center pt-12 px-6 relative">
      {/* Botón Editar */}
      <div className="absolute top-8 right-10">
        <button
          onClick={() => setIsEditing(!isEditing)}
          className="flex items-center gap-1 bg-white/5 hover:bg-white/10 text-white/70 px-3 py-1.5 rounded-md text-sm transition"
        >
          <Pencil className="w-3.5 h-3.5" />
          {isEditing ? 'Save' : 'Edit'}
        </button>
      </div>

      {/* Imagen de perfil */}
      <div className="relative w-40 h-40 rounded-full bg-white/5 border border-white/10 flex items-center justify-center">
        <img
          src={avatarUrl || '/images/tu_foto.jpg'}
          alt="Profile"
          className="w-full h-full rounded-full object-cover opacity-90"
        />
        <button className="absolute bottom-2 right-2 bg-white text-black rounded-full p-1.5 shadow-md hover:scale-105 transition">
          <Camera className="w-4 h-4" />
        </button>
      </div>

      {/* Campos */}
      <div className="w-full max-w-[600px] mt-10 flex flex-col gap-6">
        <ProfileField label="Name" value={name || '—'} editable={isEditing} />
        <ProfileField label="Email" value={email || '—'} editable={isEditing} />
        <ProfileField label="ID" value={String(id ?? '—')} editable={false} />
      </div>
    </div>
  );
}

interface ProfileFieldProps {
  label: string;
  value: string;
  editable?: boolean;
}

function ProfileField({ label, value, editable = false }: ProfileFieldProps) {
  return (
    <div>
      <label className="block text-sm text-white/60 mb-2">{label}</label>
      {editable ? (
        <input
          type="text"
          defaultValue={value}
          className="w-full bg-white/5 text-white text-center rounded-lg py-3 px-4 outline-none border border-white/10 focus:border-[#8B3DFF] focus:ring-1 focus:ring-[#8B3DFF] transition"
        />
      ) : (
        <div className="w-full bg-white/5 text-white/80 text-center rounded-lg py-3 px-4 border border-white/5">
          {value}
        </div>
      )}
    </div>
  );
}
