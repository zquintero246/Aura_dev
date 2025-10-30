import React, { useState } from 'react';

export default function ChangePassword() {
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [confirm, setConfirm] = useState('');

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    alert('Cambiar contraseña: pendiente de backend');
  };

  return (
    <div className="p-6 text-white max-w-md">
      <h1 className="text-xl font-semibold">Cambiar contraseña</h1>
      <form className="mt-4 space-y-3" onSubmit={submit}>
        <input className="w-full px-3 py-2 rounded bg-white/10" type="password" placeholder="Contraseña actual" value={current} onChange={(e)=>setCurrent(e.target.value)} />
        <input className="w-full px-3 py-2 rounded bg-white/10" type="password" placeholder="Nueva contraseña" value={next} onChange={(e)=>setNext(e.target.value)} />
        <input className="w-full px-3 py-2 rounded bg-white/10" type="password" placeholder="Confirmar contraseña" value={confirm} onChange={(e)=>setConfirm(e.target.value)} />
        <button className="px-4 py-2 rounded bg-[#7B2FE3]" type="submit">Guardar</button>
      </form>
    </div>
  );
}

