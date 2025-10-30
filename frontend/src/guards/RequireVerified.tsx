import React, { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { me } from '../lib/auth';

type Props = { children: React.ReactElement };

export default function RequireVerified({ children }: Props) {
  const [status, setStatus] = useState<'checking' | 'unauth' | 'unverified' | 'ok'>('checking');

  useEffect(() => {
    let mounted = true;
    const run = async () => {
      try {
        const res = await me();
        if (!mounted) return;
        if (res?.user?.email_verified_at) setStatus('ok');
        else setStatus('unverified');
      } catch (e: any) {
        setStatus('unauth');
      }
    };
    run();
    return () => { mounted = false; };
  }, []);

  if (status === 'checking') return null;
  if (status === 'unauth') return <Navigate to="/login" replace />;
  if (status === 'unverified') return <Navigate to="/verify-email" replace />;
  return children;
}

