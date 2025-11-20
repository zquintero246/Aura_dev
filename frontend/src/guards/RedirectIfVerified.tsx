import React, { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { me } from '@/lib/auth';

type Props = { children: React.ReactElement };

export default function RedirectIfVerified({ children }: Props) {
  const [redirect, setRedirect] = useState<null | 'chat' | 'stay'>(null);

  useEffect(() => {
    let mounted = true;
    const run = async () => {
      try {
        const res = await me();
        if (!mounted) return;
        if (res?.user?.email_verified_at) setRedirect('chat');
        else setRedirect('stay');
      } catch (_) {
        setRedirect('stay');
      }
    };
    run();
    return () => { mounted = false; };
  }, []);

  if (redirect === 'chat') return <Navigate to="/chat" replace />;
  return children;
}

