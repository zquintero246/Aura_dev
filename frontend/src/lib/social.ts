// Social OAuth popup flow utilities
export type SocialAuthPayload = {
  status: 'ok' | string;
  action: 'auth-complete' | string;
  redirect?: string;
  token?: string | null;
};

function openCenteredPopup(url: string, title = 'Auth', w = 600, h = 700) {
  const dualScreenLeft = window.screenLeft !== undefined ? window.screenLeft : window.screenX;
  const dualScreenTop = window.screenTop !== undefined ? window.screenTop : window.screenY;

  const width = window.innerWidth || document.documentElement.clientWidth || screen.width;
  const height = window.innerHeight || document.documentElement.clientHeight || screen.height;

  const systemZoom = width / window.screen.availWidth;
  const left = (width - w) / 2 / systemZoom + dualScreenLeft;
  const top = (height - h) / 2 / systemZoom + dualScreenTop;
  const features = `scrollbars=yes, width=${w / systemZoom}, height=${h / systemZoom}, top=${top}, left=${left}`;

  const newWindow = window.open(url, title, features);
  if (newWindow) newWindow.focus();
  return newWindow;
}

export async function socialLogin(provider: 'github' | 'google'): Promise<SocialAuthPayload | null> {
  const base = (import.meta.env.VITE_BACKEND_URL as string) || 'http://127.0.0.1:8000';
  const popup = openCenteredPopup(`${base}/auth/${provider}/redirect`, `Login with ${provider}`);
  if (!popup) return null;

  return new Promise<SocialAuthPayload | null>((resolve) => {
    let received = false;

    const cleanup = () => {
      window.removeEventListener('message', onMessage);
      clearInterval(closeCheck);
    };

    const onMessage = (event: MessageEvent) => {
      const data = (event && (event as any).data) as SocialAuthPayload;
      if (data && data.status === 'ok' && data.action === 'auth-complete') {
        received = true;
        cleanup();
        try { popup.close(); } catch {}
        resolve(data);
      }
    };

    window.addEventListener('message', onMessage);

    const closeCheck = window.setInterval(() => {
      if (popup.closed && !received) {
        cleanup();
        resolve(null);
      }
    }, 500);
  });
}
