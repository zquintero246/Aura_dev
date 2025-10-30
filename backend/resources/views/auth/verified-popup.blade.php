<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Correo verificado</title>
    <style>
        html, body { height: 100%; margin: 0; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif; background: #0B0E19; color: #fff; }
        .wrap { height: 100%; display: flex; align-items: center; justify-content: center; padding: 24px; }
        .card { max-width: 520px; width: 100%; background: rgba(15,18,30,0.9); border-radius: 24px; padding: 32px; border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 10px 50px rgba(0,0,0,0.6); text-align: center; }
        .title { font-size: 20px; font-weight: 700; margin: 0 0 8px; }
        .subtitle { font-size: 14px; color: #A3A3A3; line-height: 1.6; margin: 0 0 16px; }
        .counter { font-weight: 600; color: #CA5CF5; }
        .btn { display:inline-block; margin-top: 16px; background: linear-gradient(90deg,#CA5CF5 0%,#7405B4 100%); color:#fff; text-decoration:none; font-weight:600; font-size:14px; padding:10px 18px; border-radius:12px; box-shadow:0 4px 20px rgba(116,5,180,0.4); }
    </style>
    <script>
        // Notify opener (SPA) that email was verified and attempt to close after 5s.
        let seconds = 5;
        function notifyOpener() {
            try {
                if (window.opener && !window.opener.closed) {
                    window.opener.postMessage({ type: 'aura-email-verified', status: 'ok' }, '*');
                }
            } catch (e) { /* ignore */ }
        }
        function tick() {
            seconds -= 1;
            var el = document.getElementById('counter');
            if (el) el.textContent = String(seconds);
            if (seconds <= 0) {
                try { window.close(); } catch (e) {}
            } else {
                setTimeout(tick, 1000);
            }
        }
        window.addEventListener('DOMContentLoaded', function(){
            notifyOpener();
            // also re-notify after a small delay in case SPA listener attaches late
            setTimeout(notifyOpener, 800);
            setTimeout(notifyOpener, 1600);
            setTimeout(tick, 1000);
        });
    </script>
    </head>
<body>
    <div class="wrap">
        <div class="card">
            <img src="/images/logo_aura.svg" alt="Aura" style="width:72px;height:auto;margin:0 auto 16px;display:block;opacity:.9" />
            <h1 class="title">¡Tu correo ha sido verificado!</h1>
            <p class="subtitle">Ahora puedes volver a la aplicación. Esta ventana intentará cerrarse automáticamente en <span id="counter" class="counter">5</span> segundos.</p>
            <a class="btn" href="#" onclick="try{window.close();}catch(e){} return false;">Cerrar ahora</a>
        </div>
    </div>
</body>
</html>
