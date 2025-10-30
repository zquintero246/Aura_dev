@extends('layouts.app')

@section('content')
    <style>
        body {
            background: radial-gradient(circle at top, #1f2937, #0f172a 60%, #020617);
            color: #e5e7eb;
            font-family: 'Nunito', sans-serif;
        }

        .aura-verification {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: calc(100vh - 120px);
        }

        .aura-card {
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 24px;
            padding: 3rem;
            max-width: 520px;
            width: 100%;
            box-shadow: 0 40px 80px rgba(14, 21, 37, 0.45);
            position: relative;
            overflow: hidden;
        }

        .aura-card::before {
            content: '';
            position: absolute;
            inset: -60% -20% auto auto;
            width: 240px;
            height: 240px;
            background: radial-gradient(circle, rgba(124, 58, 237, 0.4), transparent 70%);
            z-index: 0;
            transform: rotate(25deg);
        }

        .aura-card::after {
            content: '';
            position: absolute;
            inset: auto auto -50% -15%;
            width: 260px;
            height: 260px;
            background: radial-gradient(circle, rgba(14, 165, 233, 0.35), transparent 70%);
            z-index: 0;
            transform: rotate(-20deg);
        }

        .aura-card-content {
            position: relative;
            z-index: 1;
        }

        .aura-title {
            font-size: 1.9rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: #f9fafb;
        }

        .aura-subtitle {
            font-size: 1rem;
            line-height: 1.75rem;
            color: #cbd5f5;
            margin-bottom: 2rem;
        }

        .aura-highlight {
            color: #a855f7;
            font-weight: 600;
        }

        .aura-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.85rem 1.6rem;
            background: linear-gradient(135deg, #6366f1, #8b5cf6 40%, #ec4899);
            border-radius: 9999px;
            color: #fdf4ff;
            text-decoration: none;
            font-weight: 600;
            letter-spacing: 0.02em;
            border: none;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .aura-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 32px rgba(99, 102, 241, 0.35);
            color: #fff;
        }

        .aura-footnote {
            margin-top: 1.75rem;
            font-size: 0.95rem;
            color: #94a3b8;
        }

        .alert-success {
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.4);
            color: #bbf7d0;
            border-radius: 12px;
            padding: 0.9rem 1.2rem;
            margin-bottom: 1.5rem;
        }
    </style>

    <div class="aura-verification">
        <div class="aura-card">
            <div class="aura-card-content text-center">
                <h1 class="aura-title">Verifica tu correo electrónico</h1>

                @if (session('resent'))
                    <div class="alert alert-success" role="alert">
                        ¡Listo! Te acabamos de enviar un nuevo enlace de verificación.
                    </div>
                @endif

                <p class="aura-subtitle">
                    Hemos enviado un mensaje a <span class="aura-highlight">{{ auth()->user()->email }}</span>.
                    Abre tu bandeja de entrada y haz clic en el botón de verificación para activar tu cuenta.
                </p>

                <p class="aura-subtitle">
                    ¿No recibiste el correo? Puedes solicitar uno nuevo a continuación.
                </p>

                <form method="POST" action="{{ route('verification.resend') }}">
                    @csrf
                    <button type="submit" class="aura-button">Reenviar enlace de verificación</button>
                </form>

                <p class="aura-footnote">
                    Si el enlace no funciona, copia y pega la URL completa en tu navegador.
                    <br>Gracias por confiar en <strong>Aura</strong>.
                </p>
            </div>
        </div>
    </div>
@endsection
