@php
    $year = date('Y');
@endphp

<body
    style="background-color:#0B0E19; margin:0; padding:32px; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" align="center"
        style="background-color:#0B0E19; text-align:center;">
        <tr>
            <td align="center">

                {{-- CONTENEDOR PRINCIPAL --}}
                <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="540"
                    style="background:rgba(15,18,30,0.9); border-radius:28px; padding:48px 40px; color:#fff; border:1px solid rgba(255,255,255,0.08); box-shadow:0 10px 50px rgba(0,0,0,0.6); text-align:center;">

                    {{-- LOGO --}}
                    <tr>
                        <td align="center" style="padding-bottom:24px;">
                            <img src="https://i.ibb.co/mrqd4Fkj/Logo-debajo-aura.png" alt="Aura" width="72"
                                height="auto" style="display:block; margin:0 auto;">
                        </td>
                    </tr>

                    {{-- TÍTULO --}}
                    <tr>
                        <td align="center" style="font-size:18px; font-weight:700; color:#ffffff; padding-bottom:12px;">
                            Gracias por elegirnos
                        </td>
                    </tr>

                    {{-- TEXTO --}}
                    <tr>
                        <td align="center"
                            style="font-size:14px; color:#A3A3A3; line-height:1.6; padding-bottom:28px; max-width:440px;">
                            Estamos emocionados de que formes parte de nuestra comunidad.<br>
                            Por favor, haz clic en el siguiente botón para verificar tu dirección<br>
                            de correo electrónico y activar tu cuenta.
                        </td>
                    </tr>

                    {{-- BOTÓN --}}
                    @isset($actionText)
                        <tr>
                            <td align="center" style="padding:32px 0;">
                                @php
                                    $overrideUrl = preg_replace(
                                        '/https?:\/\/[^\/]+/',
                                        config('app.verification_url', $actionUrl),
                                        $actionUrl,
                                    );
                                @endphp
                                <a href="{{ $overrideUrl }}" target="_blank" rel="noopener"
                                    style="display:inline-block; background:linear-gradient(90deg,#CA5CF5 0%,#7405B4 100%); color:#fff; text-decoration:none; font-weight:600; font-size:15px; padding:14px 36px; border-radius:14px; box-shadow:0 4px 20px rgba(116,5,180,0.4);">
                                    {{ $actionText }}
                                </a>
                            </td>
                        </tr>
                    @endisset

                    {{-- FOOTER --}}
                    <tr>
                        <td align="center" style="font-size:13px; color:#6B7280; line-height:1.6; padding-top:24px;">
                            Si no te registraste en Aura, puedes ignorar este correo.<br><br>
                            Saludos,<br>
                            El equipo de <a href="#" style="color:#CA5CF5; text-decoration:none;">Aura</a>.
                        </td>
                    </tr>

                    {{-- COPYRIGHT --}}
                    <tr>
                        <td align="center" style="font-size:11px; color:#374151; padding-top:24px;">
                            © {{ $year }} AURA. Todos los derechos reservados.
                        </td>
                    </tr>

                </table>
            </td>
        </tr>
    </table>
</body>
