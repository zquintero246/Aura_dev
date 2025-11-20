# Documentación del frontend de Aura (React + Vite)

Este documento describe cómo está organizado el frontend `frontend/`, qué rutas expone, cómo se comunica con el backend/servicios, y qué componentes y librerías gobiernan la experiencia de registro, verificación y chat.

## 1. Visión general

- Proyecto Vite + React (`frontend/vite.config.ts`) con alias (`@/src`), puerto 4028 y salida `build/`.
- Usa Tailwind + componentes propios (`components/ui`, `components/common`) para mantener estilos consistentes, además de utilitarios como `GradientBlinds`, `GridBeams` y `Particles` para fondos animados.
- La entrada (`frontend/src/main.tsx`) aplica el tema guardado (`localStorage:aura:theme`) antes de montar el árbol, carga `src/styles/index.css` (Tailwind + tipografía Inter) y arranca `App`.

## 2. Navegación y protección de rutas

- `App.tsx` renderiza solo `<Routes />`, que define la navegación con `react-router-dom` (`frontend/src/Routes.tsx`).
- Rutas públicas (`/`, `/register`, `/login`, `/verify-email`) están envueltas en `RedirectIfVerified` para llevar a usuarios ya validados directamente a `/chat`.
- `/chat`, `/profile` y `/change-password` están protegidas mediante `RequireVerified`: consulta `lib/auth.me()` y redirige a login o `/verify-email` si la sesión no existe o el email no está verificado.
- La SPA mantiene su propio estado de sesión (cookies + CSRF) a través de los helpers en `lib/auth.ts`.

## 3. Autenticación y verificación

- `pages/Authentication/Register.tsx` y `Login.tsx` usan formularios estilizados (`EditText`, `Button`) para llamar a `lib/auth.register`/`login`. Tras registrarse se redirige a `/verify-email`; tras login se valida `email_verified_at`.
- Ambas vistas ofrecen login social (`socialLogin` en `lib/social.ts`): abren popup centrado hacia `/auth/{provider}/redirect`, escuchan `postMessage` y redirigen según el payload devuelto por el backend (la misma ventana que renderiza `backend/resources/views/auth/verified-popup.blade.php`).
- `VerifyEmailPage.tsx`:
  * Pregunta al backend (/api/auth/me) periódicamente para detectar que `email_verified_at` se llenó.
  * Lanza `resendVerification()` para disparar `/api/auth/email/resend` y muestra un toast con cooldown progresivo (15 → 60 s).
  * Escucha `window.postMessage` desde el popup del backend para cerrar automáticamente la ruta y navegar a `/chat`.

## 4. Cómo se integra la API de Laravel

- `frontend/src/lib/api.ts` es el singleton Axios que apunta a `VITE_BACKEND_URL` (por defecto `http://127.0.0.1:8000`), envía `withCredentials` y soporta CSRF (`XSRF-TOKEN`), por lo tanto las rutas `/api/*` del backend comparten sesión.
- `lib/auth.ts` expone `register`, `login`, `logout`, `me`, `resendVerification`, y `mintChatToken`. Esta última ruta (`/api/auth/token`) devuelve un token Sanctum que se guarda en `localStorage:aura:pat` y se usa en las llamadas al microservicio de chat (ver sección 5).
- `ProfilePanel` y `HomeRegistration` usan `lib/profile.updateProfile` y `lib/location.saveLocation` para persistir nombre/avatar/ubicación del usuario vía `/api/profile` y `/api/location`.
- `lib/chat.ts` encola los mensajes antes/después de llamar a `/api/chat`, construye errores amigables (429, timeout, fallos de upstream) y replica localmente el historial enviando cada mensaje al microservicio (`chatApi.sendMessage`) para que el cliente siempre pueda mostrar registros guardados.

## 5. Arquitectura del chat y microservicios

- El chat principal vive en `pages/Chat/AppLayout.tsx` + sus paneles:
  * `IconRail`, `ConversationsPanel`, `GroupsPanel`, `MainPanel`, `ProfilePanel` y `ChatPanel` organizan el layout, la lista de conversaciones (pineadas/recentes), los estados de creación/renombrado/eliminado y las transiciones visuales (`bootStage`, `theme-xfade`).
  * `MainPanel` alterna entre chat real, vistas de grupo/proyecto o telemetría (`HomeAssistantPanel`), dependiendo de la selección del rail.
  * `ChatPanel` gestiona la lista de mensajes (user/assistant), la búsqueda, los ajustes de tema y las llamadas reales a OpenRouter. Usa `lib/chat` para hablar con `/api/chat` y `lib/chatApi` para persistir el mensaje e ir guardando el título si es necesario.
- `lib/chatApi.ts` apunta a la URL `VITE_CHAT_API_BASE` (`http://127.0.0.1:5080` por defecto) y expone endpoints REST (`/chat/start`, `/chat/history`, `/chat/message`, `/chat/conversations/:id`). Este microservicio requiere un token `Authorization: Bearer`.
  * El interceptor solicita `ensureChatToken()` (que canjea `/api/auth/token`) y, en caso de 401, refresca el token y reintenta.
  * También incluye helpers para listar, actualizar y borrar conversaciones (`frontend/src/lib/conversations.ts` delega en `chatApi` y ofrece wrappers con tipos).
- `AppLayout` llama a `listConversations()` al iniciar sesión, dispara eventos globales (`aura:conversation:realized`, `aura:conversations:close`) para sincronizar la UI y mantiene `localStorage:aura:theme`, `aura:pat`, `aura:uid`.

## 6. Panel de cuenta, ubicación y telemetría

- `pages/Account/Profile.tsx` permite:
  * Actualizar el nombre (envía `FormData` a `/api/profile`).
  * Subir avatar y guardar la URL (`Storage::disk('public')` en backend) via `<input type="file">`.
  * Obtener ubicación desde el backend (`lib/location.getMyLocation`) y mostrarla junto a datos de zona horaria/country (prefijos fijos para LATAM).
- `HomeAssistantPanel` (en `pages/Chat`) representa la telemetría:
  * Usa `lib/location`, `lib/weatherApi` (microservicio en `VITE_MICROSERVICES_URL/api/dashboard`), `lib/weatherHistory`, `lib/devices` y `lib/countriesNow` para mostrar clima, historia, dispositivos simulados y registrar la casa.
  * Si el backend no tiene ubicación, renderiza `HomeRegistration`, que llama a `lib/countriesNow` para listar países LATAM, estados/ciudades y traduce la ciudad seleccionada en coordenadas (`getPositionByCity`). Luego usa `saveLocation` para persistir lat/lon y guarda el estado en `localStorage`.
- `lib/devices` y `lib/weatherApi` se comunican con el microservicio configurado en `VITE_MICROSERVICES_URL` para obtener/operar dispositivos y dashboards de clima. `weatherHistory` usa la API pública `open-meteo.com`.

## 7. UI/estilos comunes

- `components/ui` (Buttons, grids, textos animados) se reutilizan en las pantallas de marketing y formularios.
- `components/common/Header` expone los botones de registro/login del home (`pages/Home`) y se coloca sobre los backgrounds `GradientBlinds` y `GridBeams`.
- La base CSS (`styles/index.css`) importa Tailwind, guarda un tema oscuro por defecto y aplica un filtro `invert()` cuando la clase `theme-light` está presente para evitar reescribir los cientos de utilidades de color.
- `lib/utils.cn` combina clases con `clsx` + `twMerge` para evitar estilos duplicados en componentes con variantes.

## 8. Variables de entorno y dependencias

| Variable | Uso principal |
| --- | --- |
| `VITE_BACKEND_URL` | Base URL para `/api/*`, login, registro, verificación y emisión de tokens. |
| `VITE_CHAT_API_BASE` | URL del microservicio de chat (conversaciones, mensajes, temas). |
| `VITE_MICROSERVICES_URL` | Base para los servicios de clima/dispositivos (`/api/dashboard`, `/api/devices`). |

- Dependencias clave: `react`, `react-dom`, `react-router-dom`, `axios` (backend + microservicios), `recharts` (gráficas de telemetría), `lucide-react`/`react-icons` (íconos), `framer-motion`/`gsap`/`motion`/`ogl` (animaciones de fondos). DevDeps: Vite + TypeScript + Tailwind + ESLint/Prettier (ver `frontend/package.json`).

- Scripts: `npm run dev` / `start` (Vite), `build` (TS + Vite), `type-check`, `format`, `preview`.

## 9. Siguientes pasos sugeridos

1. Documentar los endpoints del microservicio de chat y los datos que espera `chatApi` para que el equipo pueda reproducir la telemetry.
2. Mantener sincronizados los tokens en `localStorage:aura:pat` si la SPA agrega logout global (ya se limpian en `chatApi.clearChatToken`).
3. Ampliar `HomeAssistantPanel` con dashboards reales una vez que los microservicios de clima/devices estén disponibles; por ahora hay simulaciones que deben guardarse en `localStorage`.
