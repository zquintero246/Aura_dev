# Documentación del backend de Aura (Laravel)

Este documento resume qué hace cada pieza importante del backend de `backend/`, cómo se conectan las APIs, qué datos persisten y cómo se maneja la verificación de correo.

## 1. Visión general

- El backend es una app Laravel monolítica con autenticación tradicional (`Auth::routes`) y rutas API que consumen el SPA y el microservicio de chat.
- Usa bases de datos relacionales (PostgreSQL/pgsql por defecto, con migraciones para `users`, `homes`, `personal_access_tokens`), y un canal adicional de MongoDB para almacenar conversaciones y mensajes históricos.
- El archivo `backend/.env` activa proveedores externos (OpenRouter, mailers, Socialite, Mongo) y define URLs/credenciales; se menciona abajo en la sección de variables.

## 2. Documentos clave

### 2.1 Rutas (`routes/web.php` y `routes/api.php`)

- `routes/web.php` expone vistas públicas, activa `Auth::routes(['verify' => true])`, define el popup de verificación (`/home` hacia `auth.verified-popup`), los endpoints OAuth (`/auth/{provider}`) y las rutas protegidas API (`api/*`) que consume la SPA. También mantiene la compatibilidad con Sanctum para tokens (`auth:sanctum`).
- `routes/api.php` mantiene solo dos rutas autenticadas para emitir/revocar tokens de Sanctum mediante `TokenController`.

### 2.2 Controladores de API

- `AuthController` (`app/Http/Controllers/Api/AuthController.php`): registra, inicia/cierra sesión, reenvía verificación y devuelve el perfil. Usa validaciones con `Validator`, registra al usuario con `Hash`, inicia sesión (`Auth::login`) y envía la notificación si el email no estaba verificado.
- `ChatController` (`app/Http/Controllers/Api/ChatController.php`): proxy a OpenRouter. Valida el cuerpo (`messages`, `model`, `conversationId`), inyecta un mensaje `system` por defecto, configura encabezados (Authorization, Referer, `X-Title`) y reintenta hasta tres veces ante códigos 4xx/5xx o respuestas vacías. Usa `MongoChatService` para registrar intercambios y titular automáticamente la conversación.
- `ConversationsController` y `ConversationController`: el primero lista/crea conversaciones en Mongo y requiere sesión o token; el segundo guarda pares (mensaje + respuesta) en la colección `conversations` (Mongo) y opcionalmente actualiza el campo `last_conversation_at` en PostgreSQL.
- `LocationController`: valida latitud/longitud, upserta datos en la tabla `homes` (modelo `HomeLocation`) y expone `store`/`me`. Usa conexión PostgreSQL por defecto.
- `ProfileController`: valida nombre/archivo de avatar, almacena el archivo en `storage/public/avatars` y actualiza `avatar_url` con `Storage::disk('public')->url(...)`.
- `TokenController`: mina/revoca tokens Sanctum llamados `chat` (columnas en `personal_access_tokens`), útiles para consumo de otras APIs o scripts.

### 2.3 Controladores de autenticación social (Socialite)

- `SocialiteController` maneja el redireccionamiento y callback para proveedores (`github`, `google`, etc.). Valida la configuración (`config/services.php`), fuerza la sesión `oauth_with_popup`, y una vez que el proveedor responde:
  - busca o crea el usuario (`findOrCreateUser`), enlazando `*_id` en `users`.
  - dispara verificación (Laravel la manda automáticamente gracias a `MustVerifyEmail`).
  - responde con HTML mínimo que postMessages al SPA y cierra la ventana; si el popup no existe, redirige a `${FRONTEND_URL}/verify-email`.

### 2.4 Modelos y persistencia

- `User` (`app/Models/User.php`): implementa `MustVerifyEmail`, `HasApiTokens`, su tabla principal (`users`) y define columnas `google_id`, `facebook_id`, etc. Relaciona con `Conversation`.
- `Conversation` Mongo (`Jenssegers\Mongodb`): colección `conversations`, sin timestamps automáticos, almacena `user_id`, `message`, `response`, `timestamp`.
- `HomeLocation`: tabla `homes` con PK de `user_id`, lat/lon con precisión decimal y timestamps; un registro por usuario.
- Las migraciones (`database/migrations/...`) crean la tabla `users`, extienden con campos sociales y avatar, crean `homes` y tokens personales.

### 2.5 Servicios auxiliares

- `MongoChatService` (`app/Services/MongoChatService.php`) encapsula `MongoDB\Driver\Manager`. Ofrece:
  1. `createConversation` y `listConversations`.
  2. `appendExchange` para insertar mensajes en `messages` (role + content).
  3. `updateTitleIfDefault` para reescribir la conversación si todavía tiene el título genérico.
  4. Manejo de credenciales (`MONGO_DB_*`) y detección de la extensión de Mongo.

### 2.6 Configuration files

- `config/auth.php`: define guard `web`, provider `users`, reset tokens, tiempo de confirmación.
- `config/database.php`: arranca con conexión `sqlite` por defecto pero declara `pgsql`, `mysql`, `mongodb` y Redis. Mongo se usa con `driver='mongodb'` y variables `DB_HOST_MONGO` etc.
- `config/services.php`: habilita Postmark/Resend/Ses, Socialite, Slack, y define `GOOGLE_...`, `GITHUB_...`.
- `config/mail.php`: la app usa `MAIL_MAILER` (por defecto `log` en local) y puede apuntar a SMTP/Ses/Postmark/Resend. El correo de verificación usa el mailer por omisión y comparte el `from`.

### 2.7 Vistas y flujo de verificación

- `resources/views/auth/verified-popup.blade.php`: vista minimalista que indica que el correo fue verificado, notificando al SPA con `window.opener.postMessage` y cerrando la ventana pasados 5 segundos. También se usa para el callback de Socialite cuando la verificación ocurre por OAuth.
- `resources/views/auth/verify.blade.php`: formulario tradicional de reenvío de correo (poco usado por SPA).

## 3. Flujo de APIs

1. **SPA de Aura** consume `api/auth/*` para registrar/login vía AJAX; la respuesta incluye `user` con `email_verified_at`.
2. `api/chat` (sin middleware) envía el historial al controlador, OpenRouter genera la respuesta y `MongoChatService` registra el turno.
3. Las rutas protegidas (`auth` o `auth:sanctum`) guardan conv., mensajes, ubicación y perfil; reintentan si el usuario pierde sesión.
4. Los tokens Sanctum se usan si el SPA (o un cliente) quiere usar autenticación basada en header `Authorization: Bearer`. El endpoint de mint revoca tokens anteriores antes de emitir uno nuevo.

## 4. Integración con OpenRouter y Mongo

- `ChatController::chat` valida la estructura, añade un prompt `system`, y construye una petición POST a `https://openrouter.ai/api/v1/chat/completions`.
- El cliente HTTP (`Illuminate\Support\Facades\Http`) configura `timeout`, `Authorization`, `Referer`, `X-Title` y opcionalmente deshabilita la verificación SSL.
- Reintenta ante errores de red/códigos 408/429/5xx o si `content` viene vacío, con backoff exponencial y respeta `Retry-After`.
- Antes de devolver la respuesta, llama a `$logExchange` para almacenar en Mongo los mensajes y titular automáticamente la conversación si aún tiene una etiqueta genérica.

## 5. Verificación de correo y notificaciones

- El modelo `User` implementa `MustVerifyEmail`, por lo que Laravel envía la notificación `VerifyEmail` cada vez que se crea un usuario sin verificar (registro normal o Socialite).
- Las rutas `Auth::routes(['verify' => true])` exponen `/email/verify/{id}/{hash}` (GET firmado) y `/email/verification-notification` (POST).
- Después de verificar, el middleware redirige a `route('home')`, que carga `auth.verified-popup`.
- También existe `api/auth/email/resend` para que la SPA solicite reenvío. Si el usuario ya está verificado, responde con mensaje entendible.

## 6. Variables de entorno críticas

| Clave                                                                   | Uso principal                                                      |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `APP_URL`, `FRONTEND_URL`                                               | Referencia en encabezados y redirecciones desde OAuth/Chat.        |
| `OPENROUTER_API_KEY`, `OPENROUTER_DEFAULT_MODEL`                        | Credenciales y modelo para el asistente.                           |
| `MONGO_DB_*` / `DB_HOST_MONGO`                                          | Conexión de `MongoChatService` y el modelo `Conversation`.         |
| `DB_CONNECTION`, `DB_HOST`, `DB_DATABASE`, `DB_USERNAME`, `DB_PASSWORD` | PostgreSQL/SQLite para `users`, `homes`, `personal_access_tokens`. |
| `MAIL_MAILER`, `MAIL_HOST`, etc.                                        | Transporte usado para enviar correos de verificación.              |
| `GOOGLE_CLIENT_ID`, `GITHUB_CLIENT_ID`, ...                             | Socialite; se usan en `config/services.php`.                       |

## 7. Siguientes pasos recomendados

1. Mantener actualizada la documentación `docs/` cuando se añadan nuevos endpoints o servicios (por ejemplo, nuevos proveedores de Socialite).
2. Verificar la disponibilidad de Mongo y los secrets en `env` antes de desplegar; si Mongo no está disponible, las rutas de conversación devuelven mensajes claros.
3. Si se necesita un token API aparte del SPA, la lógica de `TokenController` se puede extender para incluir scopes o expiraciones personalizadas.
