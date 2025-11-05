# Aura_dev — Guía Completa (Windows y Ubuntu)

Este proyecto une:
- Backend principal en Laravel (PHP) con PostgreSQL para identidad/autenticación (aura_main).
- Microservicio de chat en Flask/Python que persiste conversaciones en MongoDB (aura_chat), enlazadas al user_id de PostgreSQL usando tokens Laravel Sanctum.
- Frontend en React/Vite.

Todos los servicios de datos comparten una red Docker común (`aura_network`) para resolver hostnames internos estables: `aura_postgres`, `aura_mongo`.

Puertos por defecto:
- PostgreSQL: `5432`
- MongoDB: `27017` (Mongo Express UI en `8081`)
- Chat API (Flask): `5060`
- Backend Laravel (Nginx en docker): `8080` (opcional; también puedes correr Laravel local en `8000`)

Estructura relevante:
- `backend/` (Laravel, Nginx y PHP-FPM dockerizado)
- `frontend/` (React/Vite)
- `postgresql/` (docker-compose con `aura_postgres`)
- `mongodb/` (docker-compose con `aura_mongo` y `aura_mongo_express`)
- `microservices/chat_service/` (Flask + Mongo + PG, valida tokens Sanctum)
- `docker-compose.yml` (raíz): declara la red global `aura_network`

Requisitos mínimos (Dev):
- Docker Desktop (Windows) o Docker Engine + Compose v2 (Ubuntu)
- Git
- Node.js LTS (18/20) + npm
- PHP 8.2+ y Composer (si ejecutas Laravel localmente; opcional si usas los contenedores de `backend/`)

Nota sobre secretos: los valores en `backend/.env` son de ejemplo. Ajusta claves y contraseñas antes de producción.

**Red Global**
- Ejecuta una vez para crear la red si no existe:
  - Windows (PowerShell): `docker network create aura_network`
  - Ubuntu: `docker network create aura_network`

**Bases De Datos**
- PostgreSQL
  - Ubicación: `postgresql/docker-compose.yml`
  - Comando: `cd postgresql && docker compose up -d`
  - Host interno para otros contenedores: `aura_postgres`
  - Credenciales por defecto: `DB=aura_main`, `USER=aura_user`, `PASS=aura_pass`
  - Volúmenes: `./data` (datos), `./init.sql` (seed opcional)

- MongoDB + Mongo Express
  - Ubicación: `mongodb/docker-compose.yml`
  - Comando: `cd mongodb && docker compose up -d`
  - Host interno: `aura_mongo`
  - Admin por defecto: `aura_admin` / `aura_pass`
  - UI: `http://localhost:8081`

**Chat Service (Flask)**
- Ubicación: `microservices/chat_service/`
- Env por defecto (en su `docker-compose.yml`):
  - `DB_HOST=aura_postgres`, `DB_DATABASE=aura_main`, `DB_USERNAME=aura_user`, `DB_PASSWORD=aura_pass`
  - `MONGO_URI=mongodb://aura_admin:aura_pass@aura_mongo:27017/aura_chat?authSource=admin`
  - `MONGO_DB=aura_chat`
  - `CORS_ALLOW_ORIGINS=http://127.0.0.1:5173,http://localhost:5173`
- Arranque:
  - `cd microservices/chat_service && docker compose up -d --build`
- Logs esperados al iniciar:
  - `Connected to MongoDB database: aura_chat`
  - `Indexes ensured successfully`

**Backend Laravel**
Tienes dos opciones. La integración con Sanctum funciona en ambos modos.

- Opción A: Ejecutar Laravel local (recomendado para dev rápido)
  - Requisitos: PHP 8.2+, Composer, PostgreSQL accesible en `localhost:5432` (publicado por Docker)
  - Pasos:
    - `cd backend`
    - `composer install`
    - Copia `.env` (si no existe) y ajusta:
      - `DB_CONNECTION=pgsql`
      - `DB_HOST=127.0.0.1`
      - `DB_PORT=5432`
      - `DB_DATABASE=aura_main`
      - `DB_USERNAME=aura_user`
      - `DB_PASSWORD=aura_pass`
      - `SANCTUM_STATEFUL_DOMAINS=localhost,127.0.0.1,localhost:5173,127.0.0.1:5173`
      - `FRONTEND_URL=http://127.0.0.1:5173`
      - (Opcional) Ajusta `MAIL_*` y claves de AI si las usas
    - `php artisan key:generate` (si aplica)
    - `php artisan migrate`
    - Servir:
      - `php artisan serve --host=127.0.0.1 --port=8000`
    - Backend quedará en `http://127.0.0.1:8000`

- Opción B: Ejecutar Laravel dockerizado (Nginx + PHP-FPM)
  - `cd backend && docker compose up -d`
  - Nginx expone `http://localhost:8080`
  - Nota: Este compose usa su propia red (`aura_net`). Para que el backend se conecte por hostname a `aura_postgres`/`aura_mongo`, puedes:
    - Usar los puertos publicados (DB_HOST=127.0.0.1 y Mongo en `localhost:27017`).
    - O modificar `backend/docker-compose.yml` para unir `app` y `nginx` a `aura_network` (externa).

Endpoints añadidos para tokens Sanctum (backend):
- POST `/api/auth/token` (sesión requerida): emite token `id|secret`.
- POST `/api/auth/token/revoke` (opcional): revoca tokens llamados `chat`.
  - Implementación: `backend/app/Http/Controllers/Api/TokenController.php:1`

**Frontend (React/Vite)**
- Requisitos: Node LTS + npm
- Pasos:
  - `cd frontend`
  - `npm install`
  - Crea `.env.local` con:
    - `VITE_BACKEND_URL=http://127.0.0.1:8000` (opción A) o `http://127.0.0.1:8080` (opción B)
    - `VITE_CHAT_API_BASE=http://127.0.0.1:5060`
  - `npm run dev`
  - Abre `http://127.0.0.1:5173`

Flujo de autenticación y chat:
- Inicia sesión en la app (Laravel).
- El frontend llama `POST /api/auth/token` y guarda el token en `localStorage` (`aura:pat`).
- Las llamadas al Chat Service usan `Authorization: Bearer <token>`.
- Chat Service valida contra PostgreSQL y guarda conversaciones/mensajes en Mongo con `user_id` real.

**Pruebas de conectividad (Docker)**
- Ver contenedores: `docker ps`
- Desde Flask:
  - `docker exec -it chat_api ping -c 2 aura_mongo`
  - `docker exec -it chat_api ping -c 2 aura_postgres`
- Ver Mongo Express: `http://localhost:8081`

**Windows — Pasos Rápidos**
- Instala Docker Desktop y habilita WSL2.
- PowerShell:
  - `git clone <repo> Aura_dev && cd Aura_dev`
  - `docker network create aura_network`
  - `cd mongodb && docker compose up -d`
  - `cd ../postgresql && docker compose up -d`
  - `cd ../microservices/chat_service && docker compose up -d --build`
  - Backend local: `cd ../../backend && composer install && php artisan migrate && php artisan serve`
  - Frontend: `cd ../frontend && npm i && npm run dev`

**Ubuntu Server — Pasos Rápidos**
- Instala Docker Engine y Compose v2 (plugin):
  - `sudo apt-get update && sudo apt-get install -y ca-certificates curl gnupg`
  - Instala Docker (ver docs oficiales) y añade tu usuario al grupo `docker`.
- Terminal:
  - `git clone <repo> Aura_dev && cd Aura_dev`
  - `docker network create aura_network`
  - `cd mongodb && docker compose up -d`
  - `cd ../postgresql && docker compose up -d`
  - `cd ../microservices/chat_service && docker compose up -d --build`
  - Backend (local o dockerizado). Si local: instala PHP/Composer y corre migraciones.
  - Frontend: instala Node y corre `npm run build` o `npm run dev` según necesidad.

**Solución De Problemas**
- Chat Service no autentica (401): verifica que el frontend recibió token de `/api/auth/token` y que `aura:pat` existe en localStorage.
- Error PG desde Flask: revisa env `DB_HOST=aura_postgres` y que `aura_postgres` esté en `docker ps` y en la red `aura_network`.
- Error Mongo desde Flask: valida `MONGO_URI` y logs de inicio (debe imprimir la conexión e índices).
- CORS: agrega orígenes del frontend a `CORS_ALLOW_ORIGINS` (Flask) y a `backend/config/cors.php`.

**Seguridad (recomendaciones)**
- Cambia usuarios/contraseñas por defecto para Postgres/Mongo.
- Revoca tokens con `/api/auth/token/revoke` al cerrar sesión si deseas.
- No subas `.env` con secretos a repos públicos.

**Referencias De Código**
- Chat Service (Flask): `microservices/chat_service/app.py:1`
- Auth Sanctum (Flask→PG): `microservices/chat_service/services/auth.py:1`
- Mongo CRUD: `microservices/chat_service/services/mongo.py:1`
- TokenController (Laravel): `backend/app/Http/Controllers/Api/TokenController.php:1`
- Frontend Chat API: `frontend/src/lib/chatApi.ts:1`

