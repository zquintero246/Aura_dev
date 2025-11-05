Chat Service (Flask + Mongo + PostgreSQL)

Overview
- Validates Laravel Sanctum tokens against PostgreSQL (aura_main).
- Stores conversations and messages in MongoDB (aura_chat), linked by user_id from Postgres.

Endpoints
- POST /chat/start { title? } -> { id, title, created_at }
- POST /chat/message { conversation_id, content, role? } -> { ok, message }
- GET  /chat/history -> { conversations: [...] }
- GET  /chat/conversations/:id/messages -> { messages: [...] }

Environment
- DB_HOST, DB_PORT, DB_DATABASE, DB_USERNAME, DB_PASSWORD
- MONGO_URI (e.g. mongodb://aura_admin:aura_pass@host.docker.internal:27017)
- MONGO_DB (default aura_chat)
- CORS_ALLOW_ORIGINS (comma list for dev)

Run locally (Docker)
1) Ensure Mongo is up (see mongodb/docker-compose.yml)
2) docker build -t chat_api ./microservices/chat_service
3) docker run --rm -p 5060:5060 \
   -e DB_HOST=host.docker.internal -e DB_DATABASE=aura_main -e DB_USERNAME=aura_user -e DB_PASSWORD=aura_pass \
   -e MONGO_URI="mongodb://aura_admin:aura_pass@host.docker.internal:27017" -e MONGO_DB=aura_chat \
   chat_api

