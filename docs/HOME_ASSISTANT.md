# Home Assistant (AURA) – Guía Técnica Completa

Esta guía documenta el módulo Home Assistant de AURA: arquitectura, APIs, servicios, flujo de datos, ejecución local y despliegue. Incluye la integración de ubicación de usuario, clima, y un microservicio de dispositivos simulados con persistencia.

## Resumen

- Frontend: React + Vite. Pantalla principal: `HomeAssistantPanel`.
- Backend: Laravel (PHP). Persiste la ubicación del usuario en Postgres (`homes`).
- Microservicios: Flask (Python) con dos capacidades:
  - Clima (Open‑Meteo): `GET /api/dashboard` por lat/lon o `user_id`.
  - Dispositivos simulados (10 tipos): `GET /api/devices`, control `POST /api/devices/<id>/power|update`.
- Persistencia adicional: estado de dispositivos en `microservices/data/devices_state.json`.

---

## Arquitectura y Flujo

1) Registro de ubicación (Frontend → Backend)
- Página: `frontend/src/pages/Account/Settings.tsx` y/o flujo de “Configurar mi casa”.
- El usuario selecciona País → Departamento → Ciudad (Open‑Meteo geocoding directo desde FE).
- Se guardan país, ciudad y coordenadas en Laravel vía `POST /api/location`.
- Tabla principal `homes` en Postgres: `user_id (PK, varchar)`, `city`, `country`, `lat numeric(10,6)`, `lon numeric(10,6)`, timestamps.
- Modelo Laravel: `backend/app/Models/HomeLocation.php`.

2) Clima (Header en HomeAssistant)
- Componente: `frontend/src/pages/Chat/HomeAssistantPanel.tsx`.
- Obtiene `user_id` y consulta al backend para leer `homes` (`getMyLocation`).
- Llama al microservicio `GET /api/dashboard?lat=..&lon=..` para clima actual (Open‑Meteo):
  - Respuesta unificada: `{ temperature, humidity, condition, air_quality_index, source: 'open-meteo' }`.
- Muestra 3 “círculos” con iconos + valores (temperatura, humedad, condición) centrados en el header.

3) Dispositivos simulados
- Microservicio levanta 10 dispositivos (Zigbee coordinator, Z‑Wave controller, Aqara, Shelly, TP‑Link, Philips Hue, Ecobee, Chromecast, Sonos…) y ejecuta un hilo que simula cambios periódicos.
- Frontend muestra un grid de “cards” estilo Home Assistant con estado y subtítulo por tipo.
- Se puede “Agregar dispositivo” (plantillas) y abrir un modal de configuración por card; los cambios se envían al microservicio y se persisten.

---

## Repos y Rutas Relevantes

- Frontend
  - Home Assistant UI: `frontend/src/pages/Chat/HomeAssistantPanel.tsx`
  - API Frontend → Backend (Laravel): `frontend/src/lib/location.ts`
  - API Frontend → Microservicios (clima): `frontend/src/lib/weatherApi.ts`
  - API Frontend → Microservicios (dispositivos): `frontend/src/lib/devices.ts`
  - Geocodificación/país/estado/ciudad: `frontend/src/lib/countriesNow.ts`
  - Histórico para gráficas (Open‑Meteo): `frontend/src/lib/weatherHistory.ts`

- Backend (Laravel)
  - Modelo ubicación: `backend/app/Models/HomeLocation.php`
  - Controlador API: `backend/app/Http/Controllers/Api/LocationController.php`
  - Rutas: `backend/routes/web.php` (`/api/location`, `/api/location/me`)
  - DB: `backend/config/database.php` (usa conexión `pgsql` a `aura_main`)

- Microservicios (Flask)
  - App principal: `microservices/app.py`
  - Clima (Open‑Meteo u OpenWeather opcional): `microservices/services/weather_service.py`
  - Dispositivos simulados: `microservices/devices/*.py`
  - Persistencia de estado: `microservices/data/devices_state.json`

---

## Microservicio: Clima (Open‑Meteo)

- Endpoint: `GET /api/dashboard`
  - Parámetros admitidos:
    - `lat` y `lon` (preferido)
    - o `user_id` (lee coord. desde Postgres si `APP_DB_DRIVER=postgres` y vars DB configuradas)
  - Respuesta:
    ```json
    {
      "temperature": 27.4,
      "humidity": 68,
      "condition": "Partly cloudy",
      "air_quality_index": 42,
      "source": "open-meteo"
    }
    ```
  - En caso de error: `{ "error": "...", "details": { "status": 503, "body": "..." } }`

- Implementación
  - Llama a Open‑Meteo Forecast (current: temperature_2m, relative_humidity_2m, weather_code) y Air‑Quality (european_aqi).
  - Fallback opcional a OpenWeatherMap si `OPENWEATHERMAP_KEY` está presente.
  - Código: `microservices/services/weather_service.py`.

- Variables de entorno (microservicios)
  - `WEATHER_API_PROVIDER=open-meteo` (por defecto)
  - `OPENWEATHERMAP_KEY` (opcional, fallback)
  - Si `user_id` se usa: `APP_DB_DRIVER=postgres` y `DB_HOST/DB_PORT/DB_DATABASE/DB_USERNAME/DB_PASSWORD`.

---

## Microservicio: Dispositivos Simulados

- Endpoints
  - `GET /api/devices` → lista el estado actual de los 10 dispositivos
  - `POST /api/devices/<id>/power` con body `{ "action": "on|off|toggle" }` → enciende/apaga
  - `POST /api/devices/<id>/update` con body según tipo → aplica cambios en `update_config(...)` del dispositivo
  - `POST /api/devices/tick_all` → fuerza un “tick” inmediato (simulación)

- Simulación y persistencia
  - Hilo de fondo (`_bg_tick_loop`) invoca `simulate_tick()` de cada dispositivo cada 2s (ajustable con `DEVICES_TICK_INTERVAL`).
  - Estados persistidos a `microservices/data/devices_state.json` en cada cambio significativo.
  - IDs determinísticos para los 10 dispositivos (p. ej. `zigbee_coordinator`, `zwave_controller`…), de modo que el estado re-hidrate tras reinicios.

- Tipos soportados y campos destacados
  - `coordinator` (Zigbee): `network_channel`, `connected_devices`
  - `controller` (Z‑Wave): `region`, `node_count`
  - `contact_sensor` (Aqara Door/Window): `opened`, `battery`
  - `motion_sensor` (Aqara Motion): `motion_detected`, `lux`, `battery`, `occupancy_timeout_s`
  - `relay` (Shelly): `power_w`, `energy_wh`
  - `smart_plug` (TP‑Link): `power_w`, `voltage_v`
  - `light_bulb` (Philips Hue): `is_on`, `brightness`, `color_temp`
  - `thermostat` (Ecobee): `hvac_mode`, `target_c`, `current_c`, `fan_on`
  - `chromecast` (Google): `playback_state`, `app_name`, `volume`
  - `speaker` (Sonos): `playback_state`, `volume`, `muted`, `track`

---

## Frontend – HomeAssistantPanel

- Archivo principal: `frontend/src/pages/Chat/HomeAssistantPanel.tsx`
- Header
  - Ciudad centrada + 3 círculos con iconos y valores (temperatura, humedad, condición). Al pulsarlos se abre un modal con gráficas históricas (48h) usando Open‑Meteo.
- Grid de dispositivos
  - Cards responsivas (1–4 columnas). Cada card muestra icono, nombre y subtítulo por tipo.
  - Botón “Agregar dispositivo” abre un modal con 10 tipos disponibles; al agregar se inserta una card (simulada) en el grid.
  - Polling automático cada 5s a `/api/devices` para reflejar los ticks del microservicio.
- Modal de configuración de dispositivo
  - Click en una card abre un modal con controles por tipo (sliders, selects, toggles).
  - Acciones invocan `POST /power` o `POST /update` y los cambios persisten.

---

## Ejecución Local

1) Red Docker compartida (si no existe)
```bash
docker network create aura_network
```

2) Base de datos principal (Postgres)
```bash
cd postgresql
docker compose up -d
```

3) Microservicios
```bash
cd microservices
# Variables sugeridas (o usar docker-compose por defecto)
# WEATHER_API_PROVIDER=open-meteo
# OPENWEATHERMAP_KEY= (opcional)
# APP_DB_DRIVER=postgres (si usarás user_id)
# DB_HOST=aura_postgres DB_PORT=5432 DB_DATABASE=aura_main DB_USERNAME=aura_user DB_PASSWORD=aura_pass

docker compose up -d --build
```

4) Backend Laravel
```bash
cd backend
docker compose up -d
```

5) Frontend
- `.env` del frontend debe apuntar al backend y microservicio:
```env
VITE_BACKEND_URL=http://127.0.0.1:8000
VITE_MICROSERVICES_URL=http://127.0.0.1:5050
```
- Levantar dev server: `npm run dev` (en `frontend/`).

---

## API de Referencia (cURL)

- Clima (Open‑Meteo)
```bash
curl "http://localhost:5050/api/dashboard?lat=7.12539&lon=-73.1198"
```

- Dispositivos
```bash
# Listar
curl http://localhost:5050/api/devices

# Encender/apagar/toggle
curl -X POST http://localhost:5050/api/devices/philips_hue_bulb/power \
  -H "Content-Type: application/json" -d '{"action":"toggle"}'

# Actualizar configuración (ejemplos)
# Bombilla: brillo/color_temp
curl -X POST http://localhost:5050/api/devices/philips_hue_bulb/update \
  -H "Content-Type: application/json" -d '{"brightness":80, "color_temp":3500}'

# Termostato
curl -X POST http://localhost:5050/api/devices/ecobee_thermostat/update \
  -H "Content-Type: application/json" -d '{"target_c":23.5, "hvac_mode":"heat"}'
```

---

## Solución de Problemas

- `404 NOT FOUND` al llamar `/api/dashboard` o `/api/devices`
  - Asegura que el contenedor `weather_api` está arriba y mapeando puerto `5050:5050`.
  - Comprueba rutas en logs (`docker compose logs -f`).
  - Verifica `VITE_MICROSERVICES_URL`.

- Dispositivos no cambian
  - Revisa que el hilo de ticks esté activo (hay `before_request` que lo arranca en el primer request).
  - Ajusta `DEVICES_TICK_INTERVAL` para simular más rápido.

- Persistencia
  - El estado se guarda en `microservices/data/devices_state.json` (IDs fijos). Si borras este archivo, vuelven a valores por defecto.

- CORS
  - Flask usa `flask-cors` para permitir orígenes del frontend durante desarrollo.

---

## Notas de Seguridad (Dev/Prod)

- El microservicio de demo no aplica autenticación ni control de acceso. En producción:
  - Protege endpoints con OAuth/JWT o API keys.
  - Usa HTTPS y restringe CORS.
  - Aísla la red de contenedores y aplica rate‑limiting cuando corresponda.

---

## Roadmap

- Acciones rápidas en cards (switch/slider inline) además del modal.
- Grupos/rooms y escenas.
- Telemetría real por protocolo (MQTT/Zigbee2MQTT/Z-Wave JS) detrás de la misma API.
- Almacenamiento histórico en TSDB para gráficas avanzadas.

