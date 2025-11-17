# Suite Cypress — Aura Dev

## 1. Cómo correr la suite

- `npm install` (desde la carpeta `frontend/`) si aún no se instalaron dependencias.
- `npx cypress open` inicia la interfaz visual; ejecuta suites en la carpeta `cypress/e2e/`.
- `npx cypress run` corre todas las pruebas en modo headless y genera resultados en la consola.
- Los ficheros TypeScript usan `tsconfig.json` para resolver módulos; no se requiere compilación adicional.

## 2. Cómo correr headless

- `npx cypress run --headless --browser chrome` ejecuta todas las suites sin UI.  
- También se puede usar `npm run test:e2e` si se añade un script que invoque `cypress run`.

## 3. Estructura del proyecto de pruebas

```
cypress/
├── e2e/
│   ├── auth/          # Login, registro, verificación de email
│   ├── chat/          # Panel de chat, telemetría y validaciones
│   ├── api/           # Endpoints del backend y microservicios
│   ├── flows/         # Flujos completos end-to-end
│   └── ui/            # Componentes y marketing básicos
├── fixtures/          # Datos reales/mocked: user, dashboard, devices, home
└── support/
    ├── commands.ts     # Comandos personalizados (login/logout, mocks, helpers)
    └── e2e.ts         # Configuración global + interceptos
```

## 4. Suites y mocks

- **auth/**: valida la navegación de `/login`, `/register` y `/verify-email`, incluyendo respuestas correctas e incorrectas.  
- **chat/**: el chat panel usa `cy.interceptApi()` para preparar autenticación, token y chats, luego verifica envíos, errores y telemetría.  
- **api/**: cada endpoint (`/chat/*`, `/api/dashboard`, `/api/devices`) tiene tests 200/400/500/missing-fields/delayed.  
- **flows/**: cubre login → chat + logout, haciendo uso de `cy.apiPost` y los intercepts globales.  
- **ui/**: comprueba que los componentes marketing básicos se renderizan correctamente.  
- Todos los tests consumen fixtures conteniendo datos reales y reutilizan comandos para mantener el SRP/ISP (SOLID) en la definición del suite.

## 5. Cómo funcionan los mocks

- `cy.interceptApi()` configura: auth (login/register/me/token/logout/email), profile, location, dashboard, dispositivos y chat.  
- `cy.mockPrediction`, `cy.createHome` y `cy.apiPost` permiten adaptar mocks por prueba sin volver a escribir intercepts.  
- Todos los mocks son idempotentes y vuelven a cargar fixtures JSON, manteniendo el principio de `Open/Closed` (SOLID) y la responsabilidad única (SRP).

## 6. Extensión del suite

1. Añadir un nuevo fixture en `cypress/fixtures/`.  
2. Crear un interceptor dentro de `support/commands.ts` si el endpoint requiere lógica adicional.  
3. Escribir un nuevo spec `.cy.ts` bajo la carpeta adecuada (`auth`, `chat`, `api`, `flows` o `ui`).  
4. Importar `cy.interceptApi()` para mantener los mocks transversales o usar `cy.apiPost` para llamadas directas al backend.

## 7. Validación, SOLID y GRASP

- **Validación**: todas las pruebas cubren inputs válidos, inválidos y errores (400/422/500) para reflejar reglas definidas en frontend/backend (`auth.ts`, `chatApi.ts`).  
- **SOLID**: los commands (`commands.ts`) siguen SRP (cada helper solo hace su tarea), OCP (nuevos endpoints se añaden sin modificar pruebas existentes), ISP (solo se expone lo necesario) y DIP (tests dependen de capas abstractas como intercepts y fixtures).  
- **GRASP**: el `Creator` se refleja en `cy.interceptApi()` que crea los objetos necesarios, `Controller` en cada spec que coordina mocks + acciones y `Pure Fabrication` al separar lógica de intercepts de las pruebas reales.

## 8. Guía para nuevos contribuyentes

1. Revisa `support/commands.ts` antes de modificar tests: los helpers están centralizados para mantener consistencia.  
2. Usa los fixtures existentes y actualízalos con datos reales de la app (ej: `user`, `dashboard`, `device`).  
3. Documenta nuevos tests en este README y asegura que cada spec tiene un objetivo claro (Validación de reglas, flujo completo, endpoint).  
4. Ejecuta `npx cypress run` antes de hacer push para garantizar que los mocks siguen funcionando tras cambios en la UI o backend.
