# Diseño orientado a objetos

Este documento señala las responsabilidades médicas, patrones y la estructura MVC que navegan entre el backend Laravel y el frontend React. Cada sección apunta al archivo relevante, muestra un fragmento clave y ofrece un botón/link directo a su ubicación en GitHub.

## Principios aplicados

### Single Responsibility
- **Ubicación**: `backend/app/Http/Controllers/Api/LocationController.php`  
- **Qué hace**: el controlador valida los datos entrantes, delega el guardado a `HomeLocation` y retorna solo la vista JSON de éxito o error.  
- **Fragmento**:
  ```php
      public function store(Request $request)
      {
          $validated = $request->validate([...]);
          $user = $request->user();
          if (!$user) return response()->json([...], 401);

          $loc = HomeLocation::updateOrCreate([...], $payload);
          return response()->json([...], 201);
      }
  ```
- **Ver en GitHub**: [Ver en GitHub](https://github.com/zquintero246/Aura_dev/blob/main/backend/app/Http/Controllers/Api/LocationController.php#L12-L49)

### Open/Closed
- **Ubicación**: `backend/app/Http/Controllers/Api/ChatController.php`  
- **Qué hace**: el controlador escoge el modelo por configuración (`OPENROUTER_DEFAULT_MODEL`), agrega encabezados comunes y reintenta sin tocar la lógica cuando se expande el proveedor de IA.  
- **Fragmento**:
  ```php
      $model = $validated['model'] ?? env('OPENROUTER_DEFAULT_MODEL', 'google/gemini-2.0-flash-exp:free');

      $client = Http::withHeaders([...])
          ->timeout(45)
          ->connectTimeout(10);
  ```
- **Ver en GitHub**: [Ver en GitHub](https://github.com/zquintero246/Aura_dev/blob/main/backend/app/Http/Controllers/Api/ChatController.php#L20-L90)

### Dependency Injection
- **Ubicación**: `backend/app/Http/Controllers/Api/ConversationsController.php`  
- **Qué hace**: Laravel inyecta `MongoChatService` directamente en los métodos `index`/`store`, manteniendo el controlador independiente de la implementación concreta.  
- **Fragmento**:
  ```php
      public function index(Request $request, MongoChatService $mongo)
      {
          $list = $mongo->listConversations($user->id);
          return response()->json(['conversations' => $list]);
      }
  ```
- **Ver en GitHub**: [Ver en GitHub](https://github.com/zquintero246/Aura_dev/blob/main/backend/app/Http/Controllers/Api/ConversationsController.php#L11-L43)

## Patrones utilizados

### Factory
- **Ubicación**: `backend/app/Http/Controllers/Api/ChatController.php`  
- **Qué hace**: el helper `$logExchange` usa `app(MongoChatService::class)` como fábrica para obtener el servicio que administra conversaciones/cronologías, permitiendo cambiar la implementación sin alterar el flujo principal.  
- **Fragmento**:
  ```php
      $logExchange = function (string $assistantContent) {
          /** @var MongoChatService $mongo */
          $mongo = app(MongoChatService::class);
          if (!$mongo || !$mongo->available()) return;
          $mongo->appendExchange(...);
      };
  ```
- **Ver en GitHub**: [Ver en GitHub](https://github.com/zquintero246/Aura_dev/blob/main/backend/app/Http/Controllers/Api/ChatController.php#L35-L52)

### Adapter
- **Ubicación**: `frontend/src/lib/chat.ts`  
- **Qué hace**: adapta la interfaz de `api` y la persistencia Mongo para que el resto de la UI trabaje con `ChatMessage`/`ChatSuccess` homogéneos; añade títulos automáticos y envía eventos para mantener sincronizado el dashboard.  
- **Fragmento**:
  ```ts
  export async function chat(...): Promise<ChatSuccess> {
    ...
    const res = await api.post('/api/chat', { messages, model, conversationId });
    return res.data as ChatSuccess;
  }
  ```
- **Ver en GitHub**: [Ver en GitHub](https://github.com/zquintero246/Aura_dev/blob/main/frontend/src/lib/chat.ts#L15-L66)

### Observer
- **Ubicación**: `frontend/src/pages/Chat/ChatPanel.tsx` y `frontend/src/pages/Chat/AppLayout.tsx`  
- **Qué hace**: `ChatPanel` dispara eventos (`aura:conversation:realized`) usando `window.dispatchEvent`, y `AppLayout` (más `ChatPage`) los escucha para actualizar listas sin acoplarse directamente.  
- **Fragmento**:
  ```ts
  window.dispatchEvent(
    new CustomEvent('aura:conversation:realized', {
      detail: { tempId: conversationId, newId: conv.id, title: conv.title },
    })
  );
  ```
- **Ver en GitHub**: [Ver en GitHub](https://github.com/zquintero246/Aura_dev/blob/main/frontend/src/pages/Chat/ChatPanel.tsx#L155-L170)  
- **Oyentes**: [AppLayout](https://github.com/zquintero246/Aura_dev/blob/main/frontend/src/pages/Chat/AppLayout.tsx#L330-L344)

### MVC
- **Ubicación**: front React `AppLayout` + backend `routes/web.php` y controladores API.  
- **Qué hace**: las rutas en `routes/web.php` llaman a controladores (`ChatController`, `LocationController`, `ConversationsController`) que gestionan la lógica y exponen JSON; el frontend React (`AppLayout`, `ChatPanel`, `ChatPage`, etc.) actúa como Vista conectada por `Fetch` y Redux/estado local.  
- **Fragmento**:
  ```php
  Route::post('api/chat', [ApiChatController::class, 'chat']);
  Route::post('api/location', [ApiLocationController::class, 'store']);
  ```
  ```tsx
  export default function AppLayout() {
    ...
    const [recent, setRecent] = useState<Conversation[]>([]);
    ...
  }
  ```
- **Ver en GitHub**: [Routes](https://github.com/zquintero246/Aura_dev/blob/main/backend/routes/web.php#L3-L55) y [AppLayout](https://github.com/zquintero246/Aura_dev/blob/main/frontend/src/pages/Chat/AppLayout.tsx#L1-L40)
