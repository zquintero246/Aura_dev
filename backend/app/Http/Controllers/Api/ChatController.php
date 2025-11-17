<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use App\Services\MongoChatService;

class ChatController extends Controller
{
    public function chat(Request $request)
    {
        $validated = $request->validate([
            'messages' => ['required', 'array'],
            'messages.*.role' => ['required', 'string'],
            'messages.*.content' => ['required', 'string'],
            'model' => ['nullable', 'string'],
            'temperature' => ['nullable', 'numeric'],
            'conversationId' => ['nullable', 'string'],
        ]);

        $model = $validated['model'] ?? env('OPENROUTER_DEFAULT_MODEL', 'google/gemini-2.0-flash-exp:free');
        $temperature = $validated['temperature'] ?? 0.7;

        \Log::info('Chat provider selection', ['model' => $model, 'provider' => 'openrouter']);

        // Extract last user message (for logging)
        $lastUserText = '';
        for ($i = count($validated['messages']) - 1; $i >= 0; $i--) {
            $m = $validated['messages'][$i];
            if (($m['role'] ?? '') === 'user') { $lastUserText = (string) ($m['content'] ?? ''); break; }
        }

        // Helper to log exchange into Mongo (no-op if Mongo not available) and
        // opportunistically auto-title the conversation based on the first exchange.
        $logExchange = function (string $assistantContent) use ($request, $validated, $lastUserText) {
            try {
                /** @var MongoChatService $mongo */
                $mongo = app(MongoChatService::class);
                if (!$mongo || !$mongo->available()) return;
                $convId = (string) ($validated['conversationId'] ?? $request->input('conversationId') ?? '');
                $userId = optional($request->user())->id;
                if ($convId && $userId && $lastUserText !== '') {
                    $mongo->appendExchange($convId, $lastUserText, $assistantContent, $userId);
                    // Generate a concise title (max 6 words) from the first user + assistant turn
                    $autoTitle = $this->generateAutoTitle($lastUserText, (string) $assistantContent);
                    if ($autoTitle !== '') {
                        $mongo->updateTitleIfDefault($convId, $userId, $autoTitle);
                    }
                }
            } catch (\Throwable $e) { /* ignore logging errors */ }
        };

        try {
            $openrouterKey = env('OPENROUTER_API_KEY');
            if (! $openrouterKey) {
                return response()->json(['message' => 'OPENROUTER_API_KEY not configured', 'provider' => 'openrouter', 'model' => $model], 500);
            }

            $messages = $validated['messages'];
            $hasSystem = false;
            foreach ($messages as $msg) {
                if (($msg['role'] ?? '') === 'system') {
                    $hasSystem = true;
                    break;
                }
            }
            if (! $hasSystem) {
                array_unshift($messages, [
                    'role' => 'system',
                    'content' => 'Eres un asistente útil y preciso.',
                ]);
            }

            $client = Http::withHeaders([
                'Authorization' => 'Bearer ' . $openrouterKey,
                'Content-Type' => 'application/json',
                'HTTP-Referer' => env('APP_URL', 'https://aura-dev.local'),
                'X-Title' => env('APP_NAME', 'Aura'),
            ])->timeout(45)->connectTimeout(10);
            if (env('AI_VERIFY_SSL', 'true') !== 'true') {
                $client = $client->withOptions(['verify' => false]);
            }

            $endpoint = 'https://openrouter.ai/api/v1/chat/completions';

            $attempt = 0;
            $maxAttempts = 3;
            $data = null;
            $content = '';
            $resp = null;
            do {
                $resp = $client->asJson()->post($endpoint, [
                    'model' => $model,
                    'messages' => $messages,
                ]);
                if ($resp->failed()) {
                    \Log::warning('OpenRouter API failed', [
                        'status' => $resp->status(),
                        'body' => $resp->body(),
                    ]);
                } else {
                    $data = $resp->json();
                    $content = $data['choices'][0]['message']['content'] ?? '';
                    if (is_string($content) && $content !== '') {
                        break;
                    }
                }
                $attempt++;
                $retryAfterHeader = isset($resp) ? (int)($resp->header('Retry-After') ?? 0) : 0;
                $shouldRetry = isset($resp) && (in_array($resp->status(), [408, 409, 425, 429, 500, 502, 503, 504]) || $content === '');
                $sleepMs = $retryAfterHeader > 0 ? ($retryAfterHeader * 1000) : (int)(pow(2, $attempt) * 250);
                if ($shouldRetry && $attempt < $maxAttempts) {
                    usleep($sleepMs * 1000);
                } else {
                    break;
                }
            } while ($attempt < $maxAttempts);

            if (!is_string($content) || $content === '') {
                $status = isset($resp) ? $resp->status() : 502;
                $retryAfter = isset($resp) ? (int)($resp->header('Retry-After') ?? 0) : 0;
                return response()->json([
                    'message' => 'OpenRouter returned no content',
                    'code' => 'no_content',
                    'retryable' => true,
                    'provider' => 'openrouter',
                    'model' => $model,
                    'retryAfter' => $retryAfter,
                    'raw' => isset($resp) ? $resp->json() : null,
                ], $status);
            }

            $logExchange($content);
            return response()->json([
                'content' => $content,
                'raw' => $data,
            ]);
        } catch (\Throwable $e) {
            \Log::error('Chat request exception', ['error' => $e->getMessage()]);
            $isTimeout = stripos($e->getMessage(), 'timeout') !== false;
            return response()->json([
                'message' => $isTimeout ? 'Upstream timeout' : 'Chat request failed',
                'code' => $isTimeout ? 'timeout' : 'unexpected',
                'retryable' => $isTimeout,
                'error' => $e->getMessage(),
                'provider' => 'openrouter',
                'model' => $model,
            ], $isTimeout ? 504 : 500);
        }
    }

    /**
     * Build a short, descriptive title from the first exchange.
     * Rules:
     *  - Max 6 words
     *  - No unnecessary punctuation at the ends
     *  - Avoid generic titles; derive from user's first message
     */
    private function generateAutoTitle(string $userText, string $assistantText = ''): string
    {
        $t = trim(preg_replace('/\s+/u', ' ', (string) $userText));
        if ($t === '') return '';

        // Remove leading Spanish inverted question/exclamation marks and trailing punctuation
        $t = preg_replace('/^[¿¡\s]+/u', '', $t);
        $t = preg_replace('/[\s\.?¡!¿,;:]+$/u', '', $t);

        $lower = mb_strtolower($t, 'UTF-8');

        // Common openings -> transform into concise imperatives or patterns
        $replacements = [
            // Ideas / brainstorming
            '/^dame\s+ideas\s+para\s+/u' => 'Ideas para ',
            '/^ideas\s+para\s+/u' => 'Ideas para ',
            // How to ...
            '/^(como|c\u00f3mo)\s+hacer\s+/u' => 'Hacer ',
            '/^(como|c\u00f3mo)\s+crear\s+/u' => 'Crear ',
            '/^(como|c\u00f3mo)\s+hago\s+/u' => 'Crear ',
            '/^(como|c\u00f3mo)\s+puedo\s+/u' => '',
            // Intent/statements
            '/^quiero\s+/u' => '',
            '/^quisiera\s+/u' => '',
            '/^necesito\s+/u' => '',
            '/^por\s+favor\s+/u' => '',
        ];

        foreach ($replacements as $pattern => $subst) {
            if (preg_match($pattern, $lower)) {
                // Replace based on original casing by slicing the original string
                $m = [];
                if (preg_match($pattern, $t, $m)) {
                    $t = preg_replace($pattern, $subst, $t);
                } else {
                    $t = preg_replace($pattern, $subst, $t);
                }
                break;
            }
        }

        // Take the first sentence
        $parts = preg_split('/(?<=[\.!?])\s+/u', $t);
        $first = is_array($parts) && count($parts) ? trim($parts[0]) : $t;
        $first = preg_replace('/[\s\.?¡!¿,;:]+$/u', '', $first);

        // Limit to 6 words
        $words = preg_split('/\s+/u', $first);
        if (is_array($words) && count($words) > 6) {
            $first = implode(' ', array_slice($words, 0, 6));
        }

        // Ensure first letter uppercase (preserve acronyms like API, Node.js)
        if ($first !== '') {
            $first = mb_strtoupper(mb_substr($first, 0, 1, 'UTF-8'), 'UTF-8') . mb_substr($first, 1, null, 'UTF-8');
        }

        // Avoid fallback generic titles
        $generic = [
            'nueva conversacion', 'nueva conversación', 'nuevo chat', 'conversacion con ia', 'charla general', 'chat general'
        ];
        $cmp = mb_strtolower($first, 'UTF-8');
        foreach ($generic as $g) {
            if ($cmp === $g) return '';
        }

        return trim($first);
    }
}
