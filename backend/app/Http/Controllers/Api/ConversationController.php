<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\Conversation;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\Log;

class ConversationController extends Controller
{
    public function store(Request $request)
    {
        $user = $request->user();
        if (! $user) {
            return response()->json(['message' => 'Unauthenticated'], 401);
        }

        $data = $request->validate([
            'message' => ['required', 'string'],
            'response' => ['required', 'string'],
            'timestamp' => ['nullable', 'date'],
        ]);

        $timestamp = isset($data['timestamp']) ? new \DateTime($data['timestamp']) : now();

        $doc = Conversation::create([
            'user_id' => (string) $user->id,
            'message' => $data['message'],
            'response' => $data['response'],
            'timestamp' => $timestamp,
        ]);

        // Optional: update a field in Postgres without breaking if it doesn't exist
        try {
            if (Schema::hasColumn('users', 'last_conversation_at')) {
                $user->forceFill(['last_conversation_at' => now()])->save();
            }
        } catch (\Throwable $e) {
            Log::debug('Optional user update skipped', ['error' => $e->getMessage()]);
        }

        return response()->json([
            'message' => 'stored',
            'id' => (string) ($doc->_id ?? ''),
        ]);
    }
}

