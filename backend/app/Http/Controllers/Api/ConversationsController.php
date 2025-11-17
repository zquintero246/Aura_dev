<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Services\MongoChatService;
use Illuminate\Http\Request;

class ConversationsController extends Controller
{
    public function index(Request $request, MongoChatService $mongo)
    {
        $user = $request->user();
        if (!$user) {
            return response()->json(['message' => 'Unauthenticated'], 401);
        }

        if (!$mongo->available()) {
            return response()->json(['conversations' => [], 'message' => 'Mongo not available'], 200);
        }

        $list = $mongo->listConversations($user->id);
        return response()->json(['conversations' => $list]);
    }

    public function store(Request $request, MongoChatService $mongo)
    {
        $user = $request->user();
        if (!$user) {
            return response()->json(['message' => 'Unauthenticated'], 401);
        }

        if (!$mongo->available()) {
            return response()->json(['message' => 'Mongo not available'], 503);
        }

        $title = (string) ($request->input('title') ?? 'Nueva conversación');
        $id = $mongo->createConversation($user->id, $title);
        if (!$id) {
            return response()->json(['message' => 'Failed to create conversation'], 500);
        }
        return response()->json(['id' => $id, 'title' => $title]);
    }
}

