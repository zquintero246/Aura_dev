<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;

class TokenController extends Controller
{
    // Crea un token Sanctum y lo devuelve como plainTextToken (id|secret)
    public function mint(Request $request)
    {
        $user = $request->user();

        if (!$user) {
            return response()->json(['message' => 'Unauthenticated.'], 401);
        }

        // Revoca tokens anteriores con nombre 'chat' y emite uno nuevo persistente
        $user->tokens()->where('name', 'chat')->delete();
        $token = $user->createToken('chat')->plainTextToken;

        return response()->json([
            'token' => $token,
            'type'  => 'Bearer',
        ]);
    }

    // Revoca los tokens llamados "chat" del usuario actual
    public function revoke(Request $request)
    {
        $user = $request->user();

        if (!$user) {
            return response()->json(['message' => 'Unauthenticated.'], 401);
        }

        $user->tokens()->where('name', 'chat')->delete();

        return response()->json(['revoked' => true]);
    }
}
