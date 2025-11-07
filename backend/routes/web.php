<?php

use App\Http\Controllers\Auth\SocialiteController;

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\Api\AuthController as ApiAuthController;
use App\Http\Controllers\Api\ChatController as ApiChatController;
use App\Http\Controllers\Api\ConversationsController as ApiConversationsController;
use App\Http\Controllers\Api\ConversationController as ApiConversationController;
use App\Http\Controllers\Api\LocationController as ApiLocationController;

Route::get('/', function () {
    return view('welcome');
});

// Enable Laravel's native email verification routes.
// Public signed endpoint used in emails: `/email/verify/{id}/{hash}` (HTTP GET, signed).
// When a user clicks that link, Laravel marks the email as verified via the built-in middleware.
Auth::routes(['verify' => true]);

// After email verification, Laravel redirects to route('home').
// Serve a minimal page that confirms verification and attempts to auto-close.
Route::get('/home', function () {
    return view('auth.verified-popup');
})->name('home');

Route::group(['prefix' => 'auth'], function () {
    Route::get('{provider}/redirect', [SocialiteController::class, 'redirectToProvider'])->name('social.redirect');
    Route::get('{provider}/callback', [SocialiteController::class, 'handleProviderCallback'])->name('social.callback');
});

// --- Minimal JSON API for SPA authentication ---
Route::group(['prefix' => 'api/auth'], function () {
    Route::post('register', [ApiAuthController::class, 'register']);
    Route::post('login', [ApiAuthController::class, 'login']);
    Route::post('logout', [ApiAuthController::class, 'logout']);
    Route::get('me', [ApiAuthController::class, 'me']);
    Route::post('email/resend', [ApiAuthController::class, 'resendVerification']);
});

// AI Chat endpoint
Route::post('api/chat', [ApiChatController::class, 'chat']);

// Conversations (require session auth)
Route::middleware('auth')->group(function () {
    Route::get('api/conversations', [ApiConversationsController::class, 'index']);
    Route::post('api/conversations', [ApiConversationsController::class, 'store']);
    // Message+response log endpoint (session auth)
    Route::post('api/conversations/log', [ApiConversationController::class, 'store']);

    // Location endpoints (session auth)
    Route::post('api/location', [ApiLocationController::class, 'store']);
    Route::get('api/location/me', [ApiLocationController::class, 'me']);
});

// Token auth (Sanctum) variant without breaking existing
Route::middleware('auth:sanctum')->group(function () {
    Route::post('api/conversations/log', [ApiConversationController::class, 'store']);

    // Location endpoints (Sanctum)
    Route::post('api/location', [ApiLocationController::class, 'store']);
    Route::get('api/location/me', [ApiLocationController::class, 'me']);
});

// IMPORTANT:
// - Removed the old catch-all `/admin/*` forward to the SPA.
// - Do NOT forward unknown paths to React.
// - React Router handles `/verify-email` entirely on the frontend.
// - The only backend → frontend redirect now is to `${FRONTEND_URL}/verify-email`
//   inside the OAuth popup callback fallback (see SocialiteController@handleProviderCallback).
