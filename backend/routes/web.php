<?php

use App\Http\Controllers\Auth\SocialiteController;

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\Api\AuthController as ApiAuthController;

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

// IMPORTANT:
// - Removed the old catch-all `/admin/*` forward to the SPA.
// - Do NOT forward unknown paths to React.
// - React Router handles `/verify-email` entirely on the frontend.
// - The only backend → frontend redirect now is to `${FRONTEND_URL}/verify-email`
//   inside the OAuth popup callback fallback (see SocialiteController@handleProviderCallback).
