<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\Api\TokenController;

Route::middleware('auth:sanctum')->group(function () {
    Route::post('/auth/token', [TokenController::class, 'mint']);
    Route::post('/auth/token/revoke', [TokenController::class, 'revoke']);
});
