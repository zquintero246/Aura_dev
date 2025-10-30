<?php

return [
    'paths' => [
        'api/*',
        'sanctum/csrf-cookie',
        'register',
        'login',
        'logout',
        'email/*',
    ],

    'allowed_methods' => ['*'],

    'allowed_origins' => [
        env('FRONTEND_URL', 'http://localhost:3000'),
        'http://localhost:4028',
        'http://127.0.0.1:4028',
    ],

    'allowed_origins_patterns' => [],

    'allowed_headers' => ['*'],

    'exposed_headers' => [],

    'max_age' => 0,

    'supports_credentials' => true,
];
