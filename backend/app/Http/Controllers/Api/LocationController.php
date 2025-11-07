<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use App\Models\HomeLocation;

class LocationController extends Controller
{
    public function store(Request $request)
    {
        $validated = $request->validate([
            'country' => ['required', 'string', 'max:100'],
            'city' => ['required', 'string', 'max:120'],
            'latitude' => ['required', 'numeric', 'between:-90,90'],
            'longitude' => ['required', 'numeric', 'between:-180,180'],
        ]);

        $user = $request->user();
        if (!$user) {
            return response()->json(['message' => 'Unauthorized'], 401);
        }

        // Upsert en tabla principal `homes` usando user_id como PK
        $payload = [
            'user_id' => (string) $user->id,
            'country' => $validated['country'],
            'city' => $validated['city'],
            'lat' => (float) $validated['latitude'],
            'lon' => (float) $validated['longitude'],
        ];

        $loc = HomeLocation::updateOrCreate([
            'user_id' => (string) $user->id,
        ], $payload);

        return response()->json([
            'message' => 'Location saved',
            'data' => [
                'user_id' => $loc->user_id,
                'country' => $loc->country,
                'city' => $loc->city,
                'latitude' => (float) $loc->lat,
                'longitude' => (float) $loc->lon,
                'created_at' => $loc->created_at,
                'updated_at' => $loc->updated_at,
            ],
        ], 201);
    }

    public function me(Request $request)
    {
        $user = $request->user();
        if (!$user) {
            return response()->json(['message' => 'Unauthorized'], 401);
        }

        $row = HomeLocation::find((string) $user->id);

        if (!$row) {
            return response()->json(['data' => null]);
        }

        return response()->json([
            'data' => [
                'user_id' => $row->user_id,
                'country' => $row->country,
                'city' => $row->city,
                'latitude' => (float) $row->lat,
                'longitude' => (float) $row->lon,
                'created_at' => $row->created_at,
                'updated_at' => $row->updated_at,
            ],
        ]);
    }
}
