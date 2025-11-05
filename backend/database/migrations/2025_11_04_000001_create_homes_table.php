<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void
    {
        Schema::create('homes', function (Blueprint $table) {
            // Using string to be compatible with microservice TEXT user_id
            $table->string('user_id')->primary();
            $table->string('city');
            $table->string('country');
            // Use decimals to keep reasonable precision on PG; microservice uses double
            $table->decimal('lat', 10, 6);
            $table->decimal('lon', 10, 6);
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('homes');
    }
};

