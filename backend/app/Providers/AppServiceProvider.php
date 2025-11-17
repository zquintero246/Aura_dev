<?php

namespace App\Providers;

use App\Support\SafeJson;
use Illuminate\Support\Facades\Blade;
use Illuminate\Support\ServiceProvider;

class AppServiceProvider extends ServiceProvider
{
    /**
     * Register any application services.
     */
    public function register(): void
    {
        //
    }

    /**
     * Bootstrap any application services.
     */
    public function boot(): void
    {
        //
        Blade::directive('json', function ($expression) {
            $expression = $expression ?: 'null';

            return "<?php echo \\" . SafeJson::class . "::encode($expression); ?>";
        });
    }
}