<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class HomeLocation extends Model
{
    protected $connection = 'pgsql';
    protected $table = 'homes';
    protected $primaryKey = 'user_id';
    public $incrementing = false;
    protected $keyType = 'string';
    public $timestamps = true;

    protected $fillable = [
        'user_id', 'city', 'country', 'lat', 'lon',
    ];
}

