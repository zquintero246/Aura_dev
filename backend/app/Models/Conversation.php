<?php

namespace App\Models;

use Jenssegers\Mongodb\Eloquent\Model as Eloquent;

class Conversation extends Eloquent
{
    protected $connection = 'mongodb';
    protected $collection = 'conversations';

    public $timestamps = false; // we manage timestamp manually as requested

    protected $fillable = [
        'user_id',
        'message',
        'response',
        'timestamp',
    ];

    protected $casts = [
        'timestamp' => 'datetime',
    ];
}

