<?php

namespace App\Models;

use Jenssegers\Mongodb\Eloquent\Model;

class Message extends Model
{
    protected $connection = 'mongodb';
    protected $collection = 'messages';

    protected $fillable = [
        'conversation_id',
        'sender',
        'text',
        'created_at'
    ];
}
