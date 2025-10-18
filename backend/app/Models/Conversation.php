<?php

namespace App\Models;

use Jenssegers\Mongodb\Eloquent\Model;

class Conversation extends Model
{
    protected $connection = 'mongodb';
    protected $collection = 'conversations';

    protected $fillable = [
        'user_id',
        'title',
        'messages',
        'created_at'
    ];
}
