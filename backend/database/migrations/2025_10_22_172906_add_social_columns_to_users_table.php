<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::table('users', function (Blueprint $table) {
            $table->string('google_id')->nullable()->unique()->after('password');
            $table->string('facebook_id')->nullable()->unique()->after('google_id');
            $table->string('github_id')->nullable()->unique()->after('facebook_id');
            $table->string('gitlab_id')->nullable()->unique()->after('github_id');
            $table->string('bitbucket_id')->nullable()->unique()->after('gitlab_id');
            $table->string('slack_id')->nullable()->unique()->after('bitbucket_id');
            $table->string('twitch_id')->nullable()->unique()->after('slack_id');
            $table->string('twitter_openid_id')->nullable()->unique()->after('twitch_id');
            $table->string('linkedin_openid_id')->nullable()->unique()->after('twitter_openid_id');
        });
    }

    public function down(): void
    {
        Schema::table('users', function (Blueprint $table) {
            $table->dropColumn([
                'google_id',
                'facebook_id',
                'github_id',
                'gitlab_id',
                'bitbucket_id',
                'slack_id',
                'twitch_id',
                'twitter_openid_id',
                'linkedin_openid_id'
            ]);
        });
    }
};