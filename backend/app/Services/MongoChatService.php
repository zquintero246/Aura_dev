<?php

namespace App\Services;

use Illuminate\Support\Facades\Log;

class MongoChatService
{
    private ?\MongoDB\Driver\Manager $manager = null;
    private string $db;

    public function __construct()
    {
        $this->db = env('MONGO_DB_DATABASE', 'aura_chat');

        try {
            if (!class_exists('MongoDB\\Driver\\Manager')) {
                Log::warning('MongoChatService: MongoDB extension not available');
                return; // manager stays null
            }

            $host = env('MONGO_DB_HOST', '127.0.0.1');
            $port = env('MONGO_DB_PORT', '27017');
            $user = env('MONGO_DB_USERNAME');
            $pass = env('MONGO_DB_PASSWORD');

            $auth = '';
            if ($user !== null && $user !== '') {
                $u = urlencode($user);
                $p = urlencode((string) $pass);
                $auth = "$u:$p@";
            }

            $uri = "mongodb://{$auth}{$host}:{$port}/{$this->db}";
            $this->manager = new \MongoDB\Driver\Manager($uri);
        } catch (\Throwable $e) {
            Log::error('MongoChatService: failed to create manager', ['error' => $e->getMessage()]);
            $this->manager = null;
        }
    }

    public function available(): bool
    {
        return $this->manager !== null;
    }

    /**
     * Create a conversation for a user.
     * @return string|null Inserted _id as hex string or null on failure.
     */
    public function createConversation(int|string $userId, ?string $title = null): ?string
    {
        if (!$this->available()) return null;

        try {
            $bulk = new \MongoDB\Driver\BulkWrite();
            $now = new \MongoDB\BSON\UTCDateTime((int) (microtime(true) * 1000));
            $doc = [
                'user_id' => (string) $userId,
                'title' => $title ?: 'Nueva conversación',
                'created_at' => $now,
            ];
            $id = $bulk->insert($doc);
            $ns = $this->db . '.conversations';
            $this->manager->executeBulkWrite($ns, $bulk);

            if ($id instanceof \MongoDB\BSON\ObjectId) {
                return (string) $id;
            }
            return is_string($id) ? $id : null;
        } catch (\Throwable $e) {
            Log::error('MongoChatService:createConversation failed', ['error' => $e->getMessage()]);
            return null;
        }
    }

    /**
     * List conversations for a user.
     * @return array<int,array{id:string,title:string,created_at:string}>
     */
    public function listConversations(int|string $userId): array
    {
        if (!$this->available()) return [];
        try {
            $filter = ['user_id' => (string) $userId];
            $options = [
                'sort' => ['created_at' => -1],
                'projection' => ['title' => 1, 'created_at' => 1],
            ];
            $query = new \MongoDB\Driver\Query($filter, $options);
            $cursor = $this->manager->executeQuery($this->db . '.conversations', $query);
            $res = [];
            foreach ($cursor as $doc) {
                $id = isset($doc->_id) ? (string) $doc->_id : '';
                $title = (string) ($doc->title ?? '');
                $createdAt = '';
                if (isset($doc->created_at) && $doc->created_at instanceof \MongoDB\BSON\UTCDateTime) {
                    $createdAt = $doc->created_at->toDateTime()->format(DATE_ATOM);
                }
                $res[] = ['id' => $id, 'title' => $title, 'created_at' => $createdAt];
            }
            return $res;
        } catch (\Throwable $e) {
            Log::error('MongoChatService:listConversations failed', ['error' => $e->getMessage()]);
            return [];
        }
    }

    /**
     * Append a pair of messages (user + assistant) to a messages collection.
     * If conversationId is string, will be converted to ObjectId when possible.
     */
    public function appendExchange(string $conversationId, string $userText, string $assistantText, int|string $userId): bool
    {
        if (!$this->available()) return false;
        try {
            $convId = $this->toObjectId($conversationId);
            $bulk = new \MongoDB\Driver\BulkWrite();
            $now = new \MongoDB\BSON\UTCDateTime((int) (microtime(true) * 1000));
            $base = [
                'conversation_id' => $convId ?? $conversationId,
                'user_id' => (string) $userId,
                'created_at' => $now,
            ];
            $bulk->insert($base + ['role' => 'user', 'content' => $userText]);
            $bulk->insert($base + ['role' => 'assistant', 'content' => $assistantText]);
            $this->manager->executeBulkWrite($this->db . '.messages', $bulk);
            return true;
        } catch (\Throwable $e) {
            Log::error('MongoChatService:appendExchange failed', ['error' => $e->getMessage()]);
            return false;
        }
    }

    /**
     * Update conversation title only if it still has a default placeholder title.
     */
    public function updateTitleIfDefault(string $conversationId, int|string $userId, string $newTitle): bool
    {
        if (!$this->available()) return false;
        $newTitle = trim($newTitle);
        if ($newTitle === '') return false;
        try {
            $convId = $this->toObjectId($conversationId);
            // Common placeholder variants (encoding-safe)
            $placeholders = [
                'Nueva conversación',
                'Nueva conversacion',
                'Nueva conversaci�n',
                'Nueva conversaci��n',
            ];
            $filter = [
                '_id' => $convId ?? $conversationId,
                'user_id' => (string) $userId,
                'title' => ['$in' => $placeholders],
            ];
            $update = ['$set' => ['title' => $newTitle]];
            $bulk = new \MongoDB\Driver\BulkWrite();
            $bulk->update($filter, $update, ['multi' => false, 'upsert' => false]);
            $result = $this->manager->executeBulkWrite($this->db . '.conversations', $bulk);
            // If any doc was modified or matched, we consider it success
            return ($result->getModifiedCount() + $result->getUpsertedCount()) > 0;
        } catch (\Throwable $e) {
            Log::error('MongoChatService:updateTitleIfDefault failed', ['error' => $e->getMessage()]);
            return false;
        }
    }

    private function toObjectId(string $id): ?\MongoDB\BSON\ObjectId
    {
        try {
            return new \MongoDB\BSON\ObjectId($id);
        } catch (\Throwable $e) {
            return null;
        }
    }
}
