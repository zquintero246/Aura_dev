import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo import MongoClient
from bson import ObjectId


MONGO_URI = os.environ.get("MONGO_URI", "mongodb://aura_admin:aura_pass@host.docker.internal:27017")
MONGO_DB = os.environ.get("MONGO_DB", "aura_chat")

_client = MongoClient(MONGO_URI)
_db = _client[MONGO_DB]

conversations = _db["conversations"]
messages = _db["messages"]


def ensure_indexes() -> None:
    # Verify connection and log database
    try:
        _client.admin.command('ping')
        print(f"Connected to MongoDB database: {MONGO_DB}")
    except Exception as e:
        print("Mongo ping failed:", e)
        raise
    conversations.create_index([("user_id", 1), ("last_message_at", -1)])
    messages.create_index([("conversation_id", 1), ("created_at", 1)])
    print("Indexes ensured successfully")


def _oid(value: str) -> ObjectId:
    return ObjectId(value)


def create_conversation(user_id: str, title: Optional[str] = None) -> Dict[str, Any]:
    now = datetime.utcnow()
    doc = {
        "user_id": str(user_id),
        "title": title or "Nueva conversación",
        "created_at": now,
        "updated_at": now,
        "last_message_at": now,
    }
    res = conversations.insert_one(doc)
    return {"id": str(res.inserted_id), "title": doc["title"], "created_at": doc["created_at"].isoformat() + "Z"}


def list_conversations(user_id: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for c in conversations.find({"user_id": str(user_id)}).sort("last_message_at", -1):
        items.append({
            "id": str(c.get("_id")),
            "title": c.get("title"),
            "created_at": c.get("created_at").isoformat() + "Z" if c.get("created_at") else None,
            "updated_at": c.get("updated_at").isoformat() + "Z" if c.get("updated_at") else None,
            "last_message_at": c.get("last_message_at").isoformat() + "Z" if c.get("last_message_at") else None,
        })
    return items


def insert_message(conversation_id: str, user_id: str, role: str, content: str) -> Dict[str, Any]:
    cid = _oid(conversation_id)
    # Guard ownership
    conv = conversations.find_one({"_id": cid, "user_id": str(user_id)})
    if not conv:
        raise PermissionError("Conversation not found or not owned by user")

    now = datetime.utcnow()
    doc = {
        "conversation_id": cid,
        "user_id": str(user_id),
        "role": role,
        "content": content,
        "created_at": now,
    }
    res = messages.insert_one(doc)
    conversations.update_one({"_id": cid}, {"$set": {"updated_at": now, "last_message_at": now}})
    return {"id": str(res.inserted_id), "created_at": now.isoformat() + "Z"}


def list_messages(conversation_id: str, user_id: str) -> List[Dict[str, Any]]:
    cid = _oid(conversation_id)
    # Guard ownership
    conv = conversations.find_one({"_id": cid, "user_id": str(user_id)})
    if not conv:
        raise PermissionError("Conversation not found or not owned by user")

    items: List[Dict[str, Any]] = []
    for m in messages.find({"conversation_id": cid}).sort("created_at", 1):
        items.append({
            "id": str(m.get("_id")),
            "role": m.get("role"),
            "content": m.get("content"),
            "created_at": m.get("created_at").isoformat() + "Z" if m.get("created_at") else None,
        })
    return items
