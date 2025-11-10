import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo import MongoClient, ReturnDocument
from bson import ObjectId
import re
import unicodedata


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


def update_conversation_title(user_id: str, conversation_id: str, title: str) -> Dict[str, Any]:
    cleaned = (title or "").strip()
    if not cleaned:
        raise ValueError("Title cannot be empty")
    cid = _oid(conversation_id)
    now = datetime.utcnow()
    updated = conversations.find_one_and_update(
        {"_id": cid, "user_id": str(user_id)},
        {"$set": {"title": cleaned, "updated_at": now}},
        return_document=ReturnDocument.AFTER,
    )
    if not updated:
        raise PermissionError("Conversation not found or not owned by user")
    return {
        "id": str(updated.get("_id")),
        "title": updated.get("title"),
        "created_at": updated.get("created_at").isoformat() + "Z" if updated.get("created_at") else None,
        "updated_at": updated.get("updated_at").isoformat() + "Z" if updated.get("updated_at") else None,
        "last_message_at": updated.get("last_message_at").isoformat() + "Z" if updated.get("last_message_at") else None,
    }


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
    # Auto-title: if this is the first user message and the conversation still has a placeholder title,
    # generate a concise title (<= 6 words) and persist it. Otherwise just update timestamps.
    try:
        if role == "user":
            def _simplify(s: str) -> str:
                if not s:
                    return ""
                try:
                    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
                except Exception:
                    pass
                s = re.sub(r"[^A-Za-z0-9\s]", "", s)
                return re.sub(r"\s+", " ", s).strip().lower()

            def _auto_title(src: str) -> str:
                t = re.sub(r"\s+", " ", (src or "").strip())
                t = re.sub(r"^[^A-Za-z0-9]+", "", t)
                t = re.sub(r"[\s\.\?\!\,;:]+$", "", t)
                rules = [
                    (r"^dame\s+ideas\s+para\s+", "Ideas para "),
                    (r"^ideas\s+para\s+", "Ideas para "),
                    (r"^como\s+hacer\s+", "Hacer "),
                    (r"^como\s+crear\s+", "Crear "),
                    (r"^como\s+hago\s+", "Crear "),
                    (r"^como\s+puedo\s+", ""),
                    (r"^quiero\s+", ""),
                    (r"^quisiera\s+", ""),
                    (r"^necesito\s+", ""),
                    (r"^por\s+favor\s+", ""),
                ]
                for pat, rep in rules:
                    if re.search(pat, t, flags=re.I):
                        t = re.sub(pat, rep, t, flags=re.I)
                        break
                first = re.split(r"(?<=[\.!?])\s+", t)[0] if t else t
                first = re.sub(r"[\s\.\?\!\,;:]+$", "", first)
                words = re.split(r"\s+", first) if first else []
                if len(words) > 6:
                    first = " ".join(words[:6])
                if first:
                    first = first[0:1].upper() + first[1:]
                if first.lower() in {"nueva conversacion", "nuevo chat", "conversacion con ia", "charla general", "chat general"}:
                    return ""
                return first.strip()

            current_title = (conv.get("title") or "")
            simp = _simplify(current_title)
            if simp.startswith("nueva convers"):
                auto = _auto_title(content or "")
                if auto:
                    conversations.update_one({"_id": cid}, {"$set": {"title": auto, "updated_at": now, "last_message_at": now}})
                else:
                    conversations.update_one({"_id": cid}, {"$set": {"updated_at": now, "last_message_at": now}})
            else:
                conversations.update_one({"_id": cid}, {"$set": {"updated_at": now, "last_message_at": now}})
        else:
            conversations.update_one({"_id": cid}, {"$set": {"updated_at": now, "last_message_at": now}})
    except Exception:
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


def delete_conversation(conversation_id: str, user_id: str) -> bool:
    """Delete a conversation and its messages if owned by user."""
    cid = _oid(conversation_id)
    conv = conversations.find_one({"_id": cid, "user_id": str(user_id)})
    if not conv:
        raise PermissionError("Conversation not found or not owned by user")
    messages.delete_many({"conversation_id": cid})
    res = conversations.delete_one({"_id": cid})
    return res.deleted_count > 0
