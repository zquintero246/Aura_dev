import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from services.auth import resolve_user_from_bearer, AuthError
from services import mongo as mongosvc


app = Flask(__name__)

# CORS (dev)
origins = [o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
if origins:
    CORS(app, origins=origins, supports_credentials=False)


def _ensure_indexes_once():
    try:
        mongosvc.ensure_indexes()
    except Exception as e:
        print("Mongo init error:", e)


_ensure_indexes_once()


def _current_user():
    try:
        data = resolve_user_from_bearer(request.headers.get("Authorization"))
        return data, None
    except AuthError as e:
        return None, str(e)


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Chat API running",
        "endpoints": [
            "/chat/start",
            "/chat/message",
            "/chat/history",
            "/chat/conversations/<id>/messages",
        ],
    })


@app.route("/chat/start", methods=["POST"])
def start_chat():
    auth, err = _current_user()
    if not auth:
        return jsonify({"message": f"Unauthorized: {err}"}), 401
    payload = request.get_json(silent=True) or {}
    title = payload.get("title")
    try:
        conv = mongosvc.create_conversation(user_id=auth["user_id"], title=title)
        return jsonify(conv), 201
    except Exception as e:
        return jsonify({"message": f"Failed to create conversation: {e}"}), 500


@app.route("/chat/message", methods=["POST"])
def post_message():
    auth, err = _current_user()
    if not auth:
        return jsonify({"message": f"Unauthorized: {err}"}), 401
    payload = request.get_json(silent=True) or {}
    conversation_id = payload.get("conversation_id")
    content = (payload.get("content") or "").strip()
    role = payload.get("role") or "user"
    if not conversation_id or not content:
        return jsonify({"message": "conversation_id y content son requeridos"}), 400
    if role not in ("user", "assistant", "system"):
        return jsonify({"message": "role inválido"}), 400
    try:
        msg = mongosvc.insert_message(
            conversation_id=conversation_id,
            user_id=auth["user_id"],
            role=role,
            content=content,
        )
        return jsonify({"ok": True, "message": msg}), 201
    except PermissionError as e:
        return jsonify({"message": str(e)}), 403
    except Exception as e:
        return jsonify({"message": f"Failed to save message: {e}"}), 500


@app.route("/chat/history", methods=["GET"])
def history():
    auth, err = _current_user()
    if not auth:
        return jsonify({"message": f"Unauthorized: {err}"}), 401
    try:
        print(f"📡 /chat/history → user_id: {auth['user_id']}")
        items = mongosvc.list_conversations(user_id=auth["user_id"])
        print(f"✅ conversaciones recuperadas: {len(items)}")
        return jsonify({"conversations": items}), 200
    except Exception as e:
        return jsonify({"message": f"Failed to load history: {e}"}), 500


@app.route("/chat/conversations/<conv_id>/messages", methods=["GET"])
def conversation_messages(conv_id: str):
    auth, err = _current_user()
    if not auth:
        return jsonify({"message": f"Unauthorized: {err}"}), 401
    try:
        items = mongosvc.list_messages(conversation_id=conv_id, user_id=auth["user_id"])
        return jsonify({"messages": items}), 200
    except PermissionError as e:
        return jsonify({"message": str(e)}), 403
    except Exception as e:
        return jsonify({"message": f"Failed to load messages: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5060"))
    app.run(host="0.0.0.0", port=port)

