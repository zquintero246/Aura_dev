import os
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

from services.auth import resolve_user_from_bearer, AuthError
from services import mongo as mongosvc


app = Flask(__name__)

# CORS (dev)
# Always enable CORS in dev; allow env override or default to local FE origins.
_env_origins = (os.environ.get("CORS_ALLOW_ORIGINS", "") or "").strip()
if _env_origins:
    _origins = [o.strip() for o in _env_origins.split(",") if o.strip()]
else:
    _origins = [
        "http://127.0.0.1:4028",
        "http://localhost:4028",
    ]
# Ensure DELETE and Authorization header are permitted for preflight
CORS(
    app,
    resources={r"/*": {"origins": _origins}},
    supports_credentials=True,
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "authorization"],
)


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
            "/chat/conversations/<id>",
            "/chat/conversations/<id> (PUT)",
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
        return jsonify({"message": "role invalido"}), 400
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
        # ascii-safe debug logs
        print(f"/chat/history -> user_id: {auth['user_id']}")
        items = mongosvc.list_conversations(user_id=auth["user_id"])
        print(f"conversations loaded: {len(items)}")
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


@app.route("/chat/conversations/<id>", methods=["DELETE", "OPTIONS"])
@cross_origin(
    origins=["http://127.0.0.1:4028", "http://localhost:4028"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "authorization"],
)
def delete_conversation(id: str):
    # --- Preflight ---
    if request.method == "OPTIONS":
        return "", 200

    # --- DELETE logic ---
    auth, err = _current_user()
    if not auth:
        return jsonify({"message": f"Unauthorized: {err}"}), 401
    try:
        ok = mongosvc.delete_conversation(conversation_id=id, user_id=auth["user_id"])
        if not ok:
            return jsonify({"message": "Conversation not found"}), 404
        return jsonify({"ok": True}), 200
    except PermissionError as e:
        return jsonify({"message": str(e)}), 403
    except Exception as e:
        return jsonify({"message": f"Failed to delete conversation: {e}"}), 500


@app.route("/chat/conversations/<conv_id>", methods=["PUT"])
def update_conversation(conv_id: str):
    auth, err = _current_user()
    if not auth:
        return jsonify({"message": f"Unauthorized: {err}"}), 401
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    if not title:
        return jsonify({"message": "title es requerido"}), 400
    try:
        conv = mongosvc.update_conversation_title(
            user_id=auth["user_id"],
            conversation_id=conv_id,
            title=title,
        )
        return jsonify({"conversation": conv}), 200
    except PermissionError as e:
        return jsonify({"message": str(e)}), 403
    except ValueError as e:
        return jsonify({"message": str(e)}), 400
    except Exception as e:
        return jsonify({"message": f"Failed to update conversation: {e}"}), 500


# --- catch-all opcional para /chat/* ---
@app.route("/chat/<path:subpath>", methods=["OPTIONS"])
@cross_origin(
    origins=["http://127.0.0.1:4028", "http://localhost:4028"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "authorization"],
)
def options_catch_all(subpath: str):
    return "", 200



# --- DEBUG: Mostrar rutas registradas ---
def _print_routes():
    print("\n=== URL MAP ===")
    for rule in app.url_map.iter_rules():
        print(f"{rule.methods} -> {rule.rule}")
    print("==============\n")

_print_routes()
# --- FIN DEBUG ---


# (Flask-CORS handles headers; no manual after_request needed)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5080"))
    app.run(host="0.0.0.0", port=port)
