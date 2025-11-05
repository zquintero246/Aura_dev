import hashlib
import hmac
from typing import Optional, Dict, Any

from .db_pg import get_conn


class AuthError(Exception):
    pass


def _hash_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_user_from_bearer(authorization_header: Optional[str]) -> Dict[str, Any]:
    """
    Valida un token Laravel Sanctum (plain text `id|secret`).
    Compara SHA256(secret) con la columna `token` en `personal_access_tokens`.
    Retorna {'user_id': str, 'token_id': int} si es válido, o lanza AuthError.
    """
    if not authorization_header:
        raise AuthError("Missing Authorization header")

    parts = authorization_header.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthError("Invalid Authorization header format")

    raw = parts[1].strip()
    if "|" not in raw:
        raise AuthError("Malformed token")

    token_id_str, secret = raw.split("|", 1)
    try:
        token_id = int(token_id_str)
    except Exception:
        raise AuthError("Invalid token id")

    token_hash = _hash_sha256(secret)
    print(f"🔑 Validando token id={token_id}")
    print(f"🧩 SHA256 calculado: {token_hash}")

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, tokenable_id, token, expires_at
            FROM personal_access_tokens
            WHERE id = %s
            """,
            (token_id,),
        )
        row = cur.fetchone()

    if not row:
        raise AuthError("Token not found")

    # row_factory=dict_row → `row` es dict
    stored_hash = (row.get("token") or "").strip()
    expected_user = row.get("tokenable_id")
    expires_at = row.get("expires_at")

    print(f"🧩 Hash esperado (Postgres): {stored_hash}")

    if not stored_hash:
        raise AuthError("Empty stored token hash")

    # Comparación constante
    if not hmac.compare_digest(stored_hash.lower(), token_hash.lower()):
        print("❌ Token mismatch")
        raise AuthError("Token mismatch")

    # Expiración opcional
    if expires_at is not None:
        import datetime as _dt
        now = _dt.datetime.now(tz=expires_at.tzinfo) if getattr(expires_at, "tzinfo", None) else _dt.datetime.now()
        if expires_at < now:
            raise AuthError("Token expired")

    user_id = str(expected_user)
    print(f"👤 Usuario autenticado: {user_id}")
    print("✅ Token validado correctamente")
    return {"user_id": user_id, "token_id": token_id}

