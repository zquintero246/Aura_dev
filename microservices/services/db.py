import os
from typing import Optional, Iterable, Tuple, Any

# Prefer PostgreSQL; fallback to SQLite for local/dev
USE_POSTGRES = (os.environ.get("APP_DB_DRIVER") or "postgres").lower().startswith("postg")

if USE_POSTGRES:
    import psycopg
    from psycopg.rows import dict_row
else:
    import sqlite3


DB_PATH = os.environ.get(
    "APP_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "app.db"),
)


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def get_conn():
    if USE_POSTGRES:
        host = os.environ.get("DB_HOST", "host.docker.internal")
        port = int(os.environ.get("DB_PORT", "5432"))
        db = os.environ.get("DB_DATABASE", "aura_main")
        user = os.environ.get("DB_USERNAME", "aura_user")
        pwd = os.environ.get("DB_PASSWORD", "aura_pass")
        dsn = f"host={host} port={port} dbname={db} user={user} password={pwd}"
        return psycopg.connect(dsn, row_factory=dict_row)
    # SQLite fallback
    _ensure_dir(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
    except Exception:
        pass
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    if USE_POSTGRES:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS homes (
              user_id TEXT PRIMARY KEY,
              city TEXT NOT NULL,
              country TEXT NOT NULL,
              lat DOUBLE PRECISION NOT NULL,
              lon DOUBLE PRECISION NOT NULL,
              created_at TIMESTAMPTZ DEFAULT NOW(),
              updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        conn.commit()
    else:
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS homes (
                user_id TEXT PRIMARY KEY,
                city TEXT NOT NULL,
                country TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS countries (
                code TEXT PRIMARY KEY,
                name TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                country_code TEXT NOT NULL,
                FOREIGN KEY(country_code) REFERENCES countries(code)
            );
            """
        )
        conn.commit()
        # Optional seed for dev
        try:
            cur.execute("SELECT COUNT(*) AS c FROM countries")
            row = cur.fetchone()
            count = row[0] if row is not None else 0
        except Exception:
            count = 0
        if (count or 0) == 0:
            countries = [
                ("CO", "Colombia"),
                ("MX", "México"),
                ("US", "Estados Unidos"),
                ("ES", "España"),
            ]
            cur.executemany("INSERT INTO countries(code, name) VALUES(?, ?)", countries)
            cities = [
                ("Bucaramanga", "CO"),
                ("Bogotá", "CO"),
                ("Medellín", "CO"),
                ("Cali", "CO"),
                ("Ciudad de México", "MX"),
                ("Guadalajara", "MX"),
                ("Monterrey", "MX"),
                ("New York", "US"),
                ("San Francisco", "US"),
                ("Miami", "US"),
                ("Madrid", "ES"),
                ("Barcelona", "ES"),
                ("Valencia", "ES"),
            ]
            cur.executemany("INSERT INTO cities(name, country_code) VALUES(?, ?)", cities)
            conn.commit()
    cur.close()
    conn.close()


def save_user(user_id: str, city: str, country: str, lat: float, lon: float) -> None:
    """Store or update user's home. (Legacy function name kept)"""
    conn = get_conn()
    cur = conn.cursor()
    if USE_POSTGRES:
        cur.execute(
            """
            INSERT INTO homes(user_id, city, country, lat, lon, updated_at)
            VALUES(%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
              city=EXCLUDED.city,
              country=EXCLUDED.country,
              lat=EXCLUDED.lat,
              lon=EXCLUDED.lon,
              updated_at=NOW()
            """,
            (str(user_id), city, country, float(lat), float(lon)),
        )
    else:
        cur.execute(
            """
            INSERT INTO homes(user_id, city, country, lat, lon)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              city=excluded.city, country=excluded.country, lat=excluded.lat, lon=excluded.lon
            """,
            (str(user_id), city, country, float(lat), float(lon)),
        )
    conn.commit()
    cur.close()
    conn.close()


def get_user(user_id: str) -> Optional[Any]:
    """Return user's home row or None. (Legacy function name kept)"""
    conn = get_conn()
    cur = conn.cursor()
    if USE_POSTGRES:
        cur.execute(
            "SELECT user_id, city, country, lat, lon FROM homes WHERE user_id = %s",
            (str(user_id),),
        )
        row = cur.fetchone()
    else:
        cur.execute(
            "SELECT user_id, city, country, lat, lon FROM homes WHERE user_id = ?",
            (str(user_id),),
        )
        row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def list_countries() -> Iterable[Tuple[str, str]]:
    # Compatible with the legacy endpoint. For PG we reply with a minimal set.
    if USE_POSTGRES:
        return [("CO", "Colombia"), ("MX", "México"), ("US", "Estados Unidos"), ("ES", "España")]
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT code, name FROM countries ORDER BY name")
    rows = cur.fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


def list_cities(country_code: str) -> Iterable[str]:
    if USE_POSTGRES:
        return []
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT name FROM cities WHERE country_code = ? ORDER BY name", (country_code,))
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]

