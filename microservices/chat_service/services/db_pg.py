import os
import psycopg
from psycopg.rows import dict_row


def get_conn():
    host = os.environ.get("DB_HOST", "host.docker.internal")
    port = int(os.environ.get("DB_PORT", "5432"))
    db = os.environ.get("DB_DATABASE", "aura_main")
    user = os.environ.get("DB_USERNAME", "aura_user")
    pwd = os.environ.get("DB_PASSWORD", "aura_pass")
    dsn = f"host={host} port={port} dbname={db} user={user} password={pwd}"
    return psycopg.connect(dsn, row_factory=dict_row)

