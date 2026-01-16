import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.environ.get("POLY_DB_PATH", "polynomialsolver.db"))


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _add_column(cur, table, column, definition):
    cur.execute(f"PRAGMA table_info({table})")
    cols = {row[1] for row in cur.fetchall()}
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            first_login INTEGER DEFAULT 1,
            created_at TEXT,
            last_login TEXT
    )
    """)
    _add_column(cur, "users", "phone", "phone TEXT")
    _add_column(cur, "users", "email", "email TEXT")
    _add_column(cur, "users", "recovery_q1", "recovery_q1 TEXT")
    _add_column(cur, "users", "recovery_a1_hash", "recovery_a1_hash TEXT")
    _add_column(cur, "users", "recovery_q2", "recovery_q2 TEXT")
    _add_column(cur, "users", "recovery_a2_hash", "recovery_a2_hash TEXT")

    conn.commit()
    conn.close()
