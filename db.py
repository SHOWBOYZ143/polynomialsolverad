import os
import sqlite3
from pathlib import Path
import psycopg2
import re

DB_PATH = Path(os.environ.get("POLY_DB_PATH", "polynomialsolver.db"))
DB_URL = os.environ.get("POLY_DATABASE_URL") or os.environ.get("DATABASE_URL")
USE_POSTGRES = bool(DB_URL)


DB_URL = os.environ.get("POLY_DATABASE_URL") or os.environ.get("DATABASE_URL")
USE_POSTGRES = bool(DB_URL)
PRAGMA_TABLE_INFO_RE = re.compile(r"PRAGMA\\s+table_info\\((?P<table>[^)]+)\\)", re.IGNORECASE)


def _convert_sql(sql: str) -> str:
    if USE_POSTGRES:
        pragma_match = PRAGMA_TABLE_INFO_RE.search(sql)
        if pragma_match:
            table = pragma_match.group("table").strip().strip('"').strip("'")
            return (
                "SELECT ordinal_position AS cid, column_name AS name "
                "FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name='{}' "
                "ORDER BY ordinal_position"
            ).format(table)
        return sql.replace("?", "%s")
    return sql


class CursorAdapter:
    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, sql, params=None):
        sql = _convert_sql(sql)
        if params is None:
            self._cursor.execute(sql)
        else:
            self._cursor.execute(sql, params)
        return self

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()


class ConnectionAdapter:
    def __init__(self, conn):
        self._conn = conn

    def cursor(self):
        return CursorAdapter(self._conn.cursor())

    def execute(self, sql, params=None):
        cur = self.cursor()
        cur.execute(sql, params)
        return cur

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

def get_connection():
    if USE_POSTGRES:
        return ConnectionAdapter(psycopg2.connect(DB_URL))
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return ConnectionAdapter(conn)


def _table_columns(cur, table):
    if USE_POSTGRES:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s
            """,
            (table,)
        )
        return {row[0] for row in cur.fetchall()}

    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _add_column(cur, table, column, definition):
    cols = _table_columns(cur, table)



  
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
