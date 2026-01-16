import base64
import hashlib
import os
from datetime import datetime

from db import init_db, get_connection

PBKDF2_ITERS = 200_000


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERS)
    return base64.b64encode(salt).decode() + ":" + base64.b64encode(dk).decode()

def seed_admin():
    init_db()
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM users WHERE username='ad'")
    exists = cur.fetchone() is not None

    if not exists:
        cur.execute(
            """
            INSERT INTO users (
                username,
                password,
                role,
                first_login,
                created_at,
                last_login,
                phone
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("ad", hash_password("ad"), "admin", 0, now_iso(), None, "0000000000")
        )

    conn.commit()
    conn.close()

if __name__ == "__main__":
    seed_admin()
