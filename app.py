import streamlit as st
import sqlite3
import os
import pandas as pd
import hashlib
import hmac
import base64
import numpy as np
import matplotlib.pyplot as plt
import io
import psycopg2
import re

import csv
from datetime import datetime
from datetime import timedelta

PBKDF2_ITERS = 200_000

st.session_state.setdefault("recovery_verified", False)
st.session_state.setdefault("reset_user", None)


if "show_menu" not in st.session_state:
    st.session_state.show_menu = False
 
if "page" not in st.session_state:
    st.session_state.page = "solver"
st.markdown("""

<style>
:root {
    --primary-color: #2563eb;
    --primary-color-dark: #1d4ed8;
    --primary-soft: #dbeafe;
    --card-border: #93c5fd;
    --card-bg: #e0f2fe;
    --page-bg: linear-gradient(180deg, #eff6ff 0%, #dbeafe 100%);
    --text-main: #111827;
    --text-muted: #4b5563;
    --btn-primary: #2563eb;
    --btn-primary-hover: #1d4ed8;
    --btn-secondary: #0ea5e9;
    --btn-secondary-hover: #0284c7;
    --btn-danger: #dc2626;
    --btn-danger-hover: #b91c1c;
}

div[data-testid="stAppViewContainer"] {
    background-color: var(--page-bg);
}

div[data-testid="stAppViewContainer"] > .main {
    background: var(--page-bg);
}

div[data-testid="metric-container"] {
    background-color: #0f1117;
    border: 1px solid #2a2d3a;
    padding: 16px;
    border-radius: 10px;
}

.auth-scope h1,
.auth-scope h2,
.auth-scope h3 {
    color: var(--text-main);
}

.auth-scope p,
.auth-scope label {
    color: var(--text-muted);
}

.auth-scope div[data-testid="stContainer"] {
    background-color: var(--card-bg);
    border: 1px solid var(--card-border);
    box-shadow: 0 12px 24px rgba(15, 23, 42, 0.06);
}

.auth-scope .stButton > button {
    background-color: var(--primary-color);
    color: #ffffff;
    border: 1px solid var(--primary-color);
}

.auth-scope .stButton > button:hover {
    background-color: var(--primary-color-dark);
    border-color: var(--primary-color-dark);
    color: #ffffff;
}

.auth-scope .stButton > button:focus {
    outline: 3px solid rgba(37, 99, 235, 0.25);
    outline-offset: 1px;
}

.stButton > button[aria-label="Login"],
.stButton > button[aria-label="Create account"],
.stButton > button[aria-label="Create user"],
.stButton > button[aria-label="Save as new version"],
.stButton > button[aria-label="Save recovery details"],
.stButton > button[aria-label="Reset password"],
.stButton > button[aria-label="Solve and Plot"] {
    background-color: var(--btn-primary);
    border-color: var(--btn-primary);
    color: #ffffff;
}

.stButton > button[aria-label="Login"]:hover,
.stButton > button[aria-label="Create account"]:hover,
.stButton > button[aria-label="Create user"]:hover,
.stButton > button[aria-label="Save as new version"]:hover,
.stButton > button[aria-label="Save recovery details"]:hover,
.stButton > button[aria-label="Reset password"]:hover,
.stButton > button[aria-label="Solve and Plot"]:hover {
    background-color: var(--btn-primary-hover);
    border-color: var(--btn-primary-hover);
}

.stButton > button[aria-label="Forgot password?"],
.stButton > button[aria-label="Back to login"],
.stButton > button[aria-label="⬅ Back to Solver"],
.stButton > button[aria-label="Reuse"],
.stButton > button[aria-label="Rename"],
.stButton > button[aria-label="Export selected run as CSV"],
.stButton > button[aria-label="Zoom in"],
.stButton > button[aria-label="Zoom out"],
.stButton > button[aria-label="Reset view"],
.stButton > button[aria-label="History"] {
    background-color: var(--btn-secondary);
    border-color: var(--btn-secondary);
    color: #ffffff;
}

.stButton > button[aria-label="Forgot password?"]:hover,
.stButton > button[aria-label="Back to login"]:hover,
.stButton > button[aria-label="⬅ Back to Solver"]:hover,
.stButton > button[aria-label="Reuse"]:hover,
.stButton > button[aria-label="Rename"]:hover,
.stButton > button[aria-label="Export selected run as CSV"]:hover,
.stButton > button[aria-label="Zoom in"]:hover,
.stButton > button[aria-label="Zoom out"]:hover,
.stButton > button[aria-label="Reset view"]:hover,
.stButton > button[aria-label="History"]:hover {
    background-color: var(--btn-secondary-hover);
    border-color: var(--btn-secondary-hover);
}

.stButton > button[aria-label="Delete"],
.stButton > button[aria-label="Logout"] {
    background-color: var(--btn-danger);
    border-color: var(--btn-danger);
    color: #ffffff;
}

.stButton > button[aria-label="Delete"]:hover,
.stButton > button[aria-label="Logout"]:hover {
    background-color: var(--btn-danger-hover);
    border-color: var(--btn-danger-hover);
}

.auth-scope div[data-testid="stTextInput"] input {
    background-color: var(--primary-soft);
    border: 1px solid var(--card-border);
}

div[data-testid="stSelectbox"],
div[data-testid="stTextInput"] {
    max-width: 480px;
}

div[data-testid="stSelectbox"] > div,
div[data-testid="stTextInput"] input {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)
    

# ======================================================
# App config
# ======================================================
st.set_page_config(page_title="Polynomial Solver Portal", layout="wide")


def _pick_db_dir() -> str:
    preferred = "/mount/data"
    if os.path.isdir(preferred) and os.access(preferred, os.W_OK):
        return preferred

    cwd = os.getcwd()
    if os.access(cwd, os.W_OK):
        return cwd

    fallback = "/tmp/polynomialsolverad"
    os.makedirs(fallback, exist_ok=True)
    return fallback


DEFAULT_DB_DIR = _pick_db_dir()
DB_URL = os.environ.get("POLY_DATABASE_URL") or os.environ.get("DATABASE_URL")
USE_POSTGRES = bool(DB_URL)

DB_PATH = os.environ.get(
    "POLY_DB_PATH",
    os.path.join(DEFAULT_DB_DIR, "polynomialsolver.db")
)
MIRROR_DB_PATH = os.environ.get(
    "POLY_MIRROR_DB_PATH",
    os.path.join(DEFAULT_DB_DIR, "dbforsql.db")
)
for _path in (DB_PATH, MIRROR_DB_PATH):
    _dir = os.path.dirname(_path)
    if _dir:
        os.makedirs(_dir, exist_ok=True)

DB_ERRORS = (sqlite3.Error, psycopg2.Error)


def _convert_sql(sql: str) -> str:
    if USE_POSTGRES:
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

    def executemany(self, sql, seq):
        sql = _convert_sql(sql)
        self._cursor.executemany(sql, seq)
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

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()
        
# ======================================================
# Time helpers
# ======================================================
def now_iso():
    return datetime.now().isoformat(timespec="seconds")

# ======================================================
# Database helpers
# ======================================================


RECOVERY_QUESTIONS = [
    "What is the name of your first school?",
    "What city were you born in?",
    "What is your favourite colour?",
    "What is your mother’s first name?",
    "What was the name of your first pet?"
]




def get_user_stats(username):
    con = get_db()
    cur = con.cursor()

    total_solves = cur.execute(
        "SELECT COUNT(*) FROM history WHERE username=?",
        (username,)
    ).fetchone()[0]

    last_solve = cur.execute(
        "SELECT MAX(created_at) FROM history WHERE username=?",
        (username,)
    ).fetchone()[0]

    user_row = cur.execute(
        "SELECT role, first_login, last_login FROM users WHERE username=?",
        (username,)
    ).fetchone()

    con.close()

    role, first_login, last_login = user_row

    inactive = False
    if last_login and last_solve:
        inactive = last_solve < last_login

    return {
        "username": username,
        "role": role,
        "total_solves": total_solves,
        "last_solve": last_solve or "—",
        "last_login": last_login or "—",
        "first_login": "Yes" if first_login else "No",
        "inactive": inactive
    }



def user_stats_view():
    st.subheader("My Statistics")

    s = get_user_stats(st.session_state.username)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total solves", s["total_solves"])
    c2.metric("Last solve time", s["last_solve"] or "None")
    c3.metric("Last login", s["last_login"] or "None")


 
def get_db():
    if USE_POSTGRES:
        return ConnectionAdapter(psycopg2.connect(DB_URL))
        
    con = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    return ConnectionAdapter(con)

def get_mirror_db():
    if USE_POSTGRES:
        return None
    con = sqlite3.connect(MIRROR_DB_PATH, timeout=30, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    return ConnectionAdapter(con)


def _table_exists(cur, table_name):
    if USE_POSTGRES:
        cur.execute("SELECT to_regclass(%s)", (table_name,))
        return cur.fetchone()[0] is not None

    cur.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name=?
    """, (table_name,))
    return cur.fetchone() is not None


def _table_columns(cur, table_name):
    if USE_POSTGRES:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s
        """, (table_name,))
        return [row[0] for row in cur.fetchall()]

    cur.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cur.fetchall()]


def _history_id_definition():
    if USE_POSTGRES:
        return "SERIAL PRIMARY KEY"
    return "INTEGER PRIMARY KEY AUTOINCREMENT"
    

def ensure_mirror_schema():
    if USE_POSTGRES:
        return
    con = get_mirror_db()
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users_mirror (
            username TEXT PRIMARY KEY,
            role TEXT DEFAULT 'user',
            phone TEXT,
            email TEXT,
            first_login INTEGER DEFAULT 1,
            created_at TEXT,
            last_login TEXT
        )
    """)
    con.commit()
    con.close()


def sync_users_to_mirror():
    if USE_POSTGRES:
        return
    try:
        con = get_db()
        cur = con.cursor()
        rows = cur.execute(
            """
            SELECT username, role, phone, email, first_login, created_at, last_login
            FROM users
            """
        ).fetchall()
        con.close()
    except DB_ERRORS:
        return

    try:
        for row in rows:
            upsert_mirror_user(*row)
    except DB_ERRORS:
        return


def ensure_history_schema_v2():
    """
    Ensures history table has the columns we need for:
    - per-user history
    - naming calculations
    - keeping versions
    - comparison and export metadata
    Works even if you already have an older history table.
    """
    con = get_db()
    cur = con.cursor()

    exists = _table_exists(cur, "history")

    if not exists:
        # Fresh create (correct structure)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id {id_def},
                username TEXT NOT NULL,
                coeffs TEXT NOT NULL,
                roots TEXT NOT NULL,
                created_at TEXT NOT NULL,
                calc_name TEXT,
                version INTEGER DEFAULT 1
            )
        """.format(id_def=_history_id_definition()))
        con.commit()
        con.close()
        return

    # Inspect existing columns
    
    cols = _table_columns(cur, "history")

    # If username column is missing, we rebuild table safely
    if "username" not in cols: 
        cur.execute("""
            ALTER TABLE history RENAME TO history_old
        """)

        cur.execute("""
            CREATE TABLE history (
                id {id_def},
                username TEXT NOT NULL,
                coeffs TEXT NOT NULL,
                roots TEXT NOT NULL,
                created_at TEXT NOT NULL,
                calc_name TEXT,
                version INTEGER DEFAULT 1
            )
         """.format(id_def=_history_id_definition()))

        # Migrate old rows. Old table had no username, we tag them as "unknown".
        cur.execute("""
            INSERT INTO history (username, coeffs, roots, created_at, calc_name, version)
            SELECT 'unknown', coeffs, roots, created_at, NULL, 1
            FROM history_old
        """)

        cur.execute("DROP TABLE history_old")

    else:
        # Add calc_name if missing
        if "calc_name" not in cols:
            cur.execute("ALTER TABLE history ADD COLUMN calc_name TEXT")

        # Add version if missing
        if "version" not in cols:
            cur.execute("ALTER TABLE history ADD COLUMN version INTEGER DEFAULT 1")

    con.commit()
    con.close()

def ensure_users_schema_v2():
    con = get_db()
    cur = con.cursor()

    cols = _table_columns(cur, "users")
    if "phone" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN phone TEXT")

    if "email" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN email TEXT")

    con.commit()
    con.close()

def ensure_users_schema_v3():
    con = get_db()
    cur = con.cursor()

    cols = _table_columns(cur, "users")

    if "recovery_q1" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN recovery_q1 TEXT")
        cur.execute("ALTER TABLE users ADD COLUMN recovery_a1_hash TEXT")
        cur.execute("ALTER TABLE users ADD COLUMN recovery_q2 TEXT")
        cur.execute("ALTER TABLE users ADD COLUMN recovery_a2_hash TEXT")

    con.commit()
    con.close()



def highlight_rows(row):
    if row["role"] == "admin":
        return ["background-color: #111827; color: #e5e7eb"] * len(row)

    if row["first_login_pending"] == "Yes":
        return ["background-color: #3b2f2f"] * len(row)
    if row["inactive"]:
        return ["color: #9ca3af"] * len(row)
    return [""] * len(row)


def ensure_schema():
    con = get_db()
    cur = con.cursor()

    # ------------------------
    # Base users table (minimal)
    # ------------------------
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

    con.commit()
    con.close()

    # ------------------------
    # Apply incremental migrations
    # ------------------------
    ensure_users_schema_v2()   # phone, email
    ensure_users_schema_v3()   # recovery questions
    ensure_history_schema_v2()
    if not USE_POSTGRES:
        ensure_mirror_schema()
    
    # ------------------------
    # Default admin user
    # ------------------------
    con = get_db()
    cur = con.cursor()

    cur.execute("SELECT 1 FROM users WHERE username='ad'")
    if not cur.fetchone():
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

    con.commit()
    con.close()

    migrate_plaintext_passwords()
    if not USE_POSTGRES:
        ensure_mirror_schema()

def ensure_history_schema():
    con = get_db()
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        phone TEXT NOT NULL,
        email TEXT,
        first_login INTEGER DEFAULT 1,
        created_at TEXT,
        last_login TEXT
    )
    """)

    con.commit()
    con.close()
    
def save_history(username, coeffs_text, roots_text, calc_name=None):
    con = get_db()
    cur = con.cursor()

    # Determine next version for this calculation
    cur.execute("""
        SELECT COALESCE(MAX(version), 0) + 1
        FROM history
        WHERE username = ? AND coeffs = ?
    """, (username, coeffs_text))

    next_version = cur.fetchone()[0]

    cur.execute("""
        INSERT INTO history (
            username,
            coeffs,
            roots,
            created_at,
            calc_name,
            version
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        username,
        coeffs_text,
        roots_text,
        now_iso(),
        calc_name,
        next_version
    ))

    con.commit()
    con.close()


def format_roots_for_storage(roots):
    if roots is None:
        return ""

    formatted = []
    for i, r in enumerate(roots, 1):
        if abs(r.imag) < 1e-6:
            formatted.append(f"x{i} = {r.real:.6f}")
        else:
            formatted.append(
                f"x{i} = {r.real:.6f} {'+' if r.imag >= 0 else '-'} {abs(r.imag):.6f}i"
            )
    return "\n".join(formatted)

    

# ======================================================
# Authentication
# ======================================================
def fetch_user(username):
    con = get_db()
    cur = con.cursor()
    cur.execute("""
        SELECT username, password, role, first_login, phone, email
        FROM users WHERE username=?
    """, (username,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return {
        "username": row[0],
        "password": row[1],
        "role": row[2],
        "first_login": row[3],
        "phone": row[4],
        "email": row[5],
    }

def authenticate(u, p):
    user = fetch_user(u)
    if not user:
        return None

    if not verify_password(p, user["password"]):
        return None

    con = get_db()
    con.execute("UPDATE users SET last_login=? WHERE username=?", (now_iso(), u))
    con.commit()
    con.close()

    return user


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERS)
    return base64.b64encode(salt).decode() + ":" + base64.b64encode(dk).decode()

def hash_answer(answer: str) -> str:
    normalised = answer.strip().lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()
    




def validate_password_strength(password: str) -> tuple[bool, str]:
    if len(password) < 5:
        return False, "Password must be at least 5 characters long."
    if not any(char.islower() for char in password):
        return False, "Password must include at least one lowercase letter."
    if not any(char.isupper() for char in password):
        return False, "Password must include at least one uppercase letter."
    if not any(not char.isalnum() for char in password):
        return False, "Password must include at least one special character."
    return True, ""



def verify_password(password: str, stored: str) -> bool:
    # stored format: "salt_b64:hash_b64"
    try:
        salt_b64, hash_b64 = stored.split(":", 1)
        salt = base64.b64decode(salt_b64.encode())
        expected = base64.b64decode(hash_b64.encode())
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERS)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False

def update_password(username, pw):
    ok, msg = validate_password_strength(pw)
    if not ok:
         return False, msg
    con = get_db()
    con.execute(
        "UPDATE users SET password=?, first_login=0 WHERE username=?",
        (hash_password(pw), username)
    )
    con.commit()
    con.close()
    return True, ""


def migrate_plaintext_passwords():
    con = get_db()
    cur = con.cursor()

    cur.execute("SELECT username, password FROM users")
    rows = cur.fetchall()

    for username, pw in rows:
        # If it already contains a colon, we assume it is hashed "salt:hash"
        if pw and ":" not in pw:
            cur.execute(
                "UPDATE users SET password=? WHERE username=?",
                (hash_password(pw), username)
            )

    con.commit()
    con.close()


def upsert_mirror_user(username, role, phone, email, first_login, created_at, last_login):
    if USE_POSTGRES:
        return
    con = get_mirror_db()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO users_mirror (
            username,
            role,
            phone,
            email,
            first_login,
            created_at,
            last_login
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            role=excluded.role,
            phone=excluded.phone,
            email=excluded.email,
            first_login=excluded.first_login,
            created_at=excluded.created_at,
            last_login=excluded.last_login
        """,
        (username, role, phone, email, first_login, created_at, last_login)
    )
    con.commit()
    con.close()


def delete_mirror_user(username):
    if USE_POSTGRES:
        return
    con = get_mirror_db()
    con.execute("DELETE FROM users_mirror WHERE username=?", (username,))
    con.commit()
    con.close()


def create_user(username, pw, role, phone, email=None):
    con = get_db()
    cur = con.cursor()

    cur.execute(
        "SELECT 1 FROM users WHERE username=?",
        (username,)
    )
    if cur.fetchone():
        con.close()
        return False, "User already exists"

    created_at = now_iso()

    cur.execute(
        """
        INSERT INTO users (
            username,
            password,
            role,
            phone,
            email,
            first_login,
            created_at,
            last_login
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            username,
            hash_password(pw),
            role,
            phone,
            email,
            1,
            created_at,
            None
        )
    )
    try:
        upsert_mirror_user(
            username=username,
            role=role,
            phone=phone,
            email=email,
            first_login=1,
            created_at=created_at,
            last_login=None
        )
    except DB_ERRORS as exc:
        con.rollback()
        con.close()
        return False, f"User not created (mirror DB error: {exc})."

    
    con.commit()
    con.close()
    return True, "User created successfully"





def human_time(ts):
    if not ts:
        return "—"
    return datetime.fromisoformat(ts).strftime("%d %b %Y, %H:%M")



def delete_user(username):
    if username == "ad":
        return False, "Default admin cannot be deleted"
    con = get_db()
    con.execute("DELETE FROM users WHERE username=?", (username,))
    con.commit()
    con.close()
    delete_mirror_user(username)
    return True, "User deleted"

def back_to_solver():
    if st.button("⬅ Back to Solver"):
        st.session_state.page = "solver"
        st.rerun()

# ======================================================
# Utilities
# ======================================================
def superscript(n):
    return str(n).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))

def format_root(r):
    if abs(r.imag) < 1e-6:
        return f"{r.real:.6f}"
    sign = "+" if r.imag >= 0 else "-"
    return f"{r.real:.6f} {sign} {abs(r.imag):.6f}i"

def parse_coeffs(text):
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(vals) < 2:
        raise ValueError
    return vals

def roots_csv(poly, roots):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Polynomial", "Root", "Value"])
    for i, r in enumerate(roots, 1):
        w.writerow([poly, f"x{superscript(i)}", format_root(r)])
    return buf.getvalue()


def fig_bytes(fig, fmt):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt.lower(), dpi=220, bbox_inches="tight")
    return buf.getvalue()

# ======================================================
# Session state
# ======================================================
def init_state():
    defaults = dict(
        logged_in=False,
        username=None,
        role=None,
        first_login=0,
        coeff_text="",
        coeffs=None,
        roots=None,
        xlim=(-10.0, 10.0),
        graph_fmt="PNG",
        login_fails=0,
        lock_until=None,
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("page", "solver")
    st.session_state.setdefault("show_menu", False)

def logout():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state["show_menu"] = False
    st.session_state["page"] = "solver"
    st.rerun()




# ======================================================
# Views
# ======================================================

def login_view():
    st.markdown("<h1 style=\"text-align: center;\">Polynomial Solver Portal</h1>", unsafe_allow_html=True)
    st.markdown("<div class=\"auth-scope\">", unsafe_allow_html=True)

    # ---------- Lockout handling ----------
    st.session_state.setdefault("login_fails", 0)
    st.session_state.setdefault("lock_until", None)

    if st.session_state.lock_until:
        if datetime.now() < st.session_state.lock_until:
            st.error("Too many attempts. Try again shortly.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        st.session_state.lock_until = None
        st.session_state.login_fails = 0

    # ---------- Login form (Enter submits) ----------
    left, center, right = st.columns([1, 2, 1])
    with center:
        with st.container(border=True):
            with st.form("login_form", clear_on_submit=False):
                u = st.text_input("Username", key="login_user").strip()
                p = st.text_input("Password", type="password", key="login_pass")
                login_clicked = st.form_submit_button("Login")

            forgot_col, signup_col = st.columns([1, 1])
            with forgot_col:
                if st.button("Forgot password?", key="forgot_btn"):
                    st.session_state.page = "recover"
                    st.rerun()

            with signup_col:
                if st.button("Create account", key="signup_btn"):
                    st.session_state.page = "signup"
                    st.rerun()

    # ---------- Handle login ----------
    if not login_clicked:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if not u or not p:
        st.error("Username and password are required.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    user = authenticate(u, p)
    if not user:
        st.session_state.login_fails += 1
        if st.session_state.login_fails >= 5:
            st.session_state.lock_until = datetime.now() + timedelta(seconds=30)
        st.error("Invalid credentials")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.session_state.login_fails = 0
    st.session_state.lock_until = None
    st.session_state.logged_in = True
    st.session_state.username = user["username"]
    st.session_state.role = user["role"]
    st.session_state.first_login = user["first_login"]

    # Decide next page
    st.session_state.page = "recovery_setup" if st.session_state.first_login == 1 else "solver"
    st.markdown("</div>", unsafe_allow_html=True)
    st.rerun()





def forced_password_change():
    st.warning("Change your password before proceeding")
    p1 = st.text_input("New password", type="password")
    p2 = st.text_input("Confirm password", type="password")
    if st.button("Save"):
        if p1 != p2:
            st.error("Passwords do not match.")
            return
        ok, msg = update_password(st.session_state.username, p1)
        if not ok:
            st.error(msg)
            return
        
        st.session_state.first_login = 0
        st.success("Password updated")
        st.rerun()

def solver_view():
    st.subheader("Polynomial Solver")

    key_ns = f"{st.session_state.username}_{st.session_state.get('page','solver')}"

    # -------------------------
    # Defaults
    # -------------------------
    st.session_state.setdefault("xlim", (-10.0, 10.0))
    st.session_state.setdefault("roots", None)
    st.session_state.setdefault("coeff_text", "")
    st.session_state.setdefault("coeffs", None)
    st.session_state.setdefault("graph_fmt", "PNG")
    st.session_state.setdefault("comparison_mode", False)

    # =========================
    # POLYNOMIAL INPUT
    # =========================
    if "coeff_text" not in st.session_state:
       st.session_state.coeff_text = ""

    if "reuse_coeffs" in st.session_state:
        st.session_state.coeff_text = st.session_state.reuse_coeffs
        del st.session_state.reuse_coeffs
       

    st.text_input(
        "Polynomial coefficients",
        key="coeff_text",
        placeholder="2, -3, 4"
    )

    try:
        st.session_state.coeffs = parse_coeffs(st.session_state.coeff_text)
    except:
        st.session_state.coeffs = None


    # =========================
    # SOLVE
    # =========================
    if st.button("Solve and Plot", key=f"solve_btn_{key_ns}"):
        if st.session_state.coeffs:
            with st.spinner("Solving and plotting..."):
                st.session_state.roots = np.roots(st.session_state.coeffs)
                roots_text = format_roots_for_storage(st.session_state.roots)

                save_history(
                    st.session_state.username,
                    st.session_state.coeff_text,
                    roots_text
                )
        else:
            st.session_state.roots = None
            st.error("Enter at least two coefficients, separated by commas.")

    # =========================
    # ROOTS (IMMEDIATELY BELOW SOLVE)
    # =========================
    if st.session_state.roots is not None:
        st.divider()

        st.markdown("### Solved Roots")

        roots_list = list(st.session_state.roots)
        cols = st.columns(len(roots_list))

        for col, (i, r) in zip(cols, enumerate(roots_list, 1)):
            col.markdown(f"**x{superscript(i)}**")
            col.markdown(format_root(r))

        if not any(abs(r.imag) < 1e-6 for r in roots_list):
            st.info("No real roots detected.")

    # =========================
    # SAVED RUNS
    # =========================
    st.markdown("### Saved Runs")

    # -----------------------------
    # State for rename panel
    # -----------------------------
    st.session_state.setdefault(f"show_rename_{key_ns}", False)

    con = get_db()
    rows = con.execute("""
        SELECT id, coeffs, roots, created_at, calc_name, version
        FROM history
        WHERE username = ?
        ORDER BY created_at DESC
    """, (st.session_state.username,)).fetchall()
    con.close()

    run_options = []
    id_map = {}

    for rid, coeffs, roots, ts, calc_name, version in rows:
        display_name = calc_name or f"Calculation v{version}"
        label = f"{ts} , {display_name} , {coeffs}"
        run_options.append(label)
        id_map[label] = (rid, coeffs, roots, ts, calc_name, version)

    if not run_options:
        st.info("No saved runs yet for your account.")
    else:
        sel = st.selectbox("Select a saved run", run_options, key=f"run_sel_{key_ns}")
        rid, coeffs_saved, roots_saved, ts, calc_name, version = id_map[sel]

        b1, b2, b3 = st.columns([1, 1, 2])

        with b1:
            if st.button("Reuse", key=f"reuse_{key_ns}"):
                st.session_state.reuse_coeffs = coeffs_saved
                st.rerun()


        with b2:
            rename_open = st.session_state.get(f"show_rename_{key_ns}", False)

            if st.button("Rename" if not rename_open else "Cancel rename", key=f"rename_toggle_{key_ns}"):
                st.session_state[f"show_rename_{key_ns}"] = not rename_open
                st.rerun()


        with b3:
            csv_text = "created_at,username,coeffs,roots\n"
            csv_text += f"\"{ts}\",\"{st.session_state.username}\",\"{coeffs_saved}\",\"{roots_saved}\"\n"
            st.download_button(
                "Export selected run as CSV",
                data=csv_text.encode(),
                file_name=f"poly_run_{rid}.csv",
                mime="text/csv",
                key=f"csv_{key_ns}"
            )

        # -----------------------------
        # Rename panel (only when open)
        # -----------------------------
        if st.session_state.get(f"show_rename_{key_ns}", False):
            new_name = st.text_input(
                "New name",
                value=calc_name or "",
                key=f"rename_input_{key_ns}"
            )

            if st.button("Save as new version", key=f"save_rename_{key_ns}"):
                new_name = (new_name or "").strip()
                if not new_name:
                    st.error("Name cannot be empty.")
                else:
                    con = get_db()
                    cur = con.cursor()

                    cur.execute("""
                        SELECT COALESCE(MAX(version), 0)
                        FROM history
                        WHERE username = ? AND calc_name = ?
                    """, (st.session_state.username, new_name))
                    next_ver = cur.fetchone()[0] + 1

                    cur.execute("""
                        INSERT INTO history (username, coeffs, roots, created_at, calc_name, version)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        st.session_state.username,
                        coeffs_saved,
                        roots_saved,
                        now_iso(),
                        new_name,
                        next_ver
                    ))

                    con.commit()
                    con.close()

                    st.success("Saved as a new version.")
                    st.session_state[f"show_rename_{key_ns}"] = False
                    st.rerun()


    # =========================
# GRAPH CONTROLS + COMPARISON + EXPORT
    # =========================
    st.markdown("### Graph Controls")

    gc1, gc2, gc3, gc4, gc5, = st.columns([1, 1, 1, 1.2, 1.2,  ])
    with gc1:
        if st.button("Zoom in", key=f"zoom_in_{key_ns}"):
            a, b = st.session_state.xlim
            st.session_state.xlim = (a / 2, b / 2)
            st.rerun()

    with gc2:
        if st.button("Zoom out", key=f"zoom_out_{key_ns}"):
            a, b = st.session_state.xlim
            st.session_state.xlim = (a * 2, b * 2)
            st.rerun()

    with gc3:
        if st.button("Reset view", key=f"reset_view_{key_ns}"):
            st.session_state.xlim = (-10.0, 10.0)
            st.rerun()

    with gc4:
        st.session_state.comparison_mode = st.toggle(
            "Comparison mode",
            value=st.session_state.comparison_mode,
            key=f"cmp_toggle_{key_ns}"
        )

    with gc5:
        st.session_state.graph_fmt = st.selectbox(
            "Export image as",
            ["PNG", "JPEG"],
            index=0 if st.session_state.graph_fmt == "PNG" else 1,
            key=f"img_fmt_{key_ns}"
        )

    

        



    # =========================
    # GRAPH
    # =========================
    st.markdown("### Graph Window")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(0)
    ax.axvline(0)

    x = np.linspace(*st.session_state.xlim, 800)




        # Plot current input polynomial
    if st.session_state.coeffs:
        y = np.polyval(st.session_state.coeffs, x)
        ax.plot(x, y)

        ymin, ymax = np.min(y), np.max(y)
        pad = 0.1 * (ymax - ymin if ymax != ymin else 1)
        ax.set_ylim(ymin - pad,ymax + pad)


    if st.session_state.comparison_mode and len(run_options) >= 2:
        a, b = st.columns(2)
        with a:
            sel_a = st.selectbox("Run A", run_options)
        with b:
            sel_b = st.selectbox("Run B", run_options)

        if sel_a != sel_b:
            _, ca, _, _, na, va = id_map[sel_a]
            _, cb, _, _, nb, vb = id_map[sel_b]

            ax.plot(x, np.polyval(parse_coeffs(ca), x), label=f"{na or 'A'} v{va}")
            ax.plot(x, np.polyval(parse_coeffs(cb), x), label=f"{nb or 'B'} v{vb}")
            ax.legend()
   


    st.pyplot(fig)

    # =========================
    # DOWNLOAD GRAPH
    # =========================
    img_bytes = fig_bytes(fig, st.session_state.graph_fmt)
    st.download_button(
        f"Download graph as {st.session_state.graph_fmt}",
        data=img_bytes,
        file_name=f"polynomial_graph.{st.session_state.graph_fmt.lower()}",
        mime="image/png" if st.session_state.graph_fmt == "PNG" else "image/jpeg"
    )

def view_users_view():
    st.subheader("All Users")


    with st.expander("Database info"):
        if USE_POSTGRES:
            st.info("Mirror database is disabled when using PostgreSQL.")
            mirror_enabled = False
        else:
            mirror_enabled = True
        primary_path = os.path.abspath(DB_PATH)
        st.write(f"Primary database file: `{primary_path}`")
        primary_exists = os.path.exists(DB_PATH)
        st.write(f"Primary database exists: **{primary_exists}**")
        if primary_exists:
            primary_size_kb = os.path.getsize(DB_PATH) / 1024
            st.write(f"Primary database size: **{primary_size_kb:.1f} KB**")
            with open(DB_PATH, "rb") as db_file:
                st.download_button(
                    "Download primary database (SQLite)",
                    data=db_file.read(),
                    file_name=os.path.basename(DB_PATH),
                    mime="application/x-sqlite3"
                )

        if mirror_enabled:
            mirror_path = os.path.abspath(MIRROR_DB_PATH)
            st.write(f"Mirror database file: `{mirror_path}`")
            mirror_exists = os.path.exists(MIRROR_DB_PATH)
            st.write(f"Mirror database exists: **{mirror_exists}**")
            if not mirror_exists:
                st.warning(
                    "Mirror database not found. Use the button below to create it "
                    "and sync current users."
                )
                if st.button("Create & sync mirror database"):
                    try:
                        ensure_mirror_schema()
                        sync_users_to_mirror()
                        st.success("Mirror database created and synced.")
                    except (sqlite3.Error, OSError) as exc:
                        st.error(f"Unable to create mirror database: {exc}")
            if mirror_exists:
                mirror_size_kb = os.path.getsize(MIRROR_DB_PATH) / 1024
                st.write(f"Mirror database size: **{mirror_size_kb:.1f} KB**")
                try:
                    mirror_con = get_mirror_db()
                    mirror_count = mirror_con.execute(
                        "SELECT COUNT(*) FROM users_mirror"
                    ).fetchone()[0]
                    mirror_con.close()
                    st.write(f"Mirror users: **{mirror_count}**")
                except sqlite3.Error:
                    st.write("Mirror users: **Unavailable**")

                with open(MIRROR_DB_PATH, "rb") as mirror_file:
                    st.download_button(
                        "Download mirror database (SQLite)",
                        data=mirror_file.read(),
                        file_name=os.path.basename(MIRROR_DB_PATH),
                        mime="application/x-sqlite3"
                    )

            if st.button("Sync users to mirror database"):
                sync_users_to_mirror()
                st.success("Mirror database synced with current users.")
    con = get_db()
    rows = con.execute(
        "SELECT username, role, first_login FROM users"
    ).fetchall()
    con.close()

    df = pd.DataFrame(
        [
            {
                "Username": r[0],
                "Role": r[1],
                "First Login": "Yes" if r[2] else "No"
            }
            for r in rows
           ]
    )

    def highlight_columns(column):
        if column.name == "Username":
            return ["background-color:#dcfce7"] * len(column)
        if column.name == "Role":
            return ["background-color:#fef9c3"] * len(column)
        if column.name == "First Login":
            return ["background-color:#ffedd5"] * len(column)
        return [""] * len(column)

    st.dataframe(
        df.style.apply(highlight_columns, axis=0), 
        use_container_width=True
    )

from datetime import datetime, timedelta
import pandas as pd
import streamlit as st


def admin_stats_view():

    inactivity_days = st.slider(
        "Inactivity threshold (days)",
        min_value=7,
        max_value=180,
        value=30,
        step=7
    )

    st.subheader("User Statistics Overview")

    col1, col2 = st.columns(2)
    with col1:
        role_filter = st.selectbox("Filter by role", ["All", "admin", "user"])
    with col2:
        activity_filter = st.selectbox("Filter by activity", ["All", "Active", "Inactive"])

    # ---------- Load data ----------
    con = get_db()
    rows = con.execute("""
        SELECT
            u.username,
            u.role,
            u.phone,
            u.email,
            u.first_login,
            u.last_login,
            COUNT(h.id) AS total_solves,
            MAX(h.created_at) AS last_solve
        FROM users u
        LEFT JOIN history h ON u.username = h.username
        GROUP BY u.username
        ORDER BY u.username

    """).fetchall()
    con.close()

    df = pd.DataFrame(
        rows,
        columns=[
            "username",
            "role",
            "phone",
            "email",
            "first_login",
            "last_login",
            "total_solves",
            "last_solve"
        ]

    )

    # ---------- Time parsing ----------
    cutoff = datetime.now() - timedelta(days=inactivity_days)

    df["last_login_dt"] = pd.to_datetime(df["last_login"], errors="coerce")
    df["last_solve_dt"] = pd.to_datetime(df["last_solve"], errors="coerce")

    # ---------- Last activity ----------
    df["last_activity"] = df[["last_login_dt", "last_solve_dt"]].max(axis=1, skipna=True)


    # ---------- Active / Inactive ----------
    df["Inactive"] = df["last_activity"].apply(
        lambda x: True if x is None or x < cutoff else False
    )

    df["Inactive Label"] = df["Inactive"].map({True: "Yes", False: "No"})

    # ---------- Since columns ----------
    df["Inactive Since"] = df.apply(
        lambda r: (
            r["last_activity"].strftime("%Y-%m-%d")
        if (
            not r["Inactive"]
            and pd.notna(r["last_activity"])
        )
        else ""
        ),
        axis=1
    )

    df["Active Since"] = df.apply(
        lambda r: (
            r["last_activity"].strftime("%Y-%m-%d")
            if (
                not r["Inactive"]
                and pd.notna(r["last_activity"])
            )
            else ""
        ),
        axis=1
    )




    # ---------- Normalised display ----------
    df["First Login"] = df["first_login"].apply(
        lambda x: "Pending" if x == 1 else "Completed"
    )

    df_display = df.rename(columns={
        "username": "Username",
        "role": "Role",
        "phone": "Phone",
        "email": "Email",
        "last_login": "Last Login",
        "total_solves": "Total Solves"
    })[
        [
            "Username",
            "Role",
            "Phone",
            "Email",
            "First Login",
            "Last Login",
            "Total Solves",
            "Active Since",
            "Inactive Since",
            "Inactive Label"
        ]
    ]

    # ---------- Filters ----------
    if role_filter != "All":
        df_display = df_display[df_display["Role"] == role_filter]

    if activity_filter == "Active":
        df_display = df_display[df_display["Inactive Label"] == "No"]
    elif activity_filter == "Inactive":
        df_display = df_display[df_display["Inactive Label"] == "Yes"]

    # ---------- Sorting ----------
    df_display = df_display.sort_values(
        by=["Role", "Total Solves"],
        ascending=[True, False]
    ).reset_index(drop=True)

    # ---------- Summary ----------
    total_users = len(df_display)
    total_solves = int(df_display["Total Solves"].sum())
    inactive_users = int((df_display["Inactive Label"] == "Yes").sum())

    # ---------- Highlighting ----------
    def highlight_rows(row):
        if row["Role"] == "admin":
            return ["background-color:#658feb; color:#b3c9f5"] * len(row)
        if row["First Login"] == "Pending":
            return ["background-color:#fcf5b1"] * len(row)
        if row["Inactive Label"] == "Yes":
            return ["color:#b0c9f5"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_display.style.apply(highlight_rows, axis=1),
        use_container_width=True
    )

    # ---------- Metrics ----------
    st.markdown("### Usage Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Users", total_users)
    c2.metric("Total Solves", total_solves)
    c3.metric("Inactive Users", inactive_users)

def recovery_setup_view():
    st.subheader("Account Recovery Setup")

    with st.form("recovery_form"):
        q1 = st.selectbox(
            "Recovery question 1",
            RECOVERY_QUESTIONS,
            key="rq1"
        )

        q2 = st.selectbox(
            "Recovery question 2",
            RECOVERY_QUESTIONS,
            key="rq2"
        )

        a1 = st.text_input("Answer for question 1", type="password")
        a2 = st.text_input("Answer for question 2", type="password")

        submitted = st.form_submit_button("Save recovery details")

    if submitted:
        con = get_db()
        cur = con.cursor()
        cur.execute("""
            UPDATE users
            SET recovery_q1=?, recovery_q2=?,
                recovery_a1_hash=?, recovery_a2_hash=?,
                first_login=0
            WHERE username=?
        """, (
            q1, q2,
            hash_answer(a1),
            hash_answer(a2),
            st.session_state.username
        ))
        con.commit()
        con.close()

        # 🔑 THIS IS THE MISSING PART
        st.session_state.first_login = 0
        st.session_state.page = "solver"

        st.success("Recovery details saved.")
        st.rerun()


def password_recovery_view():
    st.subheader("Account Recovery")
    st.markdown("<div class=\"auth-scope\">", unsafe_allow_html=True)

    # Back navigation
    if st.button("Back to login"):
        st.session_state.page = "login"
        st.markdown("</div>", unsafe_allow_html=True)
        st.rerun()

    # Initialise state
    st.session_state.setdefault("recovery_verified", False)
    st.session_state.setdefault("reset_user", None)

    username = st.text_input("Enter your username")

    if not username:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    con = get_db()
    cur = con.cursor()

    cur.execute("""
        SELECT recovery_q1, recovery_q2,
               recovery_a1_hash, recovery_a2_hash
        FROM users
        WHERE username=?
    """, (username,))
    row = cur.fetchone()
    con.close()

    if not row:
        st.error("User not found.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    q1, q2, h1, h2 = row

    if not q1 or not q2 or not h1 or not h2:
        st.error("Recovery questions are not set for this account.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # -----------------------------
    # STEP 1: VERIFY ANSWERS
    # -----------------------------
    if not st.session_state.recovery_verified:
        st.markdown("### Answer your recovery questions")

        with st.form("recovery_verify_form"):
            st.markdown(f"**{q1}**")
            ans1 = st.text_input("Answer 1", type="password")

            st.markdown(f"**{q2}**")
            ans2 = st.text_input("Answer 2", type="password")

            submitted = st.form_submit_button("Verify answers")

        if submitted:
            if hash_answer(ans1) == h1 and hash_answer(ans2) == h2:
                st.session_state.recovery_verified = True
                st.session_state.reset_user = username
                st.markdown("</div>", unsafe_allow_html=True)
                st.rerun()
            else:
                st.error("Recovery answers do not match.")
                
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # -----------------------------
    # STEP 2: RESET PASSWORD
    # -----------------------------
    st.success("Identity verified. Set a new password.")

    new_pw = st.text_input("New password", type="password")
    confirm_pw = st.text_input("Confirm password", type="password")

    if st.button("Reset password"):
        if not new_pw or not confirm_pw:
            st.error("Both fields are required.")
        elif new_pw != confirm_pw:
            st.error("Passwords do not match.")
        else:
            ok, msg = update_password(st.session_state.reset_user, new_pw)
            if not ok:
                st.error(msg)
                st.markdown("</div>", unsafe_allow_html=True)
                return

            st.success("Password reset successful. You can now log in.")

            # Cleanup
            st.session_state.recovery_verified = False
            st.session_state.reset_user = None
            st.session_state.page = "login"
            st.markdown("</div>", unsafe_allow_html=True)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def signup_view():
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.subheader("Create an account")
        st.markdown("<div class=\"auth-scope\">", unsafe_allow_html=True)

        with st.container(border=True):
            with st.form("signup_form"):
                u = st.text_input("Username", key="signup_user").strip()
                p = st.text_input("Password", type="password", key="signup_pass")
                phone = st.text_input("Phone number (required)", key="signup_phone")
                email = st.text_input("Email (optional)", key="signup_email")
                action_col, back_col = st.columns([1, 1])
                with action_col:
                    submitted = st.form_submit_button("Create account")
                with back_col:
                    back_clicked = st.form_submit_button("Back to login")

            if back_clicked:
                st.session_state.page = "login"
                st.markdown("</div>", unsafe_allow_html=True)
                st.rerun()

    if not submitted:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if not u or not p or not phone:
        st.error("Username, password, and phone number are required.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    ok, msg = validate_password_strength(p)
    if not ok:
        st.error(msg)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    ok, msg = create_user(
        username=u,
        pw=p,
        role="user",
        phone=phone,
        email=email if email else None
    )

    if ok:
        st.success("Account created successfully. Please log in.")
        st.session_state.page = "login"
        st.markdown("</div>", unsafe_allow_html=True)
        st.rerun()
    else:
        st.error(msg)

    st.markdown("</div>", unsafe_allow_html=True)

   

def create_user_view():
    
    st.markdown("<h2 style=\"text-align: center;\">Create User</h2>", unsafe_allow_html=True)

    
    left, center, right = st.columns([1, 2, 1])
    with center:
     
        with st.container(border=True):
            col_a, col_b = st.columns(2)
            with col_a:
                u = st.text_input("Username", key="create_user_name")
            with col_b:
                p = st.text_input("Temporary Password", type="password", key="create_user_pass")

            col_c, col_d = st.columns(2)
            with col_c:
                r = st.selectbox("Role", ["user", "admin"], key="create_user_role")
            with col_d:
                phone = st.text_input("Phone number (required)", key="create_user_phone")

            email = st.text_input("Email (optional)", key="create_user_email")

            if st.button("Create user"):
                if not u or not p or not phone:
                    st.error("Username, password, and phone number are required.")
                    return

                ok, msg = validate_password_strength(p)
                if not ok:
                    st.error(msg)
                    return

                ok, msg = create_user(
                    username=u,
                    pw=p,
                    role=r,
                    phone=phone,
                    email=email if email else None
                )

                if ok:
                    st.success(msg)
                else:
                    st.error(msg)



def delete_user_view():
    st.subheader("Delete User")

    u = st.text_input("Username to delete", key="delete_user_name")

    if st.checkbox("Confirm delete"):
        if st.button("Delete", key="delete_user_btn"):
            ok, msg = delete_user(u)
            if ok:
                st.success(msg)
            else:
                st.error(msg)


def admin_view():
    top_right_menu()
    st.title("Admin Dashboard")

    # ===============================
    # ROUTE HISTORY FIRST (CRITICAL)
    # ===============================
    if st.session_state.page == "history":
        admin_history_view()
        return

    # -------------------------------
    # Admin navigation (solver/users/etc)
    # -------------------------------
    page_map = {
        "Polynomial Solver": "solver",
        "View Users": "users",
        "User Statistics": "stats",
        "Create User": "create_user",
        "Delete User": "delete_user",
    }

    reverse_page_map = {v: k for k, v in page_map.items()}

    current_label = reverse_page_map.get(
        st.session_state.page,
        "Polynomial Solver"
    )

    action = st.selectbox(
        "Admin action",
        list(page_map.keys()),
        index=list(page_map.keys()).index(current_label),
        key="admin_action_select"
    )

    new_page = page_map[action]
    if new_page != st.session_state.page:
        st.session_state.page = new_page
        st.rerun()

    # -------------------------------
    # Page routing
    # -------------------------------
    if st.session_state.page == "solver":
        solver_view()

    elif st.session_state.page == "users":
        view_users_view()

    elif st.session_state.page == "stats":
        admin_stats_view()
  


    elif st.session_state.page == "create_user":
        create_user_view()

    elif st.session_state.page == "delete_user":
        delete_user_view()





            

def admin_history_view():
    st.subheader("System-wide Calculation History")

    back_to_solver()
    st.divider()

    con = get_db()
    rows = con.execute("""
        SELECT username, coeffs, roots, created_at, calc_name, version

        FROM history
        ORDER BY created_at DESC
        LIMIT 100
    """).fetchall()
    con.close()

    if not rows:
        st.info("No calculations recorded yet.")
        return

    for user, coeffs, roots, ts, calc_name, version in rows:
        display_name = calc_name or f"Calculation v{version}"

        with st.expander(f"{display_name} — {user}"):
            st.markdown("**User**")
            st.code(user)

            st.markdown("**Coefficients**")
            st.code(coeffs)

            st.markdown("**Roots**")
            st.code(roots)


def user_history_view():
    st.subheader("My Calculation History")

    back_to_solver()
    st.divider()

    con = get_db()
    rows = con.execute("""
        SELECT coeffs, roots, created_at, calc_name, version

        FROM history
        WHERE username = ?
        ORDER BY created_at DESC
    """, (st.session_state.username,)).fetchall()
    con.close()

    if not rows:
        st.info("You have no saved calculations yet.")
        return
    for coeffs, roots, ts, calc_name, version in rows:
        display_name = calc_name or f"Calculation v{version}"

        with st.expander(display_name):
            st.markdown("**Coefficients**")
            st.code(coeffs)
            st.markdown("**Roots**")
            st.code(roots if roots else "No roots stored.")



 
def user_view():
    top_right_menu()
    st.title("User Dashboard")

    user_stats_view()

    if st.session_state.page == "solver":
        solver_view()

    elif st.session_state.page == "history":
        user_history_view()

        
def top_right_menu():
    if not st.session_state.get("logged_in", False):
        return
   
    cols = st.columns([0.7, 0.15, 0.15])
    history_col, logout_col = cols[1], cols[2]

    with history_col:
        if st.button("History", key="history_btn"):
            st.session_state.page = "history"
            st.rerun()


    with logout_col:
        if st.button("Logout", key="logout_btn"):
            logout()



# ======================================================
# Main
# ======================================================
ensure_schema()
init_state()

st.session_state.setdefault("show_history", False)

if not st.session_state.logged_in:
    if st.session_state.page == "recover":
        password_recovery_view()
    elif st.session_state.page == "signup":
        signup_view()
    else:
        login_view()


else:
    # User is logged in
    if st.session_state.role == "user" and st.session_state.first_login == 1:
        # Force recovery setup before anything else
        recovery_setup_view()

    else:
        # Normal routing
        if st.session_state.role == "admin":
            admin_view()
        else:
            user_view()



