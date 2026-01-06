from db import init_db, get_connection

def seed_admin():
    init_db()
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM users")
    count = cur.fetchone()[0]

    if count == 0:
        cur.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ("ad", "ad", "admin")
        )

    conn.commit()
    conn.close()

if __name__ == "__main__":
    seed_admin()
