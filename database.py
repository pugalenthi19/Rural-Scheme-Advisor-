# ============================================================
# database.py  —  Rural Scheme Advisor  (SQLite layer)
# ============================================================
import sqlite3
import hashlib

DB_PATH = "rural_advisor.db"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _hash(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()


def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


# ──────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────
def init_db():
    con = _conn()
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            role          TEXT    DEFAULT 'farmer',
            created_at    TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            session_id TEXT    NOT NULL,
            role       TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            timestamp  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER NOT NULL,
            question  TEXT    NOT NULL,
            rating    INTEGER NOT NULL,          -- 1 = helpful, 0 = not helpful
            timestamp TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS questions_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER NOT NULL,
            question  TEXT    NOT NULL,
            language  TEXT    DEFAULT 'en',
            timestamp TEXT    DEFAULT (datetime('now'))
        );
    """)

    # Seed default accounts
    for uname, pwd, role in [
        ("admin",   "admin123",  "officer"),
        ("farmer1", "farmer123", "farmer"),
        ("farmer2", "farmer456", "farmer"),
    ]:
        cur.execute(
            "INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?,?,?)",
            (uname, _hash(pwd), role),
        )
    con.commit()
    con.close()


# ──────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────
def verify_login(username: str, password: str):
    """Returns (id, username, role) or None."""
    con = _conn()
    row = con.execute(
        "SELECT id, username, role FROM users WHERE username=? AND password_hash=?",
        (username, _hash(password)),
    ).fetchone()
    con.close()
    return row


def register_user(username: str, password: str, role: str = "farmer") -> bool:
    con = _conn()
    try:
        con.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
            (username, _hash(password), role),
        )
        con.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        con.close()


# ──────────────────────────────────────────────
# Conversation persistence
# ──────────────────────────────────────────────
def save_conversation(user_id: int, session_id: str, role: str, content: str):
    con = _conn()
    con.execute(
        "INSERT INTO conversations (user_id, session_id, role, content) VALUES (?,?,?,?)",
        (user_id, session_id, role, content),
    )
    con.commit()
    con.close()


def log_question(user_id: int, question: str, language: str = "en"):
    con = _conn()
    con.execute(
        "INSERT INTO questions_log (user_id, question, language) VALUES (?,?,?)",
        (user_id, question, language),
    )
    con.commit()
    con.close()


def save_feedback(user_id: int, question: str, rating: int):
    con = _conn()
    con.execute(
        "INSERT INTO feedback (user_id, question, rating) VALUES (?,?,?)",
        (user_id, question, rating),
    )
    con.commit()
    con.close()


# ──────────────────────────────────────────────
# Analytics
# ──────────────────────────────────────────────
def get_analytics() -> dict:
    con = _conn()
    c = con.cursor()

    def one(sql, *args):
        return c.execute(sql, args).fetchone()[0]

    stats = {
        "total_users":     one("SELECT COUNT(*) FROM users"),
        "total_questions": one("SELECT COUNT(*) FROM questions_log"),
        "helpful":         one("SELECT COUNT(*) FROM feedback WHERE rating=1"),
        "not_helpful":     one("SELECT COUNT(*) FROM feedback WHERE rating=0"),
        "top_questions":   c.execute(
            "SELECT question, COUNT(*) cnt FROM questions_log "
            "GROUP BY question ORDER BY cnt DESC LIMIT 10"
        ).fetchall(),
        "by_language":     c.execute(
            "SELECT language, COUNT(*) cnt FROM questions_log GROUP BY language"
        ).fetchall(),
        "daily_activity":  c.execute(
            "SELECT DATE(timestamp), COUNT(*) FROM questions_log "
            "WHERE timestamp >= datetime('now','-7 days') "
            "GROUP BY DATE(timestamp) ORDER BY DATE(timestamp)"
        ).fetchall(),
    }
    con.close()
    return stats