import sqlite3
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "database" / "safetynet.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            cleaned_text TEXT,
            risk TEXT NOT NULL,
            confidence REAL,
            low_prob REAL,
            medium_prob REAL,
            high_prob REAL,
            mode TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_message(text, cleaned_text, risk, confidence, low_prob, medium_prob, high_prob, mode):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO messages (
            text, cleaned_text, risk, confidence,
            low_prob, medium_prob, high_prob,
            mode, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        text,
        cleaned_text,
        risk,
        confidence,
        low_prob,
        medium_prob,
        high_prob,
        mode,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_recent_messages(limit=50):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, text, cleaned_text, risk, confidence,
               low_prob, medium_prob, high_prob, mode, created_at
        FROM messages
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cur.fetchall()
    conn.close()
    return rows


def get_recent_high(limit=30):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, text, cleaned_text, risk, confidence,
               low_prob, medium_prob, high_prob, mode, created_at
        FROM messages
        WHERE UPPER(risk) = 'HIGH'
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cur.fetchall()
    conn.close()
    return rows


def get_risk_counts():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT UPPER(risk), COUNT(*)
        FROM messages
        GROUP BY UPPER(risk)
    """)

    rows = cur.fetchall()
    conn.close()

    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

    for risk, count in rows:
        if risk in counts:
            counts[risk] = count

    return counts
