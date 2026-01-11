import sqlite3
from datetime import date, timedelta

DB = "fact_memory.db"
STALE_DAYS = 1  # refresh once per day

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            key TEXT PRIMARY KEY,
            answer TEXT,
            sources TEXT,
            updated TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_fact(key: str):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT answer, sources, updated FROM facts WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    return row

def is_stale(updated: str) -> bool:
    return date.fromisoformat(updated) < date.today() - timedelta(days=STALE_DAYS)

def save_fact(key, answer, sources):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO facts
        VALUES (?, ?, ?, ?)
    """, (key, answer, str(sources), str(date.today())))
    conn.commit()
    conn.close()
