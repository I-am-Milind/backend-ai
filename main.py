import os
import io
import socket
import sqlite3
from typing import Optional, List
from datetime import datetime, date, timedelta

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from groq import Groq
from PIL import Image
import pytesseract

from search import web_search
from sentence_transformers import SentenceTransformer
import chromadb

# =========================================================
# App Setup
# =========================================================

app = FastAPI(title="Companion AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Groq Client
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable not set")

groq_client = Groq(api_key=GROQ_API_KEY)

# =========================================================
# Persona State
# =========================================================

ACTIVE_PERSONA = {
    "name": None,
    "rules": None,
}

# =========================================================
# Chat Memory (short-term)
# =========================================================

MEMORY: List[dict] = []

# =========================================================
# Fact Memory (SQLite with refresh)
# =========================================================

DB = "fact_memory.db"
STALE_DAYS = 1

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

def save_fact(key: str, answer: str, sources: list):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO facts
        VALUES (?, ?, ?, ?)
    """, (key, answer, str(sources), str(date.today())))
    conn.commit()
    conn.close()

init_db()

# =========================================================
# Vector Semantic Memory (ChatGPT-style recall)
# =========================================================

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
vector_store = chroma_client.get_or_create_collection("semantic_memory")

def store_semantic(text: str):
    embedding = embedder.encode(text).tolist()
    vector_store.add(
        documents=[text],
        embeddings=[embedding],
        ids=[str(hash(text))]
    )

def recall_semantic(query: str) -> Optional[str]:
    embedding = embedder.encode(query).tolist()
    result = vector_store.query(
        query_embeddings=[embedding],
        n_results=1
    )
    if result["documents"]:
        return result["documents"][0][0]
    return None

# =========================================================
# Verification + Confidence
# =========================================================

def verify_answer(answer: str, sources: list) -> bool:
    if not sources:
        return False
    red_flags = ["i think", "probably", "might be", "guess", "not sure"]
    for f in red_flags:
        if f in answer.lower():
            return False
    return True

def confidence_score(sources: list, verified: bool) -> float:
    score = 0.4
    if sources:
        score += 0.3
    if len(sources) >= 2:
        score += 0.2
    if verified:
        score += 0.1
    return round(min(score, 0.99), 2)

# =========================================================
# Models
# =========================================================

class ChatRequest(BaseModel):
    message: str

class PersonaTextRequest(BaseModel):
    text: str
    name: Optional[str] = None

# =========================================================
# Utilities
# =========================================================

def get_facts():
    return {
        "date": datetime.now().strftime("%B %d, %Y"),
        "time": datetime.now().strftime("%H:%M"),
    }

def needs_live_data(message: str) -> bool:
    keywords = ["today", "latest", "current", "now", "net worth", "price", "news"]
    return any(k in message.lower() for k in keywords)

def internet_available():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except:
        return False

# =========================================================
# Chat Endpoint (FULL SYSTEM)
# =========================================================

@app.post("/chat")
def chat(data: ChatRequest):
    user_message = data.message.strip()
    key = user_message.lower().replace(" ", "_")
    facts = get_facts()

    # 1️⃣ Semantic recall (FREE, no API)
    recalled = recall_semantic(user_message)
    if recalled:
        return JSONResponse({
            "answer": recalled,
            "confidence": 0.75,
            "mode": "semantic-memory"
        })

    cached = get_fact(key)

    # 2️⃣ Live data path
    if needs_live_data(user_message):
        if cached and not is_stale(cached[2]):
            verified = verify_answer(cached[0], eval(cached[1]))
            return JSONResponse({
                "answer": cached[0],
                "sources": eval(cached[1]),
                "confidence": confidence_score(eval(cached[1]), verified),
                "mode": "cached"
            })

        if internet_available():
            result = web_search(user_message)
            verified = verify_answer(result["answer"], result["sources"])

            if verified:
                save_fact(key, result["answer"], result["sources"])
                store_semantic(result["answer"])

            return JSONResponse({
                "answer": result["answer"] if verified else "I don’t have verified information for that yet.",
                "sources": result["sources"],
                "confidence": confidence_score(result["sources"], verified),
                "mode": "live"
            })

        if cached:
            return JSONResponse({
                "answer": cached[0],
                "sources": eval(cached[1]),
                "confidence": 0.6,
                "mode": "offline-stale"
            })

        return JSONResponse({
            "answer": "I don’t have verified information for that yet.",
            "confidence": 0.4,
            "mode": "offline"
        })

    # 3️⃣ Normal Groq streaming chat (persona preserved)
    system_prompt = f"""
You are an AI persona.
You NEVER guess facts.

Today's date is {facts['date']}
Current time is {facts['time']}
"""

    if ACTIVE_PERSONA["rules"]:
        system_prompt += f"\nPERSONA RULES:\n{ACTIVE_PERSONA['rules']}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    def stream():
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.6,
            stream=True,
        )

        full_reply = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                full_reply += delta
                yield delta

        MEMORY.append({
            "user": user_message,
            "ai": full_reply,
            "timestamp": facts["date"]
        })

    return StreamingResponse(stream(), media_type="text/plain")
