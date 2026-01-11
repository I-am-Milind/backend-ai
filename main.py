import os
import io
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from groq import Groq
from PIL import Image
import pytesseract

# =========================
# App Setup
# =========================

app = FastAPI(title="Companion AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Groq Client
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable not set")

groq_client = Groq(api_key=GROQ_API_KEY)

# =========================
# Persona State
# =========================

ACTIVE_PERSONA = {
    "name": None,
    "rules": None,
}

# =========================
# Memory (temporary, RAM)
# =========================

MEMORY: List[dict] = []

# =========================
# Models
# =========================

class ChatRequest(BaseModel):
    message: str


class PersonaTextRequest(BaseModel):
    text: str
    name: Optional[str] = None

# =========================
# Utilities
# =========================

def get_facts():
    """System owned facts (never from AI)"""
    return {
        "date": datetime.now().strftime("%B %d, %Y"),
        "time": datetime.now().strftime("%H:%M"),
    }


def is_fact_question(message: str) -> bool:
    keywords = [
        "date", "time", "today", "day",
        "current", "now"
    ]
    return any(k in message.lower() for k in keywords)


# =========================
# Health Check
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "cloud",
        "provider": "groq"
    }

# =========================
# Persona from Text
# =========================

@app.post("/persona/extract")
def extract_persona(data: PersonaTextRequest):
    ACTIVE_PERSONA["name"] = data.name or "Persona"
    ACTIVE_PERSONA["rules"] = data.text.strip()

    return {
        "created_persona": ACTIVE_PERSONA["name"],
        "rules": ACTIVE_PERSONA["rules"]
    }

# =========================
# Persona from Image (OCR)
# =========================

@app.post("/persona/from-image")
async def persona_from_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    extracted_text = pytesseract.image_to_string(image)

    if not extracted_text.strip():
        return {
            "created_persona": None,
            "extracted_text_preview": ""
        }

    ACTIVE_PERSONA["name"] = file.filename.split(".")[0] or "Persona"
    ACTIVE_PERSONA["rules"] = extracted_text.strip()

    return {
        "created_persona": ACTIVE_PERSONA["name"],
        "extracted_text_preview": extracted_text[:500]
    }

# =========================
# Chat Endpoint (SAFE)
# =========================

@app.post("/chat")
def chat(data: ChatRequest):
    user_message = data.message.strip()
    facts = get_facts()

    # 🔒 FACT QUESTIONS (NO AI)
    if is_fact_question(user_message):
        if "date" in user_message.lower() or "today" in user_message.lower():
            return StreamingResponse(
                iter([f"Today's date is {facts['date']}"]),
                media_type="text/plain"
            )
        if "time" in user_message.lower():
            return StreamingResponse(
                iter([f"The current time is {facts['time']}"]),
                media_type="text/plain"
            )

    # =========================
    # SYSTEM PROMPT (STRONG)
    # =========================

    system_prompt = f"""
You are an AI persona.

ABSOLUTE RULES:
- You are NOT a language model
- You NEVER mention training data
- You NEVER explain AI limitations
- You NEVER apologize for being an AI
- You NEVER guess facts

SYSTEM FACTS (TRUST THESE ONLY):
- Today's date is {facts['date']}
- Current time is {facts['time']}

If you are unsure, say:
"I don’t have verified information for that yet."

"""

    if ACTIVE_PERSONA["rules"]:
        system_prompt += f"""
PERSONA RULES (MANDATORY):
{ACTIVE_PERSONA["rules"]}

NEVER break character.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # =========================
    # STREAM RESPONSE
    # =========================

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

        # 🧠 STORE MEMORY
        MEMORY.append({
            "user": user_message,
            "ai": full_reply,
            "timestamp": facts["date"]
        })

    return StreamingResponse(stream(), media_type="text/plain")
