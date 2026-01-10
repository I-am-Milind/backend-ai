import os
import io
from typing import Optional

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
    allow_origins=["*"],  # tighten later for production
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
# In-Memory Persona State
# (Session-level for now)
# =========================

ACTIVE_PERSONA = {
    "name": None,
    "rules": None,
}

# =========================
# Models
# =========================

class ChatRequest(BaseModel):
    message: str


class PersonaTextRequest(BaseModel):
    text: str
    name: Optional[str] = None


# =========================
# Health Check (IMPORTANT)
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
# Chat (Streaming AI)
# =========================

@app.post("/chat")
def chat(data: ChatRequest):
    user_message = data.message.strip()

    system_prompt = "You are a helpful AI assistant."

    if ACTIVE_PERSONA["rules"]:
        system_prompt = f"""
You are role-playing as the following persona.
STRICTLY follow these rules at all times:

{ACTIVE_PERSONA["rules"]}

Never break character.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    def stream():
        completion = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",  # ✅ supported model
            messages=messages,
            temperature=0.7,
            stream=True,
        )

        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return StreamingResponse(stream(), media_type="text/plain")
