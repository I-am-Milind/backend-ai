import os
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq
from PIL import Image
import pytesseract

# -----------------------------
# APP MUST BE FIRST
# -----------------------------
app = FastAPI()

# -----------------------------
# CORS MUST COME IMMEDIATELY
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "*",  # allow all for now (production safe later)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# GROQ CLIENT
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing")

client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# MODELS
# -----------------------------
class ChatRequest(BaseModel):
    message: str

class PersonaText(BaseModel):
    text: str
    name: str | None = None

# -----------------------------
# HEALTH (THIS IS REQUIRED)
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# CHAT
# -----------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",  # ACTIVE
            messages=[
                {"role": "system", "content": "You are a helpful AI companion."},
                {"role": "user", "content": req.message},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        return response.choices[0].message.content
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -----------------------------
# PERSONA FROM TEXT
# -----------------------------
@app.post("/persona/extract")
def persona_from_text(req: PersonaText):
    try:
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {"role": "system", "content": "Extract personality from text."},
                {"role": "user", "content": req.text},
            ],
            temperature=0.4,
            max_tokens=300,
        )
        return {
            "created_persona": req.name or "Unknown",
            "persona": completion.choices[0].message.content,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -----------------------------
# PERSONA FROM IMAGE
# -----------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # ✅ ACTIVE MODEL
            messages=[
                {"role": "system", "content": "You are a helpful AI companion."},
                {"role": "user", "content": req.message},
            ],
            temperature=0.7,
            max_tokens=512,
        )

        return completion.choices[0].message.content

    except Exception as e:
        print("CHAT ERROR:", str(e))  # 👈 IMPORTANT
        return JSONResponse(
            status_code=500,
            content={"error": "Groq request failed", "detail": str(e)},
        )
