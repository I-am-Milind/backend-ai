import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import pytesseract
from PIL import Image
import io

# --------------------------------------------------
# ENV
# --------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set")

client = Groq(api_key=GROQ_API_KEY)

# --------------------------------------------------
# APP
# --------------------------------------------------
app = FastAPI(title="Companion AI Backend")

# --------------------------------------------------
# CORS (THIS FIXES YOUR ISSUE PERMANENTLY)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "https://your-frontend-domain.vercel.app",  # add later
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# MODELS
# --------------------------------------------------
class ChatRequest(BaseModel):
    message: str

class PersonaExtractRequest(BaseModel):
    text: str
    name: str | None = None

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "cloud",
        "provider": "groq"
    }

# --------------------------------------------------
# CHAT ENDPOINT
# --------------------------------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",  # ACTIVE MODEL
            messages=[
                {
                    "role": "system",
                    "content": "You are a calm, helpful AI companion."
                },
                {
                    "role": "user",
                    "content": req.message
                }
            ],
            temperature=0.7,
            max_tokens=512,
        )

        reply = completion.choices[0].message.content
        return reply

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# --------------------------------------------------
# PERSONA FROM TEXT
# --------------------------------------------------
@app.post("/persona/extract")
def extract_persona(req: PersonaExtractRequest):
    try:
        prompt = f"""
Extract personality traits, tone, and preferences from this text.
Return a short persona description.

Text:
{req.text}
"""

        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {"role": "system", "content": "You extract personas from text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=300
        )

        persona = completion.choices[0].message.content

        return {
            "created_persona": req.name or "Unknown",
            "persona": persona
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --------------------------------------------------
# PERSONA FROM IMAGE (SCREENSHOT OCR)
# --------------------------------------------------
@app.post("/persona/from-image")
async def persona_from_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        extracted_text = pytesseract.image_to_string(image)

        prompt = f"""
Analyze this chat screenshot text and extract personality,
tone, habits, emotional style.

Text:
{extracted_text}
"""

        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {"role": "system", "content": "You extract personas from chat logs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=400
        )

        persona = completion.choices[0].message.content

        return {
            "created_persona": "From Image",
            "extracted_text_preview": extracted_text[:500],
            "persona": persona
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
