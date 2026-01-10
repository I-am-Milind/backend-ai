from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
import pytesseract
from PIL import Image
import os
import json
import uuid

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MEMORY_FILE = "memory.json"
PERSONA_FILE = "personas.json"
UPLOAD_DIR = "uploads"
MAX_SHORT_MEMORY = 6

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Tesseract path (Windows safe)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Groq client (API KEY from ENV)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

short_memory = []

# --------------------------------------------------
# APP
# --------------------------------------------------
app = FastAPI(title="AI Companion – Cloud Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# UTILS
# --------------------------------------------------
def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# --------------------------------------------------
# MODELS
# --------------------------------------------------
class ChatRequest(BaseModel):
    message: str

class PersonaRefineRequest(BaseModel):
    instruction: str

# --------------------------------------------------
# HEALTH
# --------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "cloud",
        "provider": "groq"
    }

# --------------------------------------------------
# OCR
# --------------------------------------------------
def ocr_image(path: str) -> str:
    img = Image.open(path)
    return pytesseract.image_to_string(img).strip()

# --------------------------------------------------
# PERSONA EXTRACTION
# --------------------------------------------------
def extract_persona_from_text(text: str):
    prompt = f"""
Analyze the following conversation or writing.

Infer personality, tone, and emotional behavior.
Assume emoji usage if emotional cues exist.

Return ONLY valid JSON:
{{
  "name": "Optional name",
  "description": "Short personality summary",
  "rules": [
    "Behavior rule",
    "Emoji rule"
  ]
}}

TEXT:
{text}
"""

    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {
            "description": "Warm, emotionally expressive personality",
            "rules": ["Speak gently", "Use emojis naturally"]
        }

# --------------------------------------------------
# PERSONA REFINEMENT
# --------------------------------------------------
def refine_persona_text(text: str):
    prompt = f"""
Turn this feedback into ONE short behavior refinement rule.
Do not change identity.

Feedback:
{text}
"""

    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

# --------------------------------------------------
# STREAMING CHAT
# --------------------------------------------------
def stream_response(user_message: str):
    global short_memory

    personas = load_json(PERSONA_FILE, {})
    memory = load_json(MEMORY_FILE, {"facts": []})

    active = personas.get("active_persona", "default")
    persona = personas["personas"].get(active, {})

    system_prompt = f"""
You are roleplaying as a REAL PERSON.

IDENTITY (LOCKED):
- Your name is {persona.get("name", "Unknown")}
- You are NOT the user
- You are NOT an AI
- Never break character

DESCRIPTION:
{persona.get("description", "")}

RULES:
"""

    for r in persona.get("rules", []):
        system_prompt += f"- {r}\n"

    if persona.get("refinements"):
        system_prompt += "\nREFINEMENTS:\n"
        for r in persona["refinements"]:
            system_prompt += f"- {r}\n"

    system_prompt += "\nIDENTITY RULES:\n"
    for r in persona.get("identity_rules", []):
        system_prompt += f"- {r}\n"

    if memory.get("facts"):
        system_prompt += "\nFACTS ABOUT USER:\n"
        for f in memory["facts"]:
            system_prompt += f"- {f}\n"

    short_memory.append({"role": "user", "content": user_message})
    short_memory[:] = short_memory[-MAX_SHORT_MEMORY:]

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(short_memory)

    try:
        stream = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            stream=True
        )
    except Exception:
        yield "⚠️ Cloud AI is unavailable. Try again later."
        return

    full = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            full += text
            yield text

    short_memory.append({"role": "assistant", "content": full})
    short_memory[:] = short_memory[-MAX_SHORT_MEMORY:]

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    return StreamingResponse(stream_response(req.message), media_type="text/plain")

@app.get("/persona")
def persona_state():
    return load_json(PERSONA_FILE, {})

@app.post("/persona/{name}")
def switch_persona(name: str):
    global short_memory
    data = load_json(PERSONA_FILE, {})
    if name not in data["personas"]:
        return {"error": "Persona not found"}

    data["active_persona"] = name
    save_json(PERSONA_FILE, data)
    short_memory = []

    return {"active_persona": name}

@app.post("/persona/refine")
def refine_persona(req: PersonaRefineRequest):
    data = load_json(PERSONA_FILE, {})
    active = data["active_persona"]

    rule = refine_persona_text(req.instruction)
    data["personas"][active]["refinements"].append(rule)

    save_json(PERSONA_FILE, data)
    return {"added_refinement": rule}

@app.post("/persona/from-image")
async def persona_from_image(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1]
    fname = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_DIR, fname)

    with open(path, "wb") as f:
        f.write(await file.read())

    text = ocr_image(path)
    persona = extract_persona_from_text(text)

    data = load_json(PERSONA_FILE, {})
    persona_name = persona.get("name") or f"persona_{uuid.uuid4().hex[:6]}"

    data["personas"][persona_name] = {
        "name": persona_name,
        "description": persona.get("description", ""),
        "rules": persona.get("rules", []),
        "identity_rules": [
            f"Your name is {persona_name}",
            "Never say you are an AI",
            "Never say you are the user",
            "Stay in character"
        ],
        "refinements": []
    }

    data["active_persona"] = persona_name
    save_json(PERSONA_FILE, data)

    return {
        "created_persona": persona_name,
        "preview": text[:500]
    }
