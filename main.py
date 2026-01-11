import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq

# -----------------------
# App setup
# -----------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for MVP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Groq client
# -----------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing")

client = Groq(api_key=GROQ_API_KEY)

# -----------------------
# Models
# -----------------------
class ChatRequest(BaseModel):
    message: str

# -----------------------
# Health check
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------
# Chat endpoint
# -----------------------
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful AI companion."},
                {"role": "user", "content": req.message},
            ],
            temperature=0.7,
            max_tokens=512,
        )

        return {
            "reply": response.choices[0].message.content
        }

    except Exception as e:
        # IMPORTANT: return JSON, NOT crash
        return JSONResponse(
            status_code=500,
            content={
                "error": "AI failed",
                "detail": str(e)
            },
        )
