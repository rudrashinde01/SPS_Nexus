from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import faiss
import pickle
import re
import os
import threading

# Load .env file
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global variables ──
embed_model = None
index = None
texts = None

# ── Load only index + texts first (FAST) ──
def load_basic():
    global index, texts
    if index is None:
        index = faiss.read_index("vector.index")
    if texts is None:
        with open("texts.pkl", "rb") as f:
            texts = pickle.load(f)

# ── Load model separately (HEAVY) ──
def load_model():
    global embed_model
    if embed_model is None:
        try:
            embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            print("✅ Model loaded")
        except Exception as e:
            print("❌ Model load error:", e)

# ── Background loading (NON-BLOCKING) ──
def preload():
    load_basic()
    load_model()

threading.Thread(target=preload).start()

# Key loads from .env
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []
    studentProfile: Optional[Dict[str, str]] = {}

GREETINGS = [
    "hi", "hello", "hey", "hii", "helo", "hlo",
    "good morning", "good afternoon", "good evening",
    "good night", "sup", "wassup", "namaste"
]

ADMIN_PATTERNS = [
    r'\btime\s*table\b',
    r'\bclass\s*schedule\b',
    r'\bmy\s+schedule\b',
    r'\battendance\b',
    r'\bmy\s+attendance\b',
    r'\bmy\s+marks\b',
    r'\binternal\s*marks\b',
    r'\bpractical\s*marks\b',
    r'\bsessional\s*marks\b',
    r'\bterm\s*work\b',
    r'\bnotice\b',
    r'\bannouncement\b',
    r'\btoday.*notice\b',
    r'\blatest.*notice\b',
    r'\bmy\s+result\b',
    r'\bresult\b',
    r'\bexam\s*schedule\b',
    r'\bexam\s*date\b',
    r'\bhall\s*ticket\b',
    r'\badmit\s*card\b',
    r'\bfee\b',
    r'\bfee\s*payment\b',
    r'\bmy\s+doubt\b',
    r'\bmy\s+doubts\b',
    r'\bstudy\s*material\b',
    r'\bsyllabus\b',
    r'\bholiday\b',
    r'\bacademic\s*calendar\b',
    r'\blab\s*schedule\b',
    r'\bassignment\s*deadline\b',
    r'\bpending\s*assignment\b',
]

def is_greeting(message: str) -> bool:
    msg = message.lower().strip()
    return msg in GREETINGS or any(msg.startswith(g) for g in GREETINGS)

def is_admin_query(message: str) -> bool:
    for pattern in ADMIN_PATTERNS:
        if re.search(pattern, message.lower()):
            return True
    return False

# ✅ Fast route (for Render port detection)
@app.get("/")
async def root():
    return {"status": "Nexus AI server is running!"}

# ✅ Optional ping
@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    load_basic()  # fast load

    if embed_model is None:
        return {"reply": "Server is starting... please try again in a few seconds."}

    msg = req.message.strip()

    if is_greeting(msg):
        return {
            "reply": "Hello! I am Nexus AI, your MSBTE diploma assistant. Ask me anything about your diploma subjects!"
        }

    if is_admin_query(msg):
        return {
            "reply": "For this information please check the SPS Nexus app!"
        }

    vector = embed_model.encode([msg])
    D, I = index.search(vector, 3)

    valid_indices = [i for i in I[0] if 0 <= i < len(texts)]
    context = "\n\n".join([texts[i] for i in valid_indices])

    if not context.strip():
        return {
            "reply": "This topic is outside my diploma syllabus."
        }

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are Nexus AI. Answer only from study material. Keep answers short."
            },
            {
                "role": "user",
                "content": f"Study material:\n{context}\n\nQuestion: {msg}"
            }
        ],
        max_tokens=512,
        temperature=0.3,
    )

    reply = response.choices[0].message.content.strip()
    return {"reply": reply}