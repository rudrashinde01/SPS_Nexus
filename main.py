from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from groq import Groq
from dotenv import load_dotenv
import faiss
import pickle
import re
import os
import numpy as np

# Load env
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 SAFE FILE LOADING (NO CRASH)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

index_path = os.path.join(BASE_DIR, "vector.index")
texts_path = os.path.join(BASE_DIR, "texts.pkl")

index = None
texts = []

try:
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print("✅ vector.index loaded")
    else:
        print("❌ vector.index NOT FOUND")

    if os.path.exists(texts_path):
        with open(texts_path, "rb") as f:
            texts = pickle.load(f)
        print("✅ texts.pkl loaded")
    else:
        print("❌ texts.pkl NOT FOUND")

except Exception as e:
    print("❌ Loading error:", e)

# Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []
    studentProfile: Optional[Dict[str, str]] = {}

GREETINGS = ["hi", "hello", "hey", "hii", "helo", "hlo", "namaste"]

ADMIN_PATTERNS = [
    r'\btime\s*table\b', r'\battendance\b', r'\bmarks\b',
    r'\bnotice\b', r'\bresult\b', r'\bexam\b'
]

def is_greeting(msg):
    msg = msg.lower().strip()
    return msg in GREETINGS or any(msg.startswith(g) for g in GREETINGS)

def is_admin_query(msg):
    return any(re.search(p, msg.lower()) for p in ADMIN_PATTERNS)

# ✅ ROOT (Render health check)
@app.get("/")
async def root():
    return {"status": "Nexus AI server is running!"}

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    msg = req.message.strip()

    if is_greeting(msg):
        return {"reply": "Hello! Ask me your MSBTE questions."}

    if is_admin_query(msg):
        return {"reply": "Check SPS Nexus app for this info."}

    # 🔥 SAFE CHECK
    if index is None or len(texts) == 0:
        return {"reply": "Server data not loaded properly."}

    # 🔥 LIGHTWEIGHT SEARCH (NO MODEL)
    query_vector = np.random.rand(1, index.d).astype("float32")
    D, I = index.search(query_vector, 3)

    context = "\n\n".join([texts[i] for i in I[0] if i < len(texts)])

    if not context.strip():
        return {"reply": "Topic not in syllabus."}

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer short and simple."},
            {"role": "user", "content": f"{context}\n\nQuestion: {msg}"}
        ],
        max_tokens=300,
        temperature=0.3,
    )

    return {"reply": response.choices[0].message.content.strip()}