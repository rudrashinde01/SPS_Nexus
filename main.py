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

# Load lightweight resources ONLY
index = faiss.read_index("vector.index")

with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)

# Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []
    studentProfile: Optional[Dict[str, str]] = {}

GREETINGS = ["hi","hello","hey","hii","helo","hlo","namaste"]

ADMIN_PATTERNS = [
    r'\btime\s*table\b', r'\battendance\b', r'\bmarks\b',
    r'\bnotice\b', r'\bresult\b', r'\bexam\b'
]

def is_greeting(msg):
    msg = msg.lower().strip()
    return msg in GREETINGS or any(msg.startswith(g) for g in GREETINGS)

def is_admin_query(msg):
    return any(re.search(p, msg.lower()) for p in ADMIN_PATTERNS)

# Fast root route (IMPORTANT for Render)
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

    # 🔥 Lightweight FAISS search (no model)
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