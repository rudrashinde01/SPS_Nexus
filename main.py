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
import uvicorn

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

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("vector.index")
with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)

# Key loads from .env — never in code!
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
    return msg in GREETINGS or any(
        msg.startswith(g) for g in GREETINGS
    )

def is_admin_query(message: str) -> bool:
    for pattern in ADMIN_PATTERNS:
        if re.search(pattern, message.lower()):
            return True
    return False


@app.post("/chat")
async def chat(req: ChatRequest):
    msg = req.message.strip()

    if is_greeting(msg):
        return {
            "reply": "Hello! I am Nexus AI, your MSBTE diploma assistant. Ask me anything about your diploma subjects!"
        }

    if is_admin_query(msg):
        return {
            "reply": "For this information please check the **SPS Nexus app**! You can find it in the relevant section of your dashboard. 📱"
        }

    vector = embed_model.encode([msg])
    D, I = index.search(vector, 3)

    valid_indices = [i for i in I[0] if 0 <= i < len(texts)]
    context = "\n\n".join([texts[i] for i in valid_indices])

    if not context.strip():
        return {
            "reply": "Sorry! This topic is not in my diploma syllabus. Please ask me about your MSBTE subjects!"
        }

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are Nexus AI, a friendly academic assistant for MSBTE diploma students in Maharashtra, India.
Rules:
- Answer ONLY from the study material given
- Give clear and simple answers for diploma students
- Use bullet points where needed
- Do NOT repeat the question
- Do NOT show Context or Question labels
- If topic not in study material say: This topic is outside my diploma syllabus. Please ask about your MSBTE subjects!
- Keep answer short and to the point"""
            },
            {
                "role": "user",
                "content": f"Study material:\n{context}\n\nQuestion: {msg}"
            }
        ],
        max_tokens=1024,
        temperature=0.3,
    )

    reply = response.choices[0].message.content.strip()
    return {"reply": reply}


# ── This is the fix for Render deployment ──
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)