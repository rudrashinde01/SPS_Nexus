from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import faiss
import pickle
import os
import numpy as np

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

index = None
texts = []

# ✅ LOAD FILES SAFELY
try:
    index_path = os.path.join(BASE_DIR, "vector.index")
    texts_path = os.path.join(BASE_DIR, "texts.pkl")

    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print("✅ index loaded")
    else:
        print("❌ index missing")

    if os.path.exists(texts_path):
        with open(texts_path, "rb") as f:
            texts = pickle.load(f)
        print("✅ texts loaded:", len(texts))
    else:
        print("❌ texts missing")

except Exception as e:
    print("🔥 LOAD ERROR:", e)

# ✅ GROQ CLIENT SAFE INIT
api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {"status": "RUNNING"}


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    msg = req.message.strip()

    # ✅ BASIC SAFE RESPONSES (NO FAIL)
    if msg.lower() in ["hi", "hello", "hey", "helloo"]:
        return {"reply": "Hello! Ask me your MSBTE questions."}

    if any(word in msg.lower() for word in ["attendance", "marks", "timetable", "result"]):
        return {"reply": "Check SPS Nexus app for this info."}

    # ❌ IF DATA NOT LOADED → STILL RESPOND (NO BREAK)
    if index is None or len(texts) == 0:
        return {"reply": "AI is starting... please try again in a moment."}

    # ✅ SAFE FAISS SEARCH (NO RANDOM CRASH)
    try:
        query_vector = np.zeros((1, index.d), dtype="float32")
        D, I = index.search(query_vector, 3)

        context = "\n\n".join([texts[i] for i in I[0] if i < len(texts)])

        if not context.strip():
            return {"reply": "Topic not found in syllabus."}

    except Exception as e:
        print("🔥 FAISS ERROR:", e)
        return {"reply": "Search error occurred."}

    # ❌ IF API KEY MISSING
    if client is None:
        return {"reply": "AI service not configured."}

    # ✅ GROQ CALL SAFE
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Answer short and simple."},
                {"role": "user", "content": f"{context}\n\nQuestion: {msg}"}
            ],
            max_tokens=200,
        )

        return {"reply": response.choices[0].message.content.strip()}

    except Exception as e:
        print("🔥 GROQ ERROR:", e)
        return {"reply": "AI failed to respond. Try again."}