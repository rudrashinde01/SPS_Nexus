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

# ✅ SAFE LOAD (NO CRASH)
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
        print("✅ texts loaded")
    else:
        print("❌ texts missing")

except Exception as e:
    print("🔥 LOAD ERROR:", e)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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

    if index is None or len(texts) == 0:
        return {"reply": "Server not ready"}

    query_vector = np.random.rand(1, index.d).astype("float32")
    D, I = index.search(query_vector, 3)

    context = "\n\n".join([texts[i] for i in I[0] if i < len(texts)])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer short"},
            {"role": "user", "content": f"{context}\n\n{req.message}"}
        ],
        max_tokens=200,
    )

    return {"reply": response.choices[0].message.content}