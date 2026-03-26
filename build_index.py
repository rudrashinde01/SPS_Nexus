from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
import numpy as np
import re

# ✅ Check dataset folder exists
folder = "dataset"
if not os.path.exists(folder):
    print("❌ Dataset folder not found!")
    exit()

print("📂 Loading dataset...")

texts = []

# ✅ Load all dataset files
for file in os.listdir(folder):
    file_path = os.path.join(folder, file)

    # Skip non-text files
    if not file.endswith(".txt"):
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by Topic (keeps structured chunks)
    chunks = re.split(r'(?=^Topic:)', content, flags=re.MULTILINE)
    texts.extend([c.strip() for c in chunks if c.strip()])

print(f"✅ Total chunks: {len(texts)}")

# ❌ Stop if no data
if len(texts) == 0:
    print("❌ No data found in dataset!")
    exit()

# ✅ Load embedding model
print("🤖 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Create embeddings
print("⚡ Creating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

dimension = embeddings.shape[1]

# ✅ Create FAISS index
print("📦 Building FAISS index...")
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ✅ Save index
faiss.write_index(index, "vector.index")

# ✅ Save text chunks
with open("texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("🎉 Index built successfully!")
print("📁 Files created: vector.index, texts.pkl")