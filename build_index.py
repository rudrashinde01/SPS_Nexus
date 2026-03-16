from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
import numpy as np
import re

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

folder = "dataset"
texts = []

# Load all dataset files
for file in os.listdir(folder):
    with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
        content = f.read()
    # Split by Topic: so each entry stays complete
    chunks = re.split(r'(?=^Topic:)', content, flags=re.MULTILINE)
    texts.extend([c.strip() for c in chunks if c.strip()])

print("Total chunks:", len(texts))

# Convert text to embeddings
embeddings = model.encode(texts, show_progress_bar=True)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index
faiss.write_index(index, "vector.index")

# Save text chunks
with open("texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("Index built successfully!")