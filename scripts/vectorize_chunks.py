from sentence_transformers import SentenceTransformer
import glob
import os
import numpy as np

print("cwd:", os.getcwd())


# Set the directory for semantic chunk files and targets for output
CHUNKS_DIR = r"C:\Users\srima\Desktop\A3_SETS\occupational_safety_\output\parsed"
OUT_VEC = os.path.join("..", "output", "chunk_vectors.npy")
OUT_TEXT = os.path.join("..", "output", "chunk_texts.txt")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

chunk_files = glob.glob(os.path.join(CHUNKS_DIR, "*.semantic_chunks.txt"))
print("Searching in:", CHUNKS_DIR)
print(f"Found {len(chunk_files)} files:")
for fname in chunk_files:
    print(" -", fname)

all_chunks = []
file_ids = []

for fname in chunk_files:
    with open(fname, encoding="utf-8") as f:
        content = f.read()
        count = 0
        for block in content.split("[Semantic Chunk ")[1:]:
            lines = block.split("]", 1)
            if len(lines) > 1:
                chunk = lines[1].strip()
                if chunk:
                    all_chunks.append(chunk)
                    file_ids.append(os.path.basename(fname))
                    count += 1
        print(f"Extracted {count} chunks from {os.path.basename(fname)}")

print(f"Total chunks found: {len(all_chunks)}")

if all_chunks:
    embeddings = model.encode(all_chunks, show_progress_bar=True)

    np.save(OUT_VEC, embeddings)
    with open(OUT_TEXT, "w", encoding="utf-8") as f:
        for fname, chunk in zip(file_ids, all_chunks):
            f.write(f"{fname}\t{chunk.replace(chr(10), ' ')}\n")

    print(f"Saved vectors to: {OUT_VEC}")
    print(f"Saved texts to: {OUT_TEXT}")
else:
    print("No chunks found to vectorize.")
