# Occupational Safety VLM & RAG

This project builds an AI assistant for occupational safety training and incident review.  
It combines **text RAG over safety documents** with a **vision‑language model (VLM)** that analyzes safety videos frame by frame.

---

## Features

- **Document RAG (text)**
  - Parses PDF/HWP/PPT safety documents into clean text.
  - Applies semantic chunking to create meaningful sections.
  - Embeds chunks and enables similarity search for Q&A over accident cases and technical guidelines.

- **Video → VLM analysis**
  - Extracts frames from safety training videos (MP4, etc.).
  - Runs a LLaVA‑style VLM on each frame.
  - Answers safety‑focused questions per frame (what is happening, risks, recommended controls).

- **Multimodal design**
  - Clean separation between **text pipeline** (RAG) and **image pipeline** (VLM).
  - Ready to be extended into multimodal RAG (using VLM captions as queries into the text corpus).

---

## Repository Structure

```├─ scripts/
│ ├─ extract_video_frames.py # OpenCV video → frame extractor
│ ├─ VLM_llava_inference.py # Runs VLM over frames and saves Q&A
│ ├─ batch_parse.py # (Optional) bulk PDF/HWP parsing
│ ├─ vectorize_chunks.py # Builds embeddings from semantic chunks
│ ├─ rag_query.py # RAG retrieval + generation (CLI & library)
│ └─ qa_annotation_app.py # (Optional) tooling for manual QA review
├─ output/
│ ├─ video_frames/ # Extracted frames (git‑ignored)
│ ├─ parsed/ # Parsed text files (git‑ignored)
│ ├─ chunk_texts.txt # All text chunks
│ ├─ chunk_vectors.npy # Embedding matrix
│ └─ vlm_video_results.txt # VLM answers per frame
├─ data/ # Local raw data: videos, PDFs, HWP, PPT (git‑ignored)
├─ .gitignore
└─ README.md
```

> Note: Large raw data files (videos, PDFs, etc.) and generated artifacts are ignored by Git.  
> Only code and small text metadata are versioned.

---

## Setup

### 1. Create and activate environment

python -m venv .venv
source .venv/bin/activate # On macOS / Linux

.venv\Scripts\activate # On Windows PowerShell
text

### 2. Install dependencies

pip install -r requirements.txt

text

Dependencies include:

- `torch`, `transformers`, `accelerate` – for VLM/LLM inference.
- `opencv-python`, `Pillow` – for video frame extraction and image loading.
- `numpy`, `scikit-learn` or vector DB client – for embeddings and retrieval.

(Adjust this list to match your actual `requirements.txt`.)

---

## Usage

### A. Video → frame extraction

Place your input videos under:

data/sample/Video_Input/

text

Then run:

python scripts/extract_video_frames.py

text

This script:

- Reads every `.mp4` / `.avi` / `.mov` in `data/sample/Video_Input/`.
- Writes frames as `.jpg` into:

output/video_frames/

text

with filenames like `videoName_frame0000.jpg`.

### B. VLM inference on frames

Configure the paths at the top of `scripts/VLM_llava_inference.py`:

input_imgs = "/absolute/path/to/output/video_frames"
out_file = "/absolute/path/to/output/vlm_video_results.txt"
MODEL_NAME = "llava-hf/llava-1.5-7b-hf" # or another compatible VLM

text

Run:

python scripts/VLM_llava_inference.py

text

The script:

- Loops over frames in `input_imgs`.
- Asks a small set of safety‑focused questions per frame.
- Saves the Q&A blocks into `vlm_video_results.txt`.

### C. Text RAG pipeline (optional)

If you have preprocessed text chunks (e.g., from safety PDFs):

1. Run the parsing/semantic chunking script (if included):

python scripts/batch_parse.py

text

2. Build embeddings:

python scripts/vectorize_chunks.py

text

This produces:

- `output/chunk_texts.txt` – one chunk per line.
- `output/chunk_vectors.npy` – embedding matrix aligned with chunks.

3. Run a RAG query:

```text
python scripts/rag_query.py --query "What PPE is required for arc welding?"
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--query` / `-q` | *(required)* | The question to answer |
| `--top-k` | `5` | Number of chunks to retrieve |
| `--vectors` | `output/chunk_vectors.npy` | Path to the embedding matrix |
| `--texts` | `output/chunk_texts.txt` | Path to the chunk text file |
| `--llm` | *(none)* | HuggingFace causal-LM model ID for answer generation (e.g. `gpt2`) |

When `--llm` is omitted, the top-k retrieved chunks are printed directly.
When a model ID is provided (e.g. `--llm gpt2`), the script loads that model and
generates a grounded answer from the retrieved context.

The `RAGPipeline` class can also be imported as a library:

```python
from scripts.rag_query import RAGPipeline

pipeline = RAGPipeline(top_k=3)
answer, sources = pipeline.query("What are the risks of working at height?")
for file_id, chunk, score in sources:
    print(f"[{score:.3f}] {file_id}: {chunk[:120]}")
print(answer)
```

---

## Example: End‑to‑end scenario

1. Safety officer uploads:
   - A training video about PPE.
   - A set of accident case PDFs.

2. System runs:
   - **Video pipeline**: extract frames → VLM produces descriptions and risk assessments.
   - **Text pipeline**: parse PDFs → semantic chunking → embeddings.

3. User asks:
   > “What are the main risks shown in this video and which regulations or guidelines apply?”

4. App:
   - Uses VLM output to summarize risks (e.g., “no safety helmet”, “working at height without guardrail”).
   - Uses those descriptions as queries into the RAG index.
   - Returns both:
     - Visual explanation of what’s wrong in the frames.
     - Relevant text snippets from accident case reports and measurement guidelines.

---

## Design Notes

- **Separation of concerns**:  
  Video/VLM logic and document/RAG logic are kept in separate scripts so they can be scaled independently or deployed as separate services.

- **Reproducibility**:  
  All heavy data (videos, PDFs, embeddings) are regenerated from code and excluded via `.gitignore`. Only code and configuration live in the repo, which keeps pushes small and avoids leaking private data.

- **Extensibility**:  
  You can swap in:
  - A different VLM (e.g., LLaVA‑Next or other multimodal models).
  - A vector database (Milvus, FAISS, etc.) instead of `.npy` files.
  - A web UI (Streamlit / FastAPI) to expose the pipelines.

---

## How to Contribute

1. Fork the repository.
2. Create a feature branch:

git checkout -b feature/your-feature-name

text

3. Make your changes and add tests or example notebooks where useful.
4. Submit a pull request with:
- A clear description of the change.
- Before/after behavior (if applicable).
- Any new dependencies or configuration.

---
