"""
rag_query.py – Retrieval-Augmented Generation query interface.

Pipeline:
  1. Load pre-built chunk embeddings (output/chunk_vectors.npy) and
     corresponding text (output/chunk_texts.txt) produced by vectorize_chunks.py.
  2. Embed the user question with the same sentence-transformer model.
  3. Rank chunks by cosine similarity and return the top-k.
  4. Optionally pass the retrieved context to a local text-generation LLM
     (e.g. a HuggingFace causal-LM) to produce a grounded answer.

Usage (CLI):
    python scripts/rag_query.py --query "What PPE is required for arc welding?"

Usage (as a library):
    from scripts.rag_query import RAGPipeline
    pipeline = RAGPipeline()
    answer, sources = pipeline.query("What are the main risks of working at height?")
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Default paths (relative to the project root; override via CLI or constructor)
# ---------------------------------------------------------------------------
_DEFAULT_VECTORS = os.path.join(os.path.dirname(__file__), "..", "output", "chunk_vectors.npy")
_DEFAULT_TEXTS   = os.path.join(os.path.dirname(__file__), "..", "output", "chunk_texts.txt")
_EMBED_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for occupational-safety documents."""

    def __init__(
        self,
        vectors_path: str = _DEFAULT_VECTORS,
        texts_path: str = _DEFAULT_TEXTS,
        embed_model: str = _EMBED_MODEL,
        llm_model_name: Optional[str] = None,
        top_k: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        vectors_path:   Path to the .npy embedding matrix (one row per chunk).
        texts_path:     Path to the tab-separated chunk texts file
                        (format: ``<file_id>\\t<chunk_text>`` per line).
        embed_model:    SentenceTransformer model name used for both indexing and
                        query embedding.  Must match the model used in
                        vectorize_chunks.py.
        llm_model_name: (Optional) HuggingFace causal-LM model ID to use for
                        answer generation.  When *None* the pipeline returns only
                        the retrieved chunks without generating a final answer.
        top_k:          Number of top chunks to retrieve.
        """
        self.top_k = top_k
        self.llm_model_name = llm_model_name

        # --- load embeddings & texts ------------------------------------------
        if not os.path.exists(vectors_path):
            raise FileNotFoundError(
                f"Chunk vectors not found at '{vectors_path}'. "
                "Run scripts/vectorize_chunks.py first."
            )
        if not os.path.exists(texts_path):
            raise FileNotFoundError(
                f"Chunk texts not found at '{texts_path}'. "
                "Run scripts/vectorize_chunks.py first."
            )

        self.vectors: np.ndarray = np.load(vectors_path)
        self.chunks: list[str] = []
        self.file_ids: list[str] = []

        with open(texts_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    self.file_ids.append(parts[0])
                    self.chunks.append(parts[1])
                else:
                    self.file_ids.append("")
                    self.chunks.append(parts[0])

        if len(self.chunks) != self.vectors.shape[0]:
            raise ValueError(
                f"Mismatch: {len(self.chunks)} text chunks but "
                f"{self.vectors.shape[0]} embedding rows."
            )

        # L2-normalise stored vectors once so dot-product == cosine similarity.
        # Replace zero norms with a small value to avoid division-by-zero.
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self.vectors_norm: np.ndarray = self.vectors / norms

        # --- embedding model --------------------------------------------------
        print(f"Loading embedding model '{embed_model}' …")
        self.embed_model = SentenceTransformer(embed_model)

        # --- optional LLM -----------------------------------------------------
        self._llm = None
        self._tokenizer = None
        if llm_model_name:
            self._load_llm(llm_model_name)

    # ------------------------------------------------------------------
    def _load_llm(self, model_name: str) -> None:
        """Lazily load a HuggingFace causal-LM for answer generation."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading LLM '{model_name}' on {device} …")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            self._device = device
        except ImportError:
            print(
                "Warning: 'transformers' or 'torch' not installed; "
                "answer generation disabled.  Install them to enable the LLM step."
            )

    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> list[tuple[str, str, float]]:
        """Return the top-k most relevant (file_id, chunk, score) tuples."""
        q_vec = self.embed_model.encode([query])
        q_norm = q_vec / np.maximum(np.linalg.norm(q_vec, keepdims=True), 1e-10)
        scores: np.ndarray = (self.vectors_norm @ q_norm.T).squeeze()
        top_indices = np.argsort(scores)[::-1][: self.top_k]
        return [
            (self.file_ids[i], self.chunks[i], float(scores[i]))
            for i in top_indices
        ]

    # ------------------------------------------------------------------
    def generate(self, query: str, context_chunks: list[str]) -> str:
        """Generate an answer using the optional LLM and the retrieved context."""
        if self._llm is None or self._tokenizer is None:
            # No LLM available – return the retrieved context as the "answer"
            return "\n\n---\n\n".join(context_chunks)

        import torch

        context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
        prompt = (
            "You are an occupational-safety expert assistant.\n"
            "Use ONLY the context below to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output_ids = self._llm.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        answer = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Strip the prompt prefix from the decoded output
        if "Answer:" in answer:
            answer = answer.split("Answer:", 1)[-1].strip()
        return answer

    # ------------------------------------------------------------------
    def query(self, question: str) -> tuple[str, list[tuple[str, str, float]]]:
        """
        Run the full RAG pipeline for a given question.

        Returns
        -------
        answer:  Generated answer string (or concatenated chunks if no LLM).
        sources: List of (file_id, chunk_text, similarity_score) tuples.
        """
        sources = self.retrieve(question)
        context_chunks = [chunk for _, chunk, _ in sources]
        answer = self.generate(question, context_chunks)
        return answer, sources


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the occupational-safety RAG pipeline."
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="The question to answer.",
    )
    parser.add_argument(
        "--vectors",
        default=_DEFAULT_VECTORS,
        help="Path to chunk_vectors.npy  (default: output/chunk_vectors.npy).",
    )
    parser.add_argument(
        "--texts",
        default=_DEFAULT_TEXTS,
        help="Path to chunk_texts.txt  (default: output/chunk_texts.txt).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve  (default: 5).",
    )
    parser.add_argument(
        "--llm",
        default=None,
        help=(
            "HuggingFace causal-LM model ID to generate a final answer "
            "(e.g. 'gpt2').  When omitted, retrieved chunks are printed directly."
        ),
    )
    args = parser.parse_args()

    pipeline = RAGPipeline(
        vectors_path=args.vectors,
        texts_path=args.texts,
        top_k=args.top_k,
        llm_model_name=args.llm,
    )

    answer, sources = pipeline.query(args.query)

    print("\n" + "=" * 60)
    print("QUERY:", args.query)
    print("=" * 60)
    print("\n--- Retrieved Sources ---")
    for rank, (fid, chunk, score) in enumerate(sources, start=1):
        print(f"\n[{rank}] score={score:.4f}  source={fid}")
        snippet = chunk[:300] + ("…" if len(chunk) > 300 else "")
        print(snippet)

    print("\n--- Answer ---")
    print(answer)


if __name__ == "__main__":
    main()
