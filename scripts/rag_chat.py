from pathlib import Path
import os
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Auto-load .env
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
VEC_PATH = ROOT / "output" / "chunk_vectors.npy"
TXT_PATH = ROOT / "output" / "chunk_texts.txt"
INDEX_PATH = ROOT / "output" / "faiss.index"

EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
HF_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-Coder-3B-Instruct")

# Globals for performance
embed_model = None
faiss_index = None
meta = None


def load():
    global meta
    print("Loading:", VEC_PATH)
    vectors = np.load(str(VEC_PATH)).astype("float32")

    meta = []
    with open(TXT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            src, txt = (line.split("\t", 1) + [""])[:2]
            meta.append((src or "unknown", txt))

    print(f"Vectors: {vectors.shape}, Texts: {len(meta)}")
    return vectors


def load_or_build_index(vectors):
    global faiss_index
    if INDEX_PATH.exists():
        print("Loading FAISS:", INDEX_PATH)
        faiss_index = faiss.read_index(str(INDEX_PATH))
    else:
        print("Building FAISS index...")
        faiss.normalize_L2(vectors)
        faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        faiss_index.add(vectors)
        faiss.write_index(faiss_index, str(INDEX_PATH))
        print("FAISS index saved:", INDEX_PATH)
    return faiss_index


def get_embed_model():
    global embed_model
    if embed_model is None:
        print("Loading embedding model:", EMBED_MODEL_NAME)
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return embed_model


def retrieve(query, k=8):
    embed = get_embed_model()
    qv = embed.encode([query]).astype("float32")
    faiss.normalize_L2(qv)
    scores, ids = faiss_index.search(qv, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if int(idx) >= 0 and int(idx) < len(meta):
            src, txt = meta[int(idx)]
            results.append({
                "rank": len(results) + 1,
                "score": float(score),
                "source": src,
                "text": txt
            })
    return results


def build_prompt(question, retrieved):
    """Korean-optimized prompt for occupational safety documents"""
    ctx_blocks = []
    for r in retrieved[:6]:  # Top 6 for better context
        ctx_blocks.append(f"[{r['rank']}] {r['source']}\n{r['text']}")

    context = "\n\n---\n\n".join(ctx_blocks)

    return f"""컨텍스트에서 {question}에 대한 답변을 작성하세요.

**필수 출력 형식 (정확히 따르세요):**
- 첫 번째 핵심 사항
- 두 번째 핵심 사항
- 세 번째 핵심 사항
출처: [1], [2], [3]

컨텍스트에 정보가 부족하면: '제공된 문서에서 확인할 수 없습니다.'

**질문:** {question}

**컨텍스트:**
{context}

**답변:**"""


def extract_bullets(text):
    """Extract only bullet points + sources"""
    if not text or "확인할 수 없습니다" in text:
        return "제공된 문서에서 확인할 수 없습니다."

    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # Find bullet start
    bullet_start = None
    for i, line in enumerate(lines):
        if line.startswith('•') or line.startswith('-') or '•' in line[:2] or '-' in line[:2]:
            bullet_start = i
            break

    if bullet_start is None:
        return "제공된 문서에서 확인할 수 없습니다."

    # Find end (sources or 5 lines max)
    end = len(lines)
    for i in range(bullet_start + 1, len(lines)):
        if '출처' in lines[i] or i >= bullet_start + 5:
            end = i + 1
            break

    result = '\n'.join(lines[bullet_start:end])
    return result if result.strip() else "제공된 문서에서 확인할 수 없습니다."


def hf_generate(prompt):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not found in environment")

    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": HF_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "한국어로 정확한 bullet point 형식으로 답변하세요. 출처를 반드시 명시하세요."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 400
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    return extract_bullets(content)


def main():
    global meta
    vectors = load()
    load_or_build_index(vectors)

    print(f"\n✅ RAG 시스템 준비 완료!")
    print(f"📚 문서: {len(meta):,}개 | 모델: {HF_MODEL}")
    print("\n질문 입력 ('exit'로 종료):\n")

    while True:
        query = input("You> ").strip()
        if query.lower() in ('exit', 'quit', '종료'):
            print("RAG 시스템 종료.")
            break
        if not query:
            continue

        # Retrieve
        retrieved = retrieve(query)
        print("\n🔍 검색된 상위 문서:")
        for r in retrieved:
            preview = r["text"][:150].replace('\n', ' ')
            print(f"[{r['rank']}] {r['score']:.3f} {r['source'][:50]}...: {preview}...")

        # Generate
        prompt = build_prompt(query, retrieved)
        print("\n🤖 답변 생성 중...")
        try:
            answer = hf_generate(prompt)
            print(f"\n📋 **답변:**\n{answer}")
        except Exception as e:
            print(f"❌ 생성 오류: {e}")
            print("HF_TOKEN과 HF_MODEL 확인하세요.")

        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
