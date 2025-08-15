import sys
import os
import re
import glob
from pathlib import Path
from typing import List, Optional

# ---- Ensure project root is in sys.path ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import chromadb
from chromadb.utils import embedding_functions
from backend.config import CHROMA_DIR, DOCS_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, OPENAI_API_KEY


# ---- Initialize Chroma Client ----
def get_chroma_client():
    from chromadb import PersistentClient
    return PersistentClient(path=str(CHROMA_DIR))


# ---- Embedding function ----
def get_embedder():
    if OPENAI_API_KEY:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        def _embed(texts: List[str]):
            resp = client.embeddings.create(
                model="text-embedding-3-small", input=texts
            )
            return [d.embedding for d in resp.data]

        return _embed, "openai-text-embedding-3-small"
    else:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        ), EMBEDDING_MODEL


# ---- Text chunking ----
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    paras = [p.strip() for p in re.split(r'\n\s*\n+', text) if p.strip()]
    chunks = []
    for p in paras:
        if len(p) <= size:
            chunks.append(p)
        else:
            for i in range(0, len(p), size - overlap):
                chunks.append(p[i:i+size])
    return chunks if chunks else [text[:size]]


# ---- Build index ----
def build_index(clear: bool = False):
    client = get_chroma_client()

    if clear:
        try:
            client.delete_collection("docs")
            print("Cleared existing 'docs' collection.")
        except Exception:
            pass

    embedder, embed_name = get_embedder()
    docs, ids, metadatas = [], [], []

    for fp in glob.glob(str(DOCS_DIR / "*.txt")):
        p = Path(fp)
        raw = p.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(raw)
        equip = "hvac" if "hvac" in p.stem.lower() else "chiller" if "chiller" in p.stem.lower() else "building"

        for i, ch in enumerate(chunks):
            docs.append(ch)
            ids.append(f"{p.stem}-{i}")
            metadatas.append({
                "source": p.name,
                "equipment": equip,
                "chunk": i,
                "embedder": embed_name
            })

    if not docs:
        print("No docs found to index.")
        return

    if callable(embedder):  # OpenAI embeddings path
        coll = client.get_or_create_collection("docs")
        embeddings = embedder(docs)
        coll.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeddings)
    else:  # SentenceTransformers path
        coll = client.get_or_create_collection("docs", embedding_function=embedder)
        coll.add(documents=docs, metadatas=metadatas, ids=ids)

    print(f"Indexed {len(docs)} chunks into Chroma.")


# ---- Retrieve docs ----
def retrieve(query: str, equipment: Optional[str] = None, k: int = TOP_K):
    client = get_chroma_client()
    coll = client.get_or_create_collection("docs")
    where = {"equipment": equipment} if equipment else None

    res = coll.query(query_texts=[query], n_results=k, where=where)
    out = []
    for i in range(len(res["ids"][0])):
        out.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "dist": res.get("distances", [[None]])[0][i] if res.get("distances") else None
        })
    return out


if __name__ == "__main__":
    build_index(clear=True)