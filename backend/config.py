from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env" if (ROOT / ".env").exists() else None)

CHROMA_DIR = os.getenv("CHROMA_DIR", str(ROOT / ".chroma"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DOCS_DIR = ROOT / "data" / "docs"
EVAL_QA = ROOT / "data" / "eval" / "qa_eval.json"
SENSOR_CSV = ROOT / "data" / "sensor_stream.csv"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))
TOP_K = int(os.getenv("TOP_K", 4))