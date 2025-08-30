from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

def _clean_key(k: str | None) -> str | None:
    if not k: return None
    k = k.strip().strip('"').strip("'")
    if "..." in k or len(k) < 20: return None
    return k

# Use OpenAI API with provided key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"

USE_LOCAL_EMBEDDINGS = False  # Use OpenAI embeddings
USE_LOCAL_LLM = False  # Use OpenAI LLM

PKG_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = PKG_DIR
STORAGE_DIR = BASE_DIR / "app" / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
VECTOR_DIR = STORAGE_DIR / "vector"
DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# LlamaParse toggle/key
LLAMA_PARSE_ENABLED = os.getenv("LLAMA_PARSE_ENABLED", "1") == "1"
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
