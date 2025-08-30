from __future__ import annotations

import os, shutil
from pathlib import Path
from typing import List, Tuple

from .settings import (
    OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL,
    USE_LOCAL_EMBEDDINGS, USE_LOCAL_LLM,
    BASE_DIR, UPLOAD_DIR, VECTOR_DIR, LLAMA_PARSE_ENABLED,  # <- LlamaParse flag
)
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

_EMBED_CONFIGURED = False
_LLM_CONFIGURED = False


def _configure_embeddings() -> None:
    """Set up the global embedding model (OpenAI by default, or local HF if toggled)."""
    global _EMBED_CONFIGURED
    if _EMBED_CONFIGURED:
        return

    if USE_LOCAL_EMBEDDINGS:
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "USE_LOCAL_EMBEDDINGS=1 but 'llama-index-embeddings-huggingface' is not installed.\n"
                "Install: pip install -U llama-index-embeddings-huggingface sentence-transformers"
            ) from e
        model_name = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)
    else:
        if not OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API key missing/invalid. Set OPENAI_API_KEY in .env or OS env.\n"
                "Or set USE_LOCAL_EMBEDDINGS=1 to use local embeddings."
            )
        from llama_index.embeddings.openai import OpenAIEmbedding
        Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

    _EMBED_CONFIGURED = True


def _configure_llm() -> None:
    """Set up the global LLM (OpenAI by default, or local mock if toggled)."""
    global _LLM_CONFIGURED
    if _LLM_CONFIGURED:
        return

    if USE_LOCAL_LLM:
        try:
            from llama_index.llms.mock import MockLLM
            Settings.llm = MockLLM(max_tokens=1024)
        except Exception:
            class MockLLM:  # very simple fallback
                def __init__(self, **kwargs): ...
                def complete(self, prompt: str) -> str:
                    return "MockLLM response:\n" + prompt[:1000]
            Settings.llm = MockLLM()
    else:
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key required for LLM unless USE_LOCAL_LLM=1.")
        from llama_index.llms.openai import OpenAI as OpenAILLM
        Settings.llm = OpenAILLM(model=LLM_MODEL, api_key=OPENAI_API_KEY)

    _LLM_CONFIGURED = True


def _persisted_index_exists() -> bool:
    if not VECTOR_DIR.exists():
        return False
    for name in ("docstore.json", "index_store.json", "vector_store.json", "graph_store.json"):
        if (VECTOR_DIR / name).exists():
            return True
    return False


def _reader() -> SimpleDirectoryReader:
    # Only index Markdown & TXT (we create .md from LlamaParse)
    return SimpleDirectoryReader(
        input_dir=str(UPLOAD_DIR),
        recursive=True,
        filename_as_id=True,
        required_exts=[".md", ".txt"],
    )


def get_index() -> VectorStoreIndex:
    _configure_embeddings()
    _configure_llm()

    if _persisted_index_exists():
        storage_context = StorageContext.from_defaults(persist_dir=str(VECTOR_DIR))
        return load_index_from_storage(storage_context)

    docs = _reader().load_data()
    idx = VectorStoreIndex.from_documents(docs)
    idx.storage_context.persist(persist_dir=str(VECTOR_DIR))
    return idx


def add_files(file_paths: List[str]) -> Tuple[int, List[str]]:
    """Copy files into uploads and (for PDFs) run LlamaParse to produce .md, then (re)index."""
    copied: List[str] = []
    from .llama_parser import parse_with_llamaparse  # <- LlamaParse, not Azure

    for src in file_paths:
        src_path = Path(src)
        if not src_path.exists():
            # Maybe the file is already in UPLOAD_DIR
            candidate = UPLOAD_DIR / src_path.name
            if candidate.exists():
                copied.append(str(candidate))
                continue
            continue

        dst = UPLOAD_DIR / src_path.name
        if src_path.resolve() != dst.resolve():
            try:
                shutil.copy2(str(src_path), str(dst))
            except Exception:
                shutil.copyfile(str(src_path), str(dst))
        copied.append(str(dst))

        # LlamaParse: convert PDFs to Markdown for clean indexing (skip if already .md/.txt)
        if LLAMA_PARSE_ENABLED and dst.suffix.lower() in {".pdf"}:
            try:
                parse_with_llamaparse(str(dst), str(dst.parent))
            except Exception as e:
                print(f"[LlamaParse] Parse failed for {dst.name}: {e}")

    # Build or update index
    get_index()
    return len(copied), copied


def query(q: str):
    """Run a semantic query; return (answer, [citations])."""
    try:
        idx = get_index()
        engine = idx.as_query_engine(similarity_top_k=int(os.getenv("TOP_K", "4")))
        res = engine.query(q)
        answer = str(getattr(res, "response", None) or str(res))
        cites = []
        for node in getattr(res, "source_nodes", []) or []:
            meta = getattr(getattr(node, "node", node), "metadata", {}) or {}
            name = meta.get("file_name") or meta.get("filename") or meta.get("source") or "document"
            if name not in cites:
                cites.append(name)
        return answer, cites
    except Exception as e:
        return f"Query failed: {type(e).__name__}: {e}", []


def clear_index(delete_uploads: bool = False) -> None:
    """Delete persisted vectors (and optionally uploads)."""
    if VECTOR_DIR.exists():
        for p in VECTOR_DIR.glob("*"):
            try:
                p.unlink()
            except IsADirectoryError:
                shutil.rmtree(p, ignore_errors=True)
    if delete_uploads and UPLOAD_DIR.exists():
        for p in UPLOAD_DIR.glob("*"):
            try:
                p.unlink()
            except IsADirectoryError:
                shutil.rmtree(p, ignore_errors=True)
