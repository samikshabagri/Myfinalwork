from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import os

def parse_with_llamaparse(input_path: str, out_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Use LlamaParse (LlamaIndex Cloud) to parse PDFs into Markdown.
    Returns (markdown_path, tables_json_path). tables_json_path is None in this minimal version.
    """
    try:
        from llama_parse import LlamaParse
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "LlamaParse not installed. Install with: pip install llama-parse"
        ) from e

    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key or len(api_key.strip()) < 10:
        raise ValueError(
            "LLAMA_CLOUD_API_KEY is missing/invalid. Set it in your .env or OS environment."
        )

    parser = LlamaParse(api_key=api_key, result_type="markdown")
    documents = parser.load_data(input_path)

    md_texts = []
    for d in documents:
        text = getattr(d, "text", None) or str(d)
        md_texts.append(text)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    md_path = out_dir_p / (Path(input_path).stem + ".md")
    md_path.write_text("\n\n---\n\n".join(md_texts).strip() + "\n", encoding="utf-8")

    return str(md_path), None
