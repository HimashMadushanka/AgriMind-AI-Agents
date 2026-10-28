"""Semantic search over expert-reviewed agricultural documents.

Only files placed in knowledge_base/documents are indexed. Synthetic QA rows are
not treated as authoritative sources. Document embeddings are cached in memory
so subsequent questions do not re-embed the entire corpus.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS_DIR = ROOT / "knowledge_base" / "documents"
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def _directory_signature(folder: Path) -> int:
    """Return a simple cache invalidation signature for all supported files."""
    total = 0
    if not folder.exists():
        return total
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                stat = path.stat()
                total ^= hash((str(path), stat.st_mtime_ns, stat.st_size))
            except OSError:
                continue
    return total


def _read_documents(docs_dir: Path) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    if not docs_dir.exists():
        return docs
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in {".txt", ".md"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
            elif suffix == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            else:
                continue
        except Exception as exc:
            print(f"Could not read {path.name}: {exc}")
            continue
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        # Character-based chunks are a simple starting point. For production,
        # consider paragraph-aware chunking and preserve PDF page numbers.
        size, overlap = 1000, 150
        start = 0
        while start < len(text):
            chunk = text[start : start + size]
            if len(chunk.strip()) > 80:
                docs.append({"source": path.name, "text": chunk})
            if start + size >= len(text):
                break
            start += size - overlap
    return docs


@lru_cache(maxsize=3)
def _load_index(folder_str: str, model_name: str, signature: int):
    """Load documents, embedding model, and normalized document vectors once."""
    from sentence_transformers import SentenceTransformer

    folder = Path(folder_str)
    docs = _read_documents(folder)
    model = SentenceTransformer(model_name)
    if docs:
        passages = ["passage: " + item["text"] for item in docs]
        vectors = model.encode(
            passages,
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
    else:
        vectors = np.empty((0, 0), dtype=np.float32)
    return docs, model, vectors


def search_agricultural_knowledge(
    query: str,
    k: int = 5,
    docs_dir: str | None = None,
    model_name: str = "intfloat/multilingual-e5-base",
) -> list[dict]:
    """Return top evidence chunks ranked by multilingual cosine similarity.

    A score is a retrieval-similarity score, not evidence that the document is
    correct. Before real deployment, store source metadata and require expert
    review of the documents and advice.
    """
    if not query or not query.strip():
        return []
    folder = Path(docs_dir).resolve() if docs_dir else DEFAULT_DOCS_DIR
    if not folder.exists():
        return []
    signature = _directory_signature(folder)
    try:
        docs, model, vectors = _load_index(str(folder), model_name, signature)
        if not docs:
            return []
        query_vector = model.encode(
            ["query: " + query.strip()], normalize_embeddings=True, show_progress_bar=False
        )[0]
        # Both vectors are normalized, so dot product equals cosine similarity.
        scores = vectors @ np.asarray(query_vector, dtype=np.float32)
        order = np.argsort(scores)[::-1][: max(1, int(k))]
        return [{**docs[int(i)], "score": float(scores[int(i)])} for i in order]
    except ImportError as exc:
        raise RuntimeError(
            "Install sentence-transformers and scikit-learn to enable semantic search."
        ) from exc
