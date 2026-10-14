"""Semantic search tool for DCI hybrid retrieval.

Builds a vector index over a corpus directory using a lightweight
sentence-transformer model, and provides top-k similarity search.

Intended to be called by the DCI agent via bash:
    python semantic_search.py "<query>" --corpus <corpus_dir> --top-k 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_TOP_K = 5
INDEX_CACHE_DIR = ".semantic_index"


def load_corpus(corpus_path: Path) -> list[dict[str, Any]]:
    docs = []
    for f in sorted(corpus_path.rglob("*.txt")):
        if ".semantic_index" in f.parts:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            rel = f.relative_to(corpus_path).as_posix()
            docs.append({"id": rel, "text": text})
    return docs


def build_index(corpus_path: Path, cache_dir: Path, model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_file = cache_dir / "embeddings.npy"
    docs_file = cache_dir / "docs.jsonl"

    if embeddings_file.exists() and docs_file.exists():
        print(f"[semantic_search] using cached index at {cache_dir}", file=sys.stderr)
        return np.load(embeddings_file)

    docs = load_corpus(corpus_path)
    if not docs:
        raise RuntimeError(f"No documents found in {corpus_path}")

    print(f"[semantic_search] indexing {len(docs)} docs ...", file=sys.stderr)
    model = SentenceTransformer(model_name)
    texts = [d["text"][:2000] for d in docs]
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(embeddings_file, embeddings)

    with open(docs_file, "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"[semantic_search] index saved to {cache_dir}", file=sys.stderr)
    return embeddings


def semantic_search(
    query: str,
    corpus_path: Path,
    top_k: int = DEFAULT_TOP_K,
    model_name: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    cache_dir = corpus_path / INDEX_CACHE_DIR
    embeddings = build_index(corpus_path, cache_dir, model_name)

    docs: list[dict[str, Any]] = []
    with open(cache_dir / "docs.jsonl", "r", encoding="utf-8") as fh:
        for line in fh:
            docs.append(json.loads(line))

    model = SentenceTransformer(model_name)
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    query_vec = np.asarray(query_vec, dtype=np.float32)

    scores = embeddings @ query_vec
    top_indices = np.argsort(-scores)[:top_k]

    return [
        {
            "id": docs[i]["id"],
            "score": float(scores[i]),
            "text": docs[i]["text"][:500],
        }
        for i in top_indices
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search over a DCI corpus.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--corpus", required=True, help="Path to corpus directory")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    results = semantic_search(
        query=args.query,
        corpus_path=Path(args.corpus),
        top_k=args.top_k,
        model_name=args.model,
    )
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()