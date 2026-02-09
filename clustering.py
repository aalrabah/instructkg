# clustering.py
"""
Global chunk clustering using sentence-transformers + UMAP + HDBSCAN.

Reads:
  - out/chunks.jsonl   (chunk_id -> chunk text)
  - out/mentions.jsonl (from llm.py: concept_id + chunk_id, etc.)

Writes:
  - out/context_clusters.jsonl (one record per cluster) like:

    {
      "cluster_id": 7,
      "label_hint": "joins-and-null-semantics",
      "count_chunks": 42,
      "chunk_ids": ["lec3__0012", "lec3__0044", ...],
      "concepts": ["LEFT_OUTER_JOIN", "NULL_VALUE", "INNER_JOIN", ...],
      "chunks": [
        {"chunk_id": "lec3__0012", "text": "..."},
        {"chunk_id": "lec3__0044", "text": "..."}
      ]
    }

Notes:
- chunks field contains top 2 representative chunks (closest to centroid) for preview
- concepts field contains ALL concepts that appear in this cluster's chunks
- chunk_ids contains ALL chunk IDs in the cluster

Install deps:
  pip install sentence-transformers scikit-learn
  pip install umap-learn hdbscan   # recommended for clustering

Run:
  python clustering.py \
    --chunks out/chunks.jsonl \
    --mentions out/mentions.jsonl \
    --out out/context_clusters.jsonl \
    --use-umap
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


logger = logging.getLogger("clustering")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


# ----------------------------
# JSONL I/O
# ----------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    # supports JSON list or JSONL
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} is JSON but not a list.")
            return data
    return read_jsonl(path)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# Text helpers
# ----------------------------

def slugify_tokens(tokens: List[str], max_tokens: int = 4) -> str:
    """
    Make a compact label like "joins-and-null-semantics" from term tokens.
    """
    toks: List[str] = []
    for t in tokens:
        t = (t or "").lower()
        t = re.sub(r"[^a-z0-9]+", "", t)
        if not t:
            continue
        if t in toks:
            continue
        toks.append(t)
        if len(toks) >= max_tokens:
            break

    if not toks:
        return "misc"
    if len(toks) == 1:
        return toks[0]
    if len(toks) == 2:
        return f"{toks[0]}-and-{toks[1]}"
    return f"{toks[0]}-and-{toks[1]}-" + "-".join(toks[2:])


def terms_to_label_hint(terms: List[str], max_tokens: int = 4) -> str:
    """
    Convert c-TF-IDF terms into a stable slug.
    """
    flat_tokens: List[str] = []
    for term in terms or []:
        for w in (term or "").split():
            if not w:
                continue
            flat_tokens.append(w)
            if len(flat_tokens) >= 12:
                break
        if len(flat_tokens) >= 12:
            break
    return slugify_tokens(flat_tokens, max_tokens=max_tokens)


# ----------------------------
# ML bits (embedding + UMAP + HDBSCAN + c-TF-IDF)
# ----------------------------

def _import_or_die():
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.preprocessing import normalize
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies. Install:\n"
            "  pip install sentence-transformers scikit-learn\n"
            "Optional but recommended for clustering:\n"
            "  pip install umap-learn hdbscan\n"
        ) from e

    try:
        import umap  # type: ignore
    except Exception:
        umap = None

    try:
        import hdbscan  # type: ignore
    except Exception:
        hdbscan = None

    return SentenceTransformer, np, CountVectorizer, TfidfTransformer, normalize, umap, hdbscan


def embed_texts(texts, model_name: str, batch_size: int, normalize_embeddings: bool):
    SentenceTransformer, np, *_ = _import_or_die()

    if isinstance(texts, str):
        texts = [texts]
    elif not isinstance(texts, list):
        texts = list(texts)

    model = SentenceTransformer(model_name)
    X = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    X = np.asarray(X, dtype="float32")
    if X.ndim == 1:
        X = X.reshape(1, -1)

    return X


def cluster_hdbscan(Xr, *, min_cluster_size: int, min_samples: Optional[int]):
    SentenceTransformer, np, CountVectorizer, TfidfTransformer, normalize, umap, hdbscan = _import_or_die()
    if hdbscan is None:
        raise RuntimeError("hdbscan not installed. Install: pip install hdbscan")
    if min_samples is None:
        min_samples = max(1, min_cluster_size // 2)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    return clusterer.fit_predict(Xr)


def ctfidf_terms_per_cluster(cluster_docs: List[str], top_terms: int) -> List[List[str]]:
    SentenceTransformer, np, CountVectorizer, TfidfTransformer, normalize, umap, hdbscan = _import_or_die()
    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    counts = vec.fit_transform(cluster_docs)
    tfidf = TfidfTransformer(norm=None).fit_transform(counts)
    terms = vec.get_feature_names_out()

    out: List[List[str]] = []
    for i in range(tfidf.shape[0]):
        row = tfidf[i].toarray().ravel()
        if row.size == 0:
            out.append([])
            continue
        idx = row.argsort()[-top_terms:][::-1]
        out.append([str(terms[j]) for j in idx if row[j] > 0])
    return out


# ----------------------------
# Global clustering
# ----------------------------

def cluster_global_chunks(
    *,
    chunks: List[Dict[str, Any]],
    mentions: List[Dict[str, Any]],
    embedding_model: str,
    batch_size: int,
    normalize_embeddings: bool,
    use_umap: bool,
    umap_components: int,
    umap_neighbors: int,
    min_cluster_size: Optional[int],
    min_samples: Optional[int],
    top_terms: int,
    top_k_chunks_per_cluster: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Global clustering across ALL chunks.

    Output records (one per global cluster):
    {
      "cluster_id": 7,
      "label_hint": "...",
      "count_chunks": 42,
      "chunk_ids": ["...", "..."],
      "concepts": ["Concept A", "Concept B", ...],  # ALL concepts in this cluster
      "chunks": [{"chunk_id": "...", "text": "..."}]  # top 2 representative chunks
    }
    """

    # ---- helpers ----
    def _get_text(ch: Dict[str, Any]) -> str:
        for k in ("text", "chunk_text", "content", "raw_text", "page_text", "body", "markdown"):
            v = ch.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for k in ("lines", "sentences", "paragraphs"):
            v = ch.get(k)
            if isinstance(v, list):
                s = "\n".join(str(x) for x in v if str(x).strip()).strip()
                if s:
                    return s
        return ""

    def _build_chunk_to_concepts(mentions_: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        """
        Build chunk_id -> set of all concepts in that chunk.
        """
        out: Dict[str, Set[str]] = defaultdict(set)
        for m in mentions_:
            cid = m.get("chunk_id")
            if not cid:
                continue
            concept = (m.get("concept_id") or "").strip()
            if not concept:
                continue
            out[str(cid)].add(concept)
        return out

    def _get_all_concepts_for_cluster(
        chunk_ids: List[str],
        chunk_to_concepts: Dict[str, Set[str]],
    ) -> List[str]:
        """
        Get ALL unique concepts across all chunks in a cluster.
        Returns sorted list for stable output.
        """
        all_concepts: Set[str] = set()
        for cid in chunk_ids:
            all_concepts.update(chunk_to_concepts.get(cid, set()))
        return sorted(all_concepts)

    # ---- collect all valid chunk texts ----
    chunk_ids: List[str] = []
    texts: List[str] = []
    for ch in chunks:
        cid = ch.get("chunk_id")
        if cid is None:
            continue
        cid = str(cid).strip()
        if not cid:
            continue
        t = _get_text(ch)
        if not t:
            continue
        chunk_ids.append(cid)
        texts.append(t)

    if not texts:
        return []

    # ---- embed ----
    X = embed_texts(
        texts,
        model_name=embedding_model,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )

    enable_clustering = bool(use_umap)

    # Build chunk -> concepts mapping
    chunk_to_concepts = _build_chunk_to_concepts(mentions)

    # If clustering disabled, return one big cluster (id=0)
    if not enable_clustering:
        try:
            terms = ctfidf_terms_per_cluster([" ".join(texts)], top_terms=top_terms)[0]
            hint = terms_to_label_hint(terms) if terms else "misc"
        except Exception:
            hint = "misc"

        all_concepts = _get_all_concepts_for_cluster(chunk_ids, chunk_to_concepts)
        top_chunks = [{"chunk_id": chunk_ids[i], "text": texts[i]} for i in range(min(top_k_chunks_per_cluster, len(texts)))]

        return [{
            "cluster_id": 0,
            "label_hint": hint,
            "count_chunks": len(chunk_ids),
            "chunk_ids": chunk_ids,
            "concepts": all_concepts,
            "chunks": top_chunks,
        }]

    # ---- reduce with UMAP + cluster ----
    SentenceTransformer, np, CountVectorizer, TfidfTransformer, normalize, umap_lib, hdbscan = _import_or_die()
    if umap_lib is None:
        raise RuntimeError("umap-learn not installed. Install: pip install umap-learn")
    
    n = len(texts)
    n_components = max(2, min(int(umap_components), n - 1, X.shape[1]))  # UMAP needs n_components < n_samples
    n_neighbors = min(int(umap_neighbors), n - 1)  # must be < n_samples
    
    reducer = umap_lib.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=42,
    )
    Xr = reducer.fit_transform(X)

    if min_cluster_size is not None:
        mcs = int(min_cluster_size)
    else:
        mcs = max(5, min(25, n // 50 if n >= 500 else max(5, n // 20)))

    labels = cluster_hdbscan(Xr, min_cluster_size=mcs, min_samples=min_samples)

    by_label: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_label[int(lab)].append(i)

    non_noise = sorted([lab for lab in by_label.keys() if lab != -1])
    cluster_docs = [" ".join(texts[i] for i in by_label[lab]) for lab in non_noise]
    term_lists = ctfidf_terms_per_cluster(cluster_docs, top_terms=top_terms) if non_noise else []
    terms_by_label = {lab: terms for lab, terms in zip(non_noise, term_lists)}

    # For centroid-based representative chunk selection
    SentenceTransformer, np, *_ = _import_or_die()
    Xnp = np.asarray(X, dtype="float32")

    out: List[Dict[str, Any]] = []
    for lab, idxs in by_label.items():
        if not idxs:
            continue

        # label hint
        if lab == -1:
            label_hint = "noise"
        else:
            label_hint = terms_to_label_hint(terms_by_label.get(lab, []))

        # representative chunks: nearest to centroid
        Xc = Xnp[idxs]
        centroid = Xc.mean(axis=0, keepdims=True)
        sims = (Xc @ centroid.T).reshape(-1)
        order = sims.argsort()[::-1]
        chosen = [idxs[int(j)] for j in order[:top_k_chunks_per_cluster]]

        cluster_chunk_ids = [chunk_ids[i] for i in idxs]
        all_concepts = _get_all_concepts_for_cluster(cluster_chunk_ids, chunk_to_concepts)

        out.append({
            "cluster_id": int(lab),
            "label_hint": label_hint,
            "count_chunks": len(idxs),
            "chunk_ids": cluster_chunk_ids,
            "concepts": all_concepts,
            "chunks": [{"chunk_id": chunk_ids[i], "text": texts[i]} for i in chosen],
        })

    # sort: bigger first, noise last
    out.sort(
        key=lambda r: (
            1 if r["cluster_id"] == -1 else 0,
            -r["count_chunks"],
            r["cluster_id"],
        )
    )
    return out


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Global chunk clustering")
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--mentions", required=True, help="Path to mentions.jsonl")
    parser.add_argument("--out", required=True, help="Output path for context_clusters.jsonl")
    parser.add_argument("--use-umap", action="store_true", help="Enable clustering (otherwise single cluster)")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--umap-components", type=int, default=15, help="UMAP components for dimensionality reduction")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors parameter")
    parser.add_argument("--min-cluster-size", type=int, default=None, help="HDBSCAN min cluster size")
    parser.add_argument("--min-samples", type=int, default=None, help="HDBSCAN min samples")
    parser.add_argument("--top-terms", type=int, default=5, help="Top terms for label hint")
    parser.add_argument("--top-k-chunks", type=int, default=None)

    args = parser.parse_args()

    logger.info(f"Loading chunks from {args.chunks}")
    chunks = read_json_or_jsonl(args.chunks)
    logger.info(f"Loaded {len(chunks)} chunks")

    logger.info(f"Loading mentions from {args.mentions}")
    mentions = read_json_or_jsonl(args.mentions)
    logger.info(f"Loaded {len(mentions)} mentions")

    logger.info("Running global clustering...")
    clusters = cluster_global_chunks(
        chunks=chunks,
        mentions=mentions,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        use_umap=args.use_umap,
        umap_components=args.umap_components,
        umap_neighbors=args.umap_neighbors,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        top_terms=args.top_terms,
        top_k_chunks_per_cluster=args.top_k_chunks,
    )

    logger.info(f"Writing {len(clusters)} clusters to {args.out}")
    write_jsonl(args.out, clusters)
    logger.info("Done!")


if __name__ == "__main__":
    main()