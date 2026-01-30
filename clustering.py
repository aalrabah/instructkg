# clustering.py
"""
Minimal per-concept unsupervised context clustering using sentence-transformers + (optional) UMAP + HDBSCAN.

Reads:
  - out/chunks.jsonl   (chunk_id -> chunk text)
  - out/mentions.jsonl (from llm.py: concept_id + chunk_id, etc.)

Writes:
  - out/context_clusters.jsonl (one record per concept_id) like:

    {
      "concept_id": "LEFT_OUTER_JOIN",
      "context_clusters": [
        {
          "cluster_id": 4,
          "count_chunks": 6,
          "label_hint": "joins-and-null-semantics",
          "chunks": [
            { "chunk_id": "lec3__0012", "text": "..." },
            { "chunk_id": "lec3__0044", "text": "..." }
          ]
        }
      ]
    }


Notes:
- We deduplicate contexts by (concept_id, chunk_id) so each chunk counts once per concept.
- label_hint is a compact slug derived from c-TF-IDF top terms for that cluster.

Install deps:
  pip install sentence-transformers scikit-learn
  pip install umap-learn hdbscan   # recommended if you want real clustering beyond fallback

Run:
  python clustering.py \
    --chunks out/chunks.jsonl \
    --mentions out/mentions.jsonl \
    --out out/context_clusters.jsonl \
    --use-umap

If you don't pass --use-umap, the script will fallback to a single cluster per concept
(keeps behavior safe/deterministic).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
from sklearn.decomposition import PCA


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
    - uses first 2 tokens as "<a>-and-<b>" when possible
    - then appends remaining tokens with "-"
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
    Convert c-TF-IDF terms (which may be multiword ngrams) into a stable-ish slug.
    Example terms: ["null semantics", "outer join", "query example"]
    -> tokens ["null","semantics","outer","join"] -> "null-and-semantics-outer-join"
    """
    flat_tokens: List[str] = []
    for term in terms or []:
        for w in (term or "").split():
            if not w:
                continue
            flat_tokens.append(w)
            if len(flat_tokens) >= 12:  # cap before slugify
                break
        if len(flat_tokens) >= 12:
            break
    return slugify_tokens(flat_tokens, max_tokens=max_tokens)


# ----------------------------
# Context building from llm.py outputs
# ----------------------------

def build_chunks_index(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a chunk_id -> chunk record index, with a few safe aliases so mentions chunk_ids
    still resolve even if your chunk ids differ slightly (e.g., '__0022' vs '__22', '.pdf' suffix).
    """
    import re

    def _aliases(cid: str) -> List[str]:
        cid = (cid or "").strip()
        if not cid:
            return []
        out = {cid}

        # common suffix normalization: PREFIX__0022 <-> PREFIX__22
        m = re.match(r"^(.*)__0*(\d+)$", cid)
        if m:
            prefix, num = m.group(1), int(m.group(2))
            out.add(f"{prefix}__{num}")          # no leading zeros
            out.add(f"{prefix}__{num:04d}")      # 4-digit
            out.add(f"{prefix}__{num:05d}")      # 5-digit (just in case)

        # drop a few common extensions/suffixes
        for ext in [".pdf", ".pptx", ".ppt", ".docx"]:
            if ext in cid:
                out.add(cid.replace(ext, ""))

        return list(out)

    by_id: Dict[str, Dict[str, Any]] = {}
    for ch in chunks:
        raw = ch.get("chunk_id")
        if raw is None:
            continue
        cid = str(raw).strip()
        if not cid:
            continue

        for a in _aliases(cid):
            # keep the first seen record for an alias (stable)
            by_id.setdefault(a, ch)

    return by_id


def build_concept_to_chunk_ids(mentions: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Deduplicate by (concept_id, chunk_id). We only need chunk IDs per concept for clustering.
    """
    seen = set()
    out: Dict[str, List[str]] = defaultdict(list)
    for m in mentions:
        concept_id = m.get("concept_id")
        chunk_id = m.get("chunk_id")
        if not concept_id or not chunk_id:
            continue
        key = (str(concept_id), str(chunk_id))
        if key in seen:
            continue
        seen.add(key)
        out[str(concept_id)].append(str(chunk_id))
    return out


def concept_context_texts(
    concept_id: str,
    chunk_ids: List[str],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    include_concept_in_text: bool = False,
    concept_label: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Returns (kept_chunk_ids, texts) with 1:1 alignment.

    Fixes your current bug: many pipelines don't store text under 'text'.
    We try multiple keys and also support list-based fields (lines/sentences).
    """
    def _get_text(ch: Dict[str, Any]) -> str:
        # try common keys
        for k in ("text", "chunk_text", "content", "raw_text", "page_text", "body", "markdown"):
            v = ch.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # try list fields
        for k in ("lines", "sentences", "paragraphs"):
            v = ch.get(k)
            if isinstance(v, list):
                s = "\n".join(str(x) for x in v if str(x).strip()).strip()
                if s:
                    return s

        return ""

    kept_chunk_ids: List[str] = []
    texts: List[str] = []

    for cid in chunk_ids:
        cid = str(cid).strip()
        if not cid:
            continue

        ch = chunks_by_id.get(cid)
        if not ch:
            continue

        t = _get_text(ch)
        if not t:
            continue

        if include_concept_in_text:
            label = concept_label or concept_id
            t = f"{t}\n\nCONCEPT: {label}"

        kept_chunk_ids.append(cid)
        texts.append(t)

    return kept_chunk_ids, texts





# ----------------------------
# ML bits (embedding + optional UMAP + HDBSCAN + c-TF-IDF)
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

    # Defensive: ensure list[str]
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
    # Defensive: if a single vector comes back, make it (1, dim)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    return X



def reduce_umap(X, *, n_neighbors: int, n_components: int):
    *_ , umap, _ = _import_or_die()[5:]  # just to keep lint quiet
    # Actually fetch proper tuple:
    SentenceTransformer, np, CountVectorizer, TfidfTransformer, normalize, umap, hdbscan = _import_or_die()
    if umap is None:
        raise RuntimeError("umap-learn not installed. Install: pip install umap-learn")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(X)


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
# Per-concept clustering (minimal output)
# ----------------------------

def cluster_concept(
    concept_id: str,
    texts: List[str],
    chunk_ids: List[str],
    *,
    embedding_model: str,
    batch_size: int,
    normalize_embeddings: bool,
    use_umap: bool,
    umap_neighbors: int,
    umap_components: int,
    min_contexts_to_cluster: int,
    min_cluster_size: Optional[int],
    min_samples: Optional[int],
    top_terms: int,
) -> Dict[str, Any]:
    """
    Output (minimal + useful):
      {
        "concept_id": ...,
        "context_clusters": [
          {
            "cluster_id": int,
            "label_hint": str,
            "count_chunks": int,
            "chunks": [{"chunk_id": str, "text": str}, ...]
          },
          ...
        ]
      }

    Notes:
    - We drop chunk_cluster_assignments (redundant once chunks are stored per cluster).
    - We keep noise chunks (cluster_id = -1) as a cluster labeled "noise" so nothing is lost.
    """
    n = len(texts)
    if len(chunk_ids) != n:
        raise ValueError(f"[{concept_id}] chunk_ids and texts length mismatch: {len(chunk_ids)} != {n}")

    def _cluster_chunk_items(indices: List[int]) -> List[Dict[str, Any]]:
        items = [{"chunk_id": chunk_ids[i], "text": texts[i]} for i in indices]
        items.sort(key=lambda x: x["chunk_id"])  # stable ordering
        return items

    def _one_cluster_label_from_texts() -> Dict[str, Any]:
        # Use c-TF-IDF on the single combined "cluster" to get a non-misc label
        try:
            terms = ctfidf_terms_per_cluster([" ".join(texts)], top_terms=top_terms)[0]
            hint = terms_to_label_hint(terms) if terms else "misc"
        except Exception:
            hint = "misc"

        return {
            "concept_id": concept_id,
            "context_clusters": [
                {
                    "cluster_id": 0,
                    "label_hint": hint,
                    "count_chunks": n,
                    "chunks": _cluster_chunk_items(list(range(n))),
                }
            ],
        }

    if n == 0:
        return {"concept_id": concept_id, "context_clusters": []}

    # IMPORTANT: in your codebase, 'use_umap' is effectively "enable clustering"
    enable_clustering = bool(use_umap)

    # Always embed
    X = embed_texts(
        texts,
        model_name=embedding_model,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )

    # Fallback: too few contexts OR clustering disabled
    if (n < int(min_contexts_to_cluster)) or (not enable_clustering):
        return _one_cluster_label_from_texts()

    # PCA reduction
    n_components = max(2, min(int(umap_components), n, X.shape[1]))
    Xr = PCA(n_components=n_components, random_state=42).fit_transform(X)

    # HDBSCAN parameters tuned for small n
    if min_cluster_size is not None:
        mcs = int(min_cluster_size)
    else:
        mcs = max(2, min(8, n // 3 if n >= 6 else 2))

    labels = cluster_hdbscan(Xr, min_cluster_size=mcs, min_samples=min_samples)

    # group indices by label
    by_label: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_label[int(lab)].append(i)

    # If all noise => treat as one cluster (but label it from text)
    non_noise = sorted([lab for lab in by_label.keys() if lab != -1])
    if not non_noise:
        return _one_cluster_label_from_texts()

    # label hints using c-TF-IDF for non-noise clusters only
    cluster_docs = [" ".join(texts[i] for i in by_label[lab]) for lab in non_noise]
    term_lists = ctfidf_terms_per_cluster(cluster_docs, top_terms=top_terms)
    terms_by_label = {lab: terms for lab, terms in zip(non_noise, term_lists)}

    context_clusters: List[Dict[str, Any]] = []

    # Build cluster records for ALL labels (including noise = -1)
    for lab, idxs in by_label.items():
        if lab == -1:
            label_hint = "noise"
        else:
            label_hint = terms_to_label_hint(terms_by_label.get(lab, []))

        context_clusters.append(
            {
                "cluster_id": int(lab),
                "label_hint": label_hint,
                "count_chunks": len(idxs),
                "chunks": _cluster_chunk_items(idxs),
            }
        )

    # Sort: bigger clusters first; put noise last
    context_clusters.sort(
        key=lambda x: (
            1 if x["cluster_id"] == -1 else 0,  # noise last
            -x["count_chunks"],                  # bigger first
            x["cluster_id"],                     # stable tiebreak
        )
    )

    return {
        "concept_id": concept_id,
        "context_clusters": context_clusters,
    }

def cluster_global_chunks(
    *,
    chunks: List[Dict[str, Any]],
    mentions: List[Dict[str, Any]],
    embedding_model: str,
    batch_size: int,
    normalize_embeddings: bool,
    use_umap: bool,
    umap_neighbors: int,
    umap_components: int,
    min_cluster_size: Optional[int],
    min_samples: Optional[int],
    top_terms: int,
    top_k_concept_role: int = 3,
    top_k_chunks_per_cluster: int = 2,
) -> List[Dict[str, Any]]:
    """
    Global clustering across ALL chunks.

    Output records (one per global cluster):
    {
      "cluster_id": 7,
      "label_hint": "...",
      "count_chunks": 42,
      "concept_role_tuples": [
        {"concept":"Gradient Descent","role":"definition","count":12},
        ...
      ],
      "chunks":[{"chunk_id":"...","text":"..."}]  # top 2 representative chunks, FULL text
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

    def _build_chunk_to_concept_role_counts(
        mentions_: List[Dict[str, Any]]
    ) -> Dict[str, Dict[Tuple[str, str], int]]:
        # chunk_id -> {(concept_label, role): count}
        out: Dict[str, Dict[Tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
        for m in mentions_:
            cid = m.get("chunk_id")
            if not cid:
                continue
            concept = (m.get("concept") or m.get("concept_id") or "").strip()
            role = (m.get("role") or "").strip().lower()
            if not concept or not role:
                continue
            out[str(cid)][(concept, role)] += 1
        return out

    def _top_concept_role_for_cluster(
        chunk_ids: List[str],
        chunk_to_counts: Dict[str, Dict[Tuple[str, str], int]],
        k: int,
    ) -> List[Dict[str, Any]]:
        agg: Dict[Tuple[str, str], int] = defaultdict(int)
        for cid in chunk_ids:
            for (concept, role), ct in chunk_to_counts.get(cid, {}).items():
                agg[(concept, role)] += int(ct)

        top = sorted(agg.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[:k]
        return [{"concept": c, "role": r, "count": n} for (c, r), n in top]

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

    # If clustering disabled, return one big cluster (id=0)
    if not enable_clustering:
        try:
            terms = ctfidf_terms_per_cluster([" ".join(texts)], top_terms=top_terms)[0]
            hint = terms_to_label_hint(terms) if terms else "misc"
        except Exception:
            hint = "misc"

        chunk_to_counts = _build_chunk_to_concept_role_counts(mentions)
        top_tuples = _top_concept_role_for_cluster(chunk_ids, chunk_to_counts, top_k_concept_role)
        top_chunks = [{"chunk_id": chunk_ids[i], "text": texts[i]} for i in range(min(top_k_chunks_per_cluster, len(texts)))]

        return [
    {
        "cluster_id": 0,
        "label_hint": hint,
        "count_chunks": len(chunk_ids),
        "chunk_ids": chunk_ids,            # ✅ NEW: all member chunk ids
        "concept_role_tuples": top_tuples, # ✅ rename to be consistent
        "chunks": top_chunks,              # top-2 preview texts
    }
]


    # ---- reduce + cluster ----
    n = len(texts)
    n_components = max(2, min(int(umap_components), n, X.shape[1]))
    Xr = PCA(n_components=n_components, random_state=42).fit_transform(X)

    if min_cluster_size is not None:
        mcs = int(min_cluster_size)
    else:
        # sensible default for global clustering
        mcs = max(5, min(25, n // 50 if n >= 500 else max(5, n // 20)))

    labels = cluster_hdbscan(Xr, min_cluster_size=mcs, min_samples=min_samples)

    by_label: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_label[int(lab)].append(i)

    non_noise = sorted([lab for lab in by_label.keys() if lab != -1])
    cluster_docs = [" ".join(texts[i] for i in by_label[lab]) for lab in non_noise]
    term_lists = ctfidf_terms_per_cluster(cluster_docs, top_terms=top_terms) if non_noise else []
    terms_by_label = {lab: terms for lab, terms in zip(non_noise, term_lists)}

    # Precompute per-chunk embedding norms for representative selection
    # Representative chunks = closest to cluster centroid (cosine, since we normalize embeddings typically)
    SentenceTransformer, np, *_ = _import_or_die()
    Xnp = np.asarray(X, dtype="float32")

    chunk_to_counts = _build_chunk_to_concept_role_counts(mentions)

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
        # cosine similarity ~ dot when normalized; still works reasonably even if not perfectly normalized
        sims = (Xc @ centroid.T).reshape(-1)
        order = sims.argsort()[::-1]  # best first
        chosen = [idxs[int(j)] for j in order[:top_k_chunks_per_cluster]]

        cluster_chunk_ids = [chunk_ids[i] for i in idxs]
        top_tuples = _top_concept_role_for_cluster(cluster_chunk_ids, chunk_to_counts, top_k_concept_role)

        out.append(
    {
        "cluster_id": int(lab),
        "label_hint": label_hint,
        "count_chunks": len(idxs),
        "chunk_ids": cluster_chunk_ids,     # ✅ NEW: all member chunk ids
        "concept_role_tuples": top_tuples,
        "chunks": [{"chunk_id": chunk_ids[i], "text": texts[i]} for i in chosen],  # still top-2
    }
)


    # sort: bigger first, noise last
    out.sort(
        key=lambda r: (
            1 if r["cluster_id"] == -1 else 0,
            -r["count_chunks"],
            r["cluster_id"],
        )
    )
    return out
