# pairpackets.py
"""
Build pair evidence packets from mentions and clustering output.

Reads:
  - out/mentions.jsonl (concept_id, chunk_id, role, etc.)
  - out/chunks.jsonl (chunk_id, text)
  - out/context_clusters.jsonl (cluster_id, chunk_ids, concepts, label_hint)

Writes:
  - out/pairpackets.jsonl

Output format per pair:
{
  "pair": ["A", "B"],
  "temporal_order": {...},
  "role_grounded_evidence": {...},
  "chunk_co_occurrence": {"count": N, "chunks": [...]},
  "cluster_co_occurrence_with_different_chunks": {"count": N, "clusters": [...]},
  "negative_evidence": {...},
  "confidence_features": {...}
}
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


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
# Text helper
# ----------------------------

def _get_text(ch: Dict[str, Any]) -> str:
    """Extract text from chunk record."""
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


# ----------------------------
# Index builders
# ----------------------------

def _lecture_order_from_mentions(mentions: List[Dict[str, Any]]) -> Dict[str, int]:
    """Sort lecture_id lexicographically and return order index."""
    lecture_ids = sorted({str(m.get("lecture_id")) for m in mentions if m.get("lecture_id") is not None})
    return {lid: i for i, lid in enumerate(lecture_ids)}


def _chunk_concepts_from_mentions(mentions: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Build chunk_id -> set of concept_ids."""
    chunk_concepts: Dict[str, Set[str]] = defaultdict(set)
    for m in mentions:
        cid = m.get("chunk_id")
        concept_id = m.get("concept_id")
        if not cid or not concept_id:
            continue
        chunk_concepts[str(cid)].add(str(concept_id))
    return chunk_concepts


def _concept_chunks_from_mentions(mentions: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Build concept_id -> set of chunk_ids."""
    concept_chunks: Dict[str, Set[str]] = defaultdict(set)
    for m in mentions:
        cid = m.get("chunk_id")
        concept_id = m.get("concept_id")
        if not cid or not concept_id:
            continue
        concept_chunks[str(concept_id)].add(str(cid))
    return concept_chunks


def _first_intro(mentions: List[Dict[str, Any]], lecture_order: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
    """Find earliest (lecture_index, chunk_index) mention per concept."""
    best: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}
    for m in mentions:
        concept_id = m.get("concept_id")
        lecture_id = m.get("lecture_id")
        chunk_id = m.get("chunk_id")
        if not concept_id or not lecture_id or not chunk_id:
            continue
        li = lecture_order.get(str(lecture_id), 0)
        ci = int(m.get("chunk_index", 0) or 0)
        key = str(concept_id)

        if key not in best or (li, ci) < (best[key][0], best[key][1]):
            best[key] = (li, ci, m)

    out: Dict[str, Dict[str, Any]] = {}
    for concept_id, (li, ci, m) in best.items():
        out[concept_id] = {
            "lecture_index": li,
            "lecture_id": str(m.get("lecture_id")),
            "chunk_id": str(m.get("chunk_id")),
        }
    return out


def _role_evidence_chunks(
    mentions: List[Dict[str, Any]],
    *,
    max_per_role: int = 3,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Build concept_id -> role -> list[{lecture_id, chunk_id, page_numbers, role, snippet}]
    """
    by_concept_role: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for m in mentions:
        concept_id = m.get("concept_id")
        role = (m.get("role") or "").lower()
        if not concept_id or role not in ("definition", "example", "assumption", "na"):
            continue

        rec = {
            "lecture_id": m.get("lecture_id"),
            "chunk_id": m.get("chunk_id"),
            "page_numbers": m.get("page_numbers") or [],
            "role": role,
            "snippet": (m.get("snippet") or ""),
        }
        bucket = by_concept_role[str(concept_id)][role]
        if len(bucket) < max_per_role:
            bucket.append(rec)

    return by_concept_role


def _build_chunk_text_map(chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    """Build chunk_id -> text mapping."""
    out: Dict[str, str] = {}
    for ch in chunks:
        cid = ch.get("chunk_id")
        if not cid:
            continue
        cid = str(cid).strip()
        text = _get_text(ch)
        if cid:
            out[cid] = text
    return out


def _build_cluster_index(
    clusters: List[Dict[str, Any]]
) -> Tuple[Dict[str, int], Dict[int, str], Dict[int, List[str]], Dict[int, Set[str]]]:
    """
    Build indices from clustering output.
    
    Returns:
        chunk_to_cluster: chunk_id -> cluster_id
        cluster_to_label: cluster_id -> label_hint
        cluster_to_chunks: cluster_id -> list of chunk_ids
        cluster_to_concepts: cluster_id -> set of concepts
    """
    chunk_to_cluster: Dict[str, int] = {}
    cluster_to_label: Dict[int, str] = {}
    cluster_to_chunks: Dict[int, List[str]] = {}
    cluster_to_concepts: Dict[int, Set[str]] = {}
    
    for c in clusters:
        cluster_id = c.get("cluster_id")
        if cluster_id is None:
            continue
        cluster_id = int(cluster_id)
        
        label_hint = str(c.get("label_hint") or "misc")
        cluster_to_label[cluster_id] = label_hint
        
        chunk_ids = c.get("chunk_ids") or []
        cluster_to_chunks[cluster_id] = [str(cid) for cid in chunk_ids]
        
        concepts = c.get("concepts") or []
        cluster_to_concepts[cluster_id] = set(str(x) for x in concepts)
        
        for cid in chunk_ids:
            chunk_to_cluster[str(cid)] = cluster_id
    
    return chunk_to_cluster, cluster_to_label, cluster_to_chunks, cluster_to_concepts


# ----------------------------
# Co-occurrence logic
# ----------------------------

def _cooc_counts(chunk_concepts: Dict[str, Set[str]]) -> Dict[Tuple[str, str], int]:
    """
    Count co-occurrence in SAME chunk.
    Returns pair counts for ordered tuples (A,B) where A < B.
    """
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for _, concepts in chunk_concepts.items():
        cs = sorted(concepts)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = cs[i], cs[j]
                counts[(a, b)] += 1
    return counts


def _get_chunk_cooccurrence(
    A: str,
    B: str,
    chunk_concepts: Dict[str, Set[str]],
    chunk_text_map: Dict[str, str],
) -> Dict[str, Any]:
    """
    Get all chunks where A and B co-occur.
    """
    shared_chunks: List[str] = []
    for chunk_id, concepts in chunk_concepts.items():
        if A in concepts and B in concepts:
            shared_chunks.append(chunk_id)
    
    shared_chunks.sort()
    
    return {
        "count": len(shared_chunks),
        "chunks": [
            {"chunk_id": cid, "text": chunk_text_map.get(cid, "")}
            for cid in shared_chunks
        ]
    }


def _get_cluster_cooccurrence_different_chunks(
    A: str,
    B: str,
    concept_chunks: Dict[str, Set[str]],
    chunk_to_cluster: Dict[str, int],
    cluster_to_label: Dict[int, str],
    cluster_to_concepts: Dict[int, Set[str]],
    chunk_text_map: Dict[str, str],
    chunk_concepts: Dict[str, Set[str]],
) -> Dict[str, Any]:
    """
    Get clusters where A and B appear in DIFFERENT chunks.
    
    A_chunks = chunks where A appears BUT B does NOT
    B_chunks = chunks where B appears BUT A does NOT
    
    Changed: Now returns evidence if EITHER a_only OR b_only has chunks
    (previously required BOTH to have chunks)
    """
    clusters_evidence: List[Dict[str, Any]] = []
    
    # Find clusters that contain both A and B
    for cluster_id, concepts in cluster_to_concepts.items():
        if cluster_id == -1:  # skip noise
            continue
        if A not in concepts or B not in concepts:
            continue
        
        # Get chunks for A in this cluster where B is NOT present
        a_only = {
            cid for cid in concept_chunks.get(A, set())
            if chunk_to_cluster.get(cid) == cluster_id
            and B not in chunk_concepts.get(cid, set())
        }
        
        # Get chunks for B in this cluster where A is NOT present
        b_only = {
            cid for cid in concept_chunks.get(B, set())
            if chunk_to_cluster.get(cid) == cluster_id
            and A not in chunk_concepts.get(cid, set())
        }
        
        # Include if EITHER has separate chunks (changed from AND to OR)
        if a_only or b_only:
            clusters_evidence.append({
                "cluster_id": cluster_id,
                "label_hint": cluster_to_label.get(cluster_id, "misc"),
                "A_chunks": [
                    {"chunk_id": cid, "text": chunk_text_map.get(cid, "")}
                    for cid in sorted(a_only)
                ],
                "B_chunks": [
                    {"chunk_id": cid, "text": chunk_text_map.get(cid, "")}
                    for cid in sorted(b_only)
                ],
            })
    
    clusters_evidence.sort(key=lambda x: x["cluster_id"])
    
    return {
        "count": len(clusters_evidence),
        "clusters": clusters_evidence,
    }


# ----------------------------
# Build pairpackets
# ----------------------------

def build_pairpackets(
    *,
    mentions: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    clusters: List[Dict[str, Any]],
    max_pairs: Optional[int] = None,
    min_cooc_chunks: int = 0,
    max_role_evidence_per_side: int = 3,
    progress_every: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Build pairpackets with:
    - temporal_order
    - role_grounded_evidence
    - chunk_co_occurrence (NEW)
    - cluster_co_occurrence_with_different_chunks (NEW)
    - negative_evidence
    - confidence_features
    """
    
    # Build indices
    lecture_order = _lecture_order_from_mentions(mentions)
    first_intro = _first_intro(mentions, lecture_order)
    chunk_concepts = _chunk_concepts_from_mentions(mentions)
    concept_chunks = _concept_chunks_from_mentions(mentions)
    cooc = _cooc_counts(chunk_concepts)
    role_evidence = _role_evidence_chunks(mentions, max_per_role=max_role_evidence_per_side)
    chunk_text_map = _build_chunk_text_map(chunks)
    chunk_to_cluster, cluster_to_label, cluster_to_chunks, cluster_to_concepts = _build_cluster_index(clusters)
    
    # Candidate pairs: chunk co-occurrence OR cluster co-occurrence
    # First get chunk-based candidates
    chunk_candidates = set(cooc.keys())
    
    # Then get cluster-based candidates (concepts in same cluster)
    cluster_candidates: Set[Tuple[str, str]] = set()
    for cluster_id, concepts in cluster_to_concepts.items():
        if cluster_id == -1:
            continue
        concepts_list = sorted(concepts)
        for i, a in enumerate(concepts_list):
            for b in concepts_list[i + 1:]:
                cluster_candidates.add((a, b))
    
    # Union of both
    all_candidates = chunk_candidates | cluster_candidates
    
    # Sort by chunk co-occurrence count (descending), then alphabetically
    candidates = sorted(all_candidates, key=lambda ab: (-cooc.get(ab, 0), ab[0], ab[1]))
    
    if max_pairs is not None:
        candidates = candidates[:max_pairs]
    
    out: List[Dict[str, Any]] = []
    total = len(candidates)
    
    for k, (A, B) in enumerate(candidates, start=1):
        if progress_every and (k == 1 or k % progress_every == 0 or k == total):
            print(f"[pairpackets] {k}/{total} A={A} B={B}")
        
        A_first = first_intro.get(A)
        B_first = first_intro.get(B)
        if not A_first or not B_first:
            continue
        
        gap = int(A_first["lecture_index"]) - int(B_first["lecture_index"])
        
        # Role-grounded evidence
        def_example_chunks = []
        for role in ("definition", "example"):
            def_example_chunks.extend(role_evidence.get(A, {}).get(role, []))
        
        A_defined_mentions_B_support = []
        A_example_mentions_B_support = []
        
        for ev in def_example_chunks:
            cid = str(ev["chunk_id"])
            present = B in chunk_concepts.get(cid, set())
            if not present:
                continue
            row = {
                "lecture_id": ev["lecture_id"],
                "chunk_id": ev["chunk_id"],
                "page_numbers": ev.get("page_numbers") or [],
                "A_role": ev["role"],
                "A_snippet": ev.get("snippet") or "",
                "B_present_in_chunk": True,
            }
            if ev["role"] == "definition":
                A_defined_mentions_B_support.append(row)
            else:
                A_example_mentions_B_support.append(row)
        
        B_defined_mentions_A_support = []
        for ev in role_evidence.get(B, {}).get("definition", []):
            cid = str(ev["chunk_id"])
            present = A in chunk_concepts.get(cid, set())
            if present:
                B_defined_mentions_A_support.append({
                    "lecture_id": ev["lecture_id"],
                    "chunk_id": ev["chunk_id"],
                    "page_numbers": ev.get("page_numbers") or [],
                    "B_role": "definition",
                    "B_snippet": ev.get("snippet") or "",
                    "A_present_in_chunk": True,
                })
        
        # Chunk co-occurrence (NEW)
        chunk_cooc = _get_chunk_cooccurrence(A, B, chunk_concepts, chunk_text_map)
        
        # Cluster co-occurrence with different chunks (NEW)
        cluster_cooc = _get_cluster_cooccurrence_different_chunks(
            A, B, concept_chunks, chunk_to_cluster, cluster_to_label, cluster_to_concepts, chunk_text_map, chunk_concepts
        )
        
        # Negative evidence
        A_chunks = concept_chunks.get(A, set())
        B_chunks = concept_chunks.get(B, set())
        a_total = len(A_chunks)
        b_total = len(B_chunks)
        together = chunk_cooc["count"]
        a_without_b = len(A_chunks - B_chunks)
        b_without_a = len(B_chunks - A_chunks)
        
        # Confidence features
        temporal_ok = 1 if int(B_first["lecture_index"]) < int(A_first["lecture_index"]) else 0
        
        role_support = len(A_defined_mentions_B_support) + len(A_example_mentions_B_support)
        a_role_total = len(def_example_chunks) if def_example_chunks else 0
        role_score = (role_support / a_role_total) if a_role_total > 0 else 0.0
        
        cooc_score = (together / a_total) if a_total > 0 else 0.0
        neg_rate = (a_without_b / a_total) if a_total > 0 else 0.0
        
        # Filter: must have at least some evidence
        total_evidence = chunk_cooc["count"] + cluster_cooc["count"]
        if total_evidence < 1 and min_cooc_chunks > 0:
            continue
        
        pairpacket = {
            "pair": [A, B],
            
            "temporal_order": {
                "B_first_introduced_at": B_first,
                "A_first_introduced_at": A_first,
                "gap_lectures": gap,
            },
            
            "role_grounded_evidence": {
                "A_defined_mentions_B": {
                    "count": len(A_defined_mentions_B_support),
                    "support": A_defined_mentions_B_support,
                },
                "A_example_mentions_B": {
                    "count": len(A_example_mentions_B_support),
                    "support": A_example_mentions_B_support,
                },
                "B_defined_mentions_A": {
                    "count": len(B_defined_mentions_A_support),
                    "support": B_defined_mentions_A_support,
                },
            },
            
            "chunk_co_occurrence": chunk_cooc,
            
            "cluster_co_occurrence_with_different_chunks": cluster_cooc,
            
            "negative_evidence": {
                "a_without_b": a_without_b,
                "b_without_a": b_without_a,
                "a_total": a_total,
                "b_total": b_total,
            },
            
            "confidence_features": {
                "temporal_ok": temporal_ok,
                "role_support": role_support,
                "a_role_total": a_role_total,
                "role_score": float(role_score),
                "cooc_score": float(cooc_score),
                "neg_rate": float(neg_rate),
            },
        }
        
        out.append(pairpacket)
    
    return out


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Build pairpackets from mentions and clusters")
    parser.add_argument("--mentions", required=True, help="Path to mentions.jsonl")
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--clusters", required=True, help="Path to context_clusters.jsonl")
    parser.add_argument("--out", required=True, help="Output path for pairpackets.jsonl")
    parser.add_argument("--max-pairs", type=int, default=None, help="Max pairs to process")
    parser.add_argument("--min-cooc-chunks", type=int, default=0, help="Minimum chunk co-occurrence (0 allows cluster-only pairs)")
    
    args = parser.parse_args()
    
    print(f"Loading mentions from {args.mentions}")
    mentions = read_json_or_jsonl(args.mentions)
    print(f"Loaded {len(mentions)} mentions")
    
    print(f"Loading chunks from {args.chunks}")
    chunks = read_json_or_jsonl(args.chunks)
    print(f"Loaded {len(chunks)} chunks")
    
    print(f"Loading clusters from {args.clusters}")
    clusters = read_json_or_jsonl(args.clusters)
    print(f"Loaded {len(clusters)} clusters")
    
    print("Building pairpackets...")
    pairpackets = build_pairpackets(
        mentions=mentions,
        chunks=chunks,
        clusters=clusters,
        max_pairs=args.max_pairs,
        min_cooc_chunks=args.min_cooc_chunks,
    )
    
    print(f"Writing {len(pairpackets)} pairpackets to {args.out}")
    write_jsonl(args.out, pairpackets)
    print("Done!")


if __name__ == "__main__":
    main()