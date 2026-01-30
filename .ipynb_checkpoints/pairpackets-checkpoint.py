# pairpackets.py
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _lecture_order_from_mentions(mentions: List[Dict[str, Any]]) -> Dict[str, int]:
    # matches your ConceptCard default behavior: sort lecture_id lexicographically
    lecture_ids = sorted({str(m.get("lecture_id")) for m in mentions if m.get("lecture_id") is not None})
    return {lid: i for i, lid in enumerate(lecture_ids)}


def _chunk_concepts_from_mentions(mentions: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    chunk_concepts: Dict[str, Set[str]] = defaultdict(set)
    for m in mentions:
        cid = m.get("chunk_id")
        concept_id = m.get("concept_id")
        if not cid or not concept_id:
            continue
        chunk_concepts[str(cid)].add(str(concept_id))
    return chunk_concepts


def _concept_chunks_from_mentions(mentions: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    concept_chunks: Dict[str, Set[str]] = defaultdict(set)
    for m in mentions:
        cid = m.get("chunk_id")
        concept_id = m.get("concept_id")
        if not cid or not concept_id:
            continue
        concept_chunks[str(concept_id)].add(str(cid))
    return concept_chunks


def _first_intro(mentions: List[Dict[str, Any]], lecture_order: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
    # earliest (lecture_index, chunk_index) mention per concept
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
    concept_id -> role -> list[{lecture_id, chunk_id, page_numbers, role, snippet}]
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


def _cooc_counts(chunk_concepts: Dict[str, Set[str]]) -> Dict[Tuple[str, str], int]:
    """
    Count co-occurrence in SAME chunk only.
    Returns pair counts for ordered tuples (A,B) where A!=B.
    """
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for _, concepts in chunk_concepts.items():
        cs = sorted(concepts)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = cs[i], cs[j]
                counts[(a, b)] += 1
                counts[(b, a)] += 1
    return counts


def build_pairpackets(
    *,
    mentions: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    max_pairs: Optional[int] = None,
    min_cooc_chunks: int = 1,
    max_role_evidence_per_side: int = 3,
    progress_every: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Step-1 PairPackets:
    - temporal
    - role-grounded coupling (definition/example chunks mention other concept in same chunk)
    - chunk co-occurrence + negative evidence
    - confidence features (no theme yet => theme_support = 0.0)
    """
    lecture_order = _lecture_order_from_mentions(mentions)
    first_intro = _first_intro(mentions, lecture_order)

    chunk_concepts = _chunk_concepts_from_mentions(mentions)
    concept_chunks = _concept_chunks_from_mentions(mentions)
    cooc = _cooc_counts(chunk_concepts)

    role_evidence = _role_evidence_chunks(mentions, max_per_role=max_role_evidence_per_side)

    # Candidate pairs: any (A,B) with cooc >= min_cooc_chunks
    candidates = [(a, b, c) for (a, b), c in cooc.items() if a != b and c >= min_cooc_chunks]

    # deterministic order: highest cooc first, then ids
    candidates.sort(key=lambda t: (-t[2], t[0], t[1]))

    if max_pairs is not None:
        candidates = candidates[:max_pairs]

    out: List[Dict[str, Any]] = []
    total = len(candidates)

    for k, (A, B, count_together) in enumerate(candidates, start=1):
        if progress_every and (k == 1 or k % progress_every == 0 or k == total):
            print(f"[pairpackets] {k}/{total} A={A} B={B} cooc={count_together}")

        A_first = first_intro.get(A)
        B_first = first_intro.get(B)
        if not A_first or not B_first:
            # if a concept somehow lacks intro, skip (should be rare)
            continue

        gap = int(A_first["lecture_index"]) - int(B_first["lecture_index"])

        # Role-grounded evidence: check A’s def/example chunks contain B in that chunk’s concept set
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

        # also symmetric check (optional but useful)
        B_defined_mentions_A_support = []
        for ev in role_evidence.get(B, {}).get("definition", []):
            cid = str(ev["chunk_id"])
            present = A in chunk_concepts.get(cid, set())
            if present:
                B_defined_mentions_A_support.append(
                    {
                        "lecture_id": ev["lecture_id"],
                        "chunk_id": ev["chunk_id"],
                        "page_numbers": ev.get("page_numbers") or [],
                        "B_role": "definition",
                        "B_snippet": ev.get("snippet") or "",
                        "A_present_in_chunk": True,
                    }
                )

        # Co-occurrence / negative evidence
        A_chunks = concept_chunks.get(A, set())
        B_chunks = concept_chunks.get(B, set())
        a_total = len(A_chunks)
        b_total = len(B_chunks)
        together = count_together
        a_without_b = len(A_chunks - B_chunks)
        b_without_a = len(B_chunks - A_chunks)

        # Confidence features (deterministic)
        temporal_ok = 1 if int(B_first["lecture_index"]) < int(A_first["lecture_index"]) else 0

        role_support = len(A_defined_mentions_B_support) + len(A_example_mentions_B_support)
        a_role_total = len(def_example_chunks) if def_example_chunks else 0
        role_score = (role_support / a_role_total) if a_role_total > 0 else 0.0

        cooc_score = (together / a_total) if a_total > 0 else 0.0
        neg_rate = (a_without_b / a_total) if a_total > 0 else 0.0

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

            # Theme coupling will be filled in Step 2
            "theme_coupling": {
                "cluster_overlap": [],
                "cluster_conditional_coupling": {},
            },

            "co_occurrence": {
                "count_chunks_together": together,
                # lecture_indices can be added later if you want; keep Step-1 minimal
            },

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
                "theme_support": 0.0,
                "cooc_score": float(cooc_score),
                "neg_rate": float(neg_rate),
            },
        }

        out.append(pairpacket)

    return out

    # ---------------------------
# Step 2: Theme coupling (clustering-aware) enrichment
# ---------------------------

from collections import defaultdict
from typing import Dict, Any, List, Tuple, Iterable, Optional
import json
from pathlib import Path


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _build_chunk_to_concepts(mentions_path: str) -> Dict[str, Set[str]]:
    """
    From out/mentions.jsonl => chunk_id -> set(concept_id)
    """
    chunk_to_concepts: Dict[str, Set[str]] = defaultdict(set)
    with open(mentions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            m = json.loads(line)
            cid = m.get("concept_id")
            ch = m.get("chunk_id")
            if cid and ch:
                chunk_to_concepts[str(ch)].add(str(cid))
    return chunk_to_concepts


def _load_cluster_index(
    clusters_with_assignments_path: str,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[int, str]]]:
    """
    Reads out/context_clusters.with_assignments.jsonl and builds:
      1) concept_chunk_to_cluster[concept_id][chunk_id] = cluster_id
      2) concept_cluster_label[concept_id][cluster_id] = label_hint
    """
    concept_chunk_to_cluster: Dict[str, Dict[str, int]] = defaultdict(dict)
    concept_cluster_label: Dict[str, Dict[int, str]] = defaultdict(dict)

    with open(clusters_with_assignments_path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            r = json.loads(line)
            concept_id = str(r.get("concept_id", "")).strip()
            if not concept_id:
                continue

            # labels
            for cc in (r.get("context_clusters") or []):
                try:
                    k = int(cc.get("cluster_id"))
                except Exception:
                    continue
                concept_cluster_label[concept_id][k] = str(cc.get("label_hint", "misc"))

            # assignments
            for a in (r.get("chunk_cluster_assignments") or []):
                ch = a.get("chunk_id")
                if not ch:
                    continue
                try:
                    k = int(a.get("cluster_id", 0))
                except Exception:
                    k = 0
                concept_chunk_to_cluster[concept_id][str(ch)] = k

            # always safe defaults
            concept_cluster_label[concept_id].setdefault(0, "misc")

    return concept_chunk_to_cluster, concept_cluster_label


def _compute_theme_coupling(
    A: str,
    B: str,
    *,
    chunk_to_concepts: Dict[str, Set[str]],
    concept_chunk_to_cluster: Dict[str, Dict[str, int]],
    concept_cluster_label: Dict[str, Dict[int, str]],
) -> Dict[str, Any]:
    """
    Theme coupling using SAME-CHUNK evidence:
      - cluster_overlap: counts of (A_cluster_id, B_cluster_id) among chunks where both appear
      - cluster_conditional_coupling: for each A_cluster_id, how often B is present when A appears in that cluster
    """
    A = str(A)
    B = str(B)

    a_chunk_map = concept_chunk_to_cluster.get(A, {})
    b_chunk_map = concept_chunk_to_cluster.get(B, {})

    a_in_cluster: Dict[int, int] = defaultdict(int)
    b_present_in_a_cluster: Dict[int, int] = defaultdict(int)
    overlap_counts: Dict[Tuple[int, int], int] = defaultdict(int)

    for chunk_id, concepts in chunk_to_concepts.items():
        if A not in concepts:
            continue

        a_k = int(a_chunk_map.get(chunk_id, 0))
        a_in_cluster[a_k] += 1

        if B in concepts:
            b_present_in_a_cluster[a_k] += 1
            b_k = int(b_chunk_map.get(chunk_id, 0))
            overlap_counts[(a_k, b_k)] += 1

    overlap_list: List[Dict[str, Any]] = []
    for (a_k, b_k), cnt in overlap_counts.items():
        overlap_list.append(
            {
                "A_cluster_id": a_k,
                "A_label_hint": concept_cluster_label.get(A, {}).get(a_k, "misc"),
                "B_cluster_id": b_k,
                "B_label_hint": concept_cluster_label.get(B, {}).get(b_k, "misc"),
                "count_chunks_together_in_theme": cnt,
            }
        )
    overlap_list.sort(key=lambda x: -x["count_chunks_together_in_theme"])

    cond: Dict[str, Any] = {}
    theme_support = 0.0
    for a_k, total in a_in_cluster.items():
        present = b_present_in_a_cluster.get(a_k, 0)
        rate = (present / total) if total > 0 else 0.0
        without = total - present
        cond[f"A_cluster_{a_k}_B_presence_rate"] = rate
        cond[f"A_cluster_{a_k}_A_without_B"] = without
        if rate > theme_support:
            theme_support = rate

    cond["theme_support"] = theme_support

    return {
        "cluster_overlap": overlap_list,
        "cluster_conditional_coupling": cond,
    }


def enrich_pairpackets_with_theme(
    *,
    pairpackets_in_path: str = "out/pairpackets.TEST.jsonl",
    mentions_path: str = "out/mentions.jsonl",
    clusters_with_assignments_path: str = "out/context_clusters.with_assignments.jsonl",
    pairpackets_out_path: str = "out/pairpackets.with_theme.jsonl",
    progress_every: int = 200,
) -> None:
    """
    Reads Step-1 pairpackets JSONL and writes a new JSONL with:
      - theme_coupling filled
      - confidence_features.theme_support set to max B_presence_rate_in_A_cluster over A clusters
    """
    from pathlib import Path

    chunk_to_concepts = _build_chunk_to_concepts(mentions_path)
    concept_chunk_to_cluster, concept_cluster_label = _load_cluster_index(clusters_with_assignments_path)

    pairs = read_jsonl(pairpackets_in_path)
    out: List[Dict[str, Any]] = []

    total = len(pairs)
    print(f"Loaded pairpackets = {total}")
    print(f"Loaded chunks (concept sets) = {len(chunk_to_concepts)}")
    print(f"Loaded cluster index concepts = {len(concept_chunk_to_cluster)}")
    print(f"Writing to: {pairpackets_out_path}")

    Path(pairpackets_out_path).parent.mkdir(parents=True, exist_ok=True)

    for i, rec in enumerate(pairs, 1):
        pair = rec.get("pair") or []
        if isinstance(pair, list) and len(pair) == 2:
            A, B = pair[0], pair[1]

            theme = _compute_theme_coupling(
                A,
                B,
                chunk_to_concepts=chunk_to_concepts,
                concept_chunk_to_cluster=concept_chunk_to_cluster,
                concept_cluster_label=concept_cluster_label,
            )
            rec["theme_coupling"] = theme

            cf = rec.get("confidence_features")
            if not isinstance(cf, dict):
                cf = {}
                rec["confidence_features"] = cf
            cf["theme_support"] = float(theme["cluster_conditional_coupling"].get("theme_support", 0.0))

        out.append(rec)

        if progress_every and (i % progress_every == 0 or i == total):
            print(f"[progress] {i}/{total}")

    write_jsonl(pairpackets_out_path, out)
    print(f"✅ Wrote {pairpackets_out_path} (pairs={len(out)})")
