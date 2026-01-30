# main.py
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import ingest
import llm
import clustering
import pairpackets

from config import OUT_DIR, OPENAI_CONCEPTS_MODEL


# ----------------------------
# JSONL helpers
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


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# LLM stage wrappers
# ----------------------------

def build_chunk_concepts_from_mentions(mentions: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    chunk_id -> list of concept_ids that appear in that chunk.
    """
    m: Dict[str, Set[str]] = defaultdict(set)
    for r in mentions:
        cid = r.get("chunk_id")
        con = r.get("concept_id")
        if cid and con:
            m[str(cid)].add(str(con))
    return {k: sorted(v) for k, v in m.items()}


async def run_llm_and_cards(
    *,
    chunks_path: str,
    mentions_out: str,
    concept_cards_out: str,
    model: str,
    batch_size: int = 8,  # NEW: accept batch_size parameter
    progress_every: int = 25,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    import os
    import time

    chunks = read_jsonl(chunks_path)
    total_chunks = len(chunks)

    provider = os.getenv("LLM_PROVIDER", "openai")
    print(f"\n[llm] provider={provider} model={model}")
    print(f"[llm] loaded chunks: {total_chunks} from {chunks_path}")

    client = llm.get_llm_client()

    mentions: List[Dict[str, Any]] = []
    processed = 0
    skipped_empty = 0
    t0 = time.time()

    # ✅ Use the passed batch_size parameter
    chunk_batch = batch_size
    print(f"[llm] chunk_batch={chunk_batch}")

    i = 0
    while i < total_chunks:
        batch = chunks[i : i + chunk_batch]

        # Keep alignment with original indices for progress/debug
        batch_items: List[Tuple[int, Dict[str, Any], str]] = []
        for j, ch in enumerate(batch):
            global_i = i + j + 1
            text = (ch.get("text") or "").strip()
            if not text:
                skipped_empty += 1
                continue
            batch_items.append((global_i, ch, text))

        # Nothing to do in this batch
        if not batch_items:
            i += chunk_batch
            continue

        texts = [t for (_, _, t) in batch_items]

        # ✅ 1 call for concept extraction (HF batched)
        concepts_lists = await llm.extract_concepts_llm_batch(client, texts=texts, model=model)

        # Build all (chunk, concept) prompts for role classification in ONE batch call
        role_prompts: List[str] = []
        role_meta: List[Tuple[int, str]] = []  # (batch_item_index, concept)
        for bi, concepts in enumerate(concepts_lists):
            if not concepts:
                continue
            _, _, text = batch_items[bi]
            for c in concepts:
                role_prompts.append(f"CHUNK:\n{text}\n\nCONCEPT:\n{c}")
                role_meta.append((bi, c))

        # ✅ 1 call for ALL role classifications (HF batched)
        role_tags = await llm.classify_concept_role_llm_batch_texts(client, role_prompts, model=model) if role_prompts else []

        # Reassemble tags per chunk item
        per_item_tags: List[List[Tuple[str, Dict[str, str]]]] = [[] for _ in batch_items]
        for (bi, c), tag in zip(role_meta, role_tags):
            per_item_tags[bi].append((c, tag))

        # Emit mentions (same schema as before)
        for bi, (global_i, ch, text) in enumerate(batch_items):
            chunk_id = ch.get("chunk_id")
            lecture_id = ch.get("lecture_id")

            processed += 1
            before = len(mentions)

            concepts = concepts_lists[bi] if bi < len(concepts_lists) else []
            tags_for_item = per_item_tags[bi]

            # keep original order by concepts list
            tag_map = {c: tag for (c, tag) in tags_for_item}

            for c in concepts:
                tag = tag_map.get(c, {"role": "na", "snippet": ""})
                mentions.append(
                    {
                        "concept_id": llm.concept_to_id(c),
                        "concept": c,
                        "lecture_id": lecture_id,
                        "chunk_id": chunk_id,
                        "chunk_index": ch.get("chunk_index", global_i - 1),
                        "page_numbers": ch.get("page_numbers") or [],
                        "role": (tag.get("role") or "na").lower(),
                        "snippet": (tag.get("snippet") or "").strip(),
                        "chunk_text": text,
                    }
                )

            added = len(mentions) - before

            if progress_every and (global_i == 1 or global_i % progress_every == 0 or global_i == total_chunks):
                elapsed = time.time() - t0
                print(
                    f"[llm] {global_i}/{total_chunks} chunk_id={chunk_id} "
                    f"concepts={len(concepts)} mentions+={added} total_mentions={len(mentions)} "
                    f"(processed={processed}, skipped_empty={skipped_empty}, {elapsed:.1f}s)"
                )

        i += chunk_batch

    write_jsonl(mentions_out, mentions)
    print(f"✅ Wrote mentions: {mentions_out} (records={len(mentions)})")

    print(f"[llm] building concept_cards from mentions...")
    chunk_concepts = build_chunk_concepts_from_mentions(mentions)
    cards = llm.build_concept_cards(mentions, chunk_concepts)
    write_jsonl(concept_cards_out, cards)
    print(f"✅ Wrote concept_cards: {concept_cards_out} (concepts={len(cards)})")

    return mentions, cards




# ----------------------------
# Clustering wrapper
# ----------------------------

def run_clustering(
    *,
    chunks_path: str,
    mentions_path: str,
    clusters_out: str,
    embedding_model: str,
    embedding_batch_size: int,  # RENAMED from batch_size
    umap_components: int,
    min_contexts_to_cluster: int,
    min_cluster_size: Optional[int],
    min_samples: Optional[int],
    top_terms: int,
    progress_every: int,
) -> None:
    # Load inputs
    chunks = read_jsonl(chunks_path)
    mentions = read_jsonl(mentions_path)

    recs = clustering.cluster_global_chunks(
        chunks=chunks,
        mentions=mentions,
        embedding_model=embedding_model,
        batch_size=embedding_batch_size,  # Use embedding_batch_size
        normalize_embeddings=True,
        use_umap=True,
        umap_neighbors=15,
        umap_components=umap_components,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        top_terms=top_terms,
        top_k_concept_role=3,          # ✅ top-3 tuples
        top_k_chunks_per_cluster=2,    # ✅ top-2 chunk texts
    )

    write_jsonl(clusters_out, recs)
    print(f"✅ Wrote GLOBAL clusters: {clusters_out} (clusters={len(recs)})")



# ----------------------------
# PairPackets wrapper (ONE final file)
# ----------------------------

def _lecture_indices_for_pair(
    A: str,
    B: str,
    *,
    mentions: List[Dict[str, Any]],
) -> List[int]:
    # Same lecture ordering as in pairpackets
    lecture_order = pairpackets._lecture_order_from_mentions(mentions)

    # chunk_id -> lecture_index
    chunk_to_li: Dict[str, int] = {}
    for m in mentions:
        ch = m.get("chunk_id")
        lid = m.get("lecture_id")
        if not ch or not lid:
            continue
        li = lecture_order.get(str(lid), 0)
        if str(ch) not in chunk_to_li:
            chunk_to_li[str(ch)] = li
        else:
            chunk_to_li[str(ch)] = min(chunk_to_li[str(ch)], li)

    concept_chunks = pairpackets._concept_chunks_from_mentions(mentions)
    co_chunks = concept_chunks.get(A, set()) & concept_chunks.get(B, set())
    lecture_indices = sorted({chunk_to_li.get(ch) for ch in co_chunks if ch in chunk_to_li})
    return [x for x in lecture_indices if x is not None]


def run_pairpackets_final(
    *,
    mentions_path: str,
    clusters_with_assignments_path: str,
    pairpackets_out: str,
    max_pairs: Optional[int],
    min_cooc_chunks: int,
    max_role_evidence_per_side: int,
    progress_every: int,
) -> None:
    mentions = read_jsonl(mentions_path)

    # Step-1 build in memory
    pairs = pairpackets.build_pairpackets(
        mentions=mentions,
        chunks=[],
        max_pairs=max_pairs,
        min_cooc_chunks=min_cooc_chunks,
        max_role_evidence_per_side=max_role_evidence_per_side,
        progress_every=progress_every,
    )
    print(f"✅ Built step-1 pairs (in-memory): {len(pairs)}")

    # Step-2 theme coupling in memory (GLOBAL clustering)
    chunk_to_concepts = pairpackets._build_chunk_to_concepts(mentions_path)

    # ✅ UPDATED: _load_cluster_index now returns 3 values
    chunk_to_cluster, cluster_label, cluster_top_chunks = pairpackets._load_cluster_index(
        clusters_with_assignments_path,
        top_k_chunks_per_cluster=2,
    )

    total = len(pairs)
    for i, rec in enumerate(pairs, 1):
        A, B = str(rec["pair"][0]), str(rec["pair"][1])

        # ✅ UPDATED: pass cluster_top_chunks (display-only)
        theme = pairpackets._compute_theme_coupling(
            A,
            B,
            chunk_to_concepts=chunk_to_concepts,
            chunk_to_cluster=chunk_to_cluster,
            cluster_label=cluster_label,
            cluster_top_chunks=cluster_top_chunks,
        )
        rec["theme_coupling"] = theme

        # confidence_features.theme_support = max conditional presence rate
        cf = rec.get("confidence_features")
        if not isinstance(cf, dict):
            cf = {}
            rec["confidence_features"] = cf

        ts = float(theme.get("cluster_conditional_coupling", {}).get("theme_support", 0.0))
        cf["theme_support"] = ts

        # Keep theme_support only in confidence_features (not inside theme block)
        if isinstance(theme.get("cluster_conditional_coupling"), dict):
            theme["cluster_conditional_coupling"].pop("theme_support", None)

        # Add lecture_indices
        rec.setdefault("co_occurrence", {})
        rec["co_occurrence"]["lecture_indices"] = _lecture_indices_for_pair(A, B, mentions=mentions)

        if progress_every and (i == 1 or i % progress_every == 0 or i == total):
            print(f"[pairpackets final] {i}/{total}")

    # Write ONE final file
    write_jsonl(pairpackets_out, pairs)
    print(f"✅ Wrote FINAL pairpackets: {pairpackets_out} (pairs={len(pairs)})")


# ----------------------------
# NEW: Relation Judger wrapper
# ----------------------------

async def run_relation_judger(
    *,
    pairpackets_path: str,
    relations_out: str,
    model: Optional[str],
    batch_size: Optional[int],
    concurrency: Optional[int],
) -> None:
    """
    Run relation judgment on pairpackets.
    """
    import os
    from relation_judger import judge_pairpackets_file
    
    # Use provided args or fall back to env
    model_name = model or os.getenv("RELATION_MODEL") or os.getenv("LLM_MODEL") or "meta-llama/Llama-3.2-3B-Instruct"
    bs = batch_size or int(os.getenv("RELATION_BATCH_SIZE", "8"))
    conc = concurrency or int(os.getenv("RELATION_CONCURRENCY", "5"))
    
    print(f"\n[relation_judger] Starting relation judgment...")
    print(f"[relation_judger] Model={model_name} BatchSize={bs} Concurrency={conc}")
    
    await judge_pairpackets_file(
        in_path=pairpackets_path,
        out_path=relations_out,
        model=model_name,
        batch_size=bs,
        concurrency=conc,
    )


def filter_mentions_by_min_unique_chunks(
    mentions: List[Dict[str, Any]],
    *,
    min_unique_chunks: int = 3,
) -> List[Dict[str, Any]]:
    """
    Keep only concepts that appear in >= min_unique_chunks DISTINCT chunks.
    Uniqueness is based on chunk_id (not number of mention rows).
    """
    concept_to_chunks: Dict[str, Set[str]] = defaultdict(set)
    for m in mentions:
        c = m.get("concept_id")
        ch = m.get("chunk_id")
        if c and ch:
            concept_to_chunks[str(c)].add(str(ch))

    allowed = {c for c, chunks in concept_to_chunks.items() if len(chunks) >= min_unique_chunks}

    filtered = [m for m in mentions if str(m.get("concept_id")) in allowed]

    print(
        f"[filter] concepts kept={len(allowed)}/{len(concept_to_chunks)} "
        f"(min_unique_chunks={min_unique_chunks}); "
        f"mentions kept={len(filtered)}/{len(mentions)}"
    )
    return filtered



def list_pdfs_in_sequence(data_dir: Path) -> List[str]:
    """
    Return PDFs sorted by an inferred sequence number (if present), otherwise by filename.
    """
    import re

    seq_re = re.compile(
        r"""(?ix)
        (?:^|[^a-z0-9])                 # boundary
        (?:lec(?:ture)?|ch(?:apter)?|wk|week|unit|module|part)  # label words
        \s*[\.\-_:]?\s*                 # optional separators
        (\d{1,4})                       # the number
        (?:[^a-z0-9]|$)                 # boundary
        """
    )

    def _key(p: Path):
        name = p.stem
        low = name.lower()

        m = seq_re.search(low)
        if m:
            num = int(m.group(1))
            return (0, num, low)

        # Fallback: if it contains any number at all, use the first number
        m2 = re.search(r"(\d{1,4})", low)
        if m2:
            num = int(m2.group(1))
            return (0, num, low)

        # No numbers found => put after sequenced files
        return (1, 10**9, low)

    pdf_paths = sorted(data_dir.glob("*.pdf"), key=_key)
    return [str(p) for p in pdf_paths]

# ----------------------------
# CLI main
# ----------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data", help="Folder containing PDFs")
    p.add_argument("--out-dir", default=OUT_DIR)

    # control
    p.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on number of pairpackets. Omit for no limit. (0/-1 also means no limit)",
    )
    p.add_argument("--min-cooc-chunks", type=int, default=1)
    p.add_argument("--max-role-evidence-per-side", type=int, default=3)
    p.add_argument("--progress-every", type=int, default=50)

    # LLM
    p.add_argument("--llm-model", default=OPENAI_CONCEPTS_MODEL)
    
    # NEW: Unified batching/concurrency for LLM and relations
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for LLM chunks AND relation pairs")
    p.add_argument("--concurrency", type=int, default=5, help="Concurrency for relation judger")

    # clustering (separate batch-size for embeddings)
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    p.add_argument("--embedding-batch-size", type=int, default=32, help="Batch size for embeddings (clustering)")
    p.add_argument("--umap-components", type=int, default=5)
    p.add_argument("--min-contexts-to-cluster", type=int, default=3)
    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument("--min-samples", type=int, default=None)
    p.add_argument("--top-terms", type=int, default=6)

    # which steps
    p.add_argument(
        "--steps",
        nargs="+",
        default=["ingest", "llm", "clustering", "pairpackets", "relations"],
        help="Any subset of: ingest llm clustering pairpackets relations",
    )

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = str(out_dir / "chunks.jsonl")
    mentions_path = str(out_dir / "mentions.jsonl")
    concept_cards_path = str(out_dir / "concept_cards.jsonl")
    clusters_path = str(out_dir / "context_clusters.with_assignments.jsonl")
    pairpackets_path = str(out_dir / "pairpackets.jsonl")
    relations_path = str(out_dir / "relations.jsonl")  # NEW

    # ✅ treat 0/-1 as "no limit" too
    if args.max_pairs is not None and args.max_pairs <= 0:
        args.max_pairs = None

    # Step 1: ingest
    if "ingest" in args.steps:
        data_dir = Path(args.data_dir)
        pdfs = list_pdfs_in_sequence(data_dir)
        if not pdfs:
            raise SystemExit(f"❌ No PDFs found in ./{args.data_dir}")
        ingest.ingest_pdfs(pdfs, out_name="chunks.jsonl", out_dir=str(out_dir))


    # Step 2: LLM => mentions + concept cards
    if "llm" in args.steps:
        asyncio.run(
            run_llm_and_cards(
                chunks_path=chunks_path,
                mentions_out=mentions_path,
                concept_cards_out=concept_cards_path,
                model=args.llm_model,
                batch_size=args.batch_size,  # ✅ Use unified batch_size
            )
        )
        mentions = read_jsonl(mentions_path)
        mentions = filter_mentions_by_min_unique_chunks(mentions, min_unique_chunks=3)
        write_jsonl(mentions_path, mentions)

    # Step 3: clustering
    if "clustering" in args.steps:
        run_clustering(
            chunks_path=chunks_path,
            mentions_path=mentions_path,
            clusters_out=clusters_path,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,  # ✅ Use separate embedding batch size
            umap_components=args.umap_components,
            min_contexts_to_cluster=args.min_contexts_to_cluster,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            top_terms=args.top_terms,
            progress_every=args.progress_every,
        )

    # Step 4: pairpackets
    if "pairpackets" in args.steps:
        run_pairpackets_final(
            mentions_path=mentions_path,
            clusters_with_assignments_path=clusters_path,
            pairpackets_out=pairpackets_path,
            max_pairs=args.max_pairs,
            min_cooc_chunks=args.min_cooc_chunks,
            max_role_evidence_per_side=args.max_role_evidence_per_side,
            progress_every=args.progress_every,
        )

    # Step 5: relations
    if "relations" in args.steps:
        asyncio.run(
            run_relation_judger(
                pairpackets_path=pairpackets_path,
                relations_out=relations_path,
                model=args.llm_model,  # ✅ Use same model
                batch_size=args.batch_size,  # ✅ Use unified batch_size
                concurrency=args.concurrency,  # ✅ Use unified concurrency
            )
        )

    print("\n=== Outputs ===")
    print("chunks:       ", chunks_path)
    print("mentions:     ", mentions_path)
    print("concept_cards:", concept_cards_path)
    print("clusters:     ", clusters_path)
    print("pairpackets:  ", pairpackets_path)
    print("relations:    ", relations_path)  # NEW



if __name__ == "__main__":
    main()