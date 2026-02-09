# relation_judger.py
"""
LLM-based relation judging from pairpackets.

Reads:
  - out/pairpackets.jsonl

Writes:
  - out/relations.jsonl

Evidence selection:
  1. If chunk_co_occurrence.count > 0 AND >= cluster count → use chunk evidence
  2. Else if cluster_co_occurrence_with_different_chunks.count > 0 → use cluster evidence
  3. Else → skip pair (no evidence)
"""

from __future__ import annotations

import os
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from adapters import get_llm_client
from config import OUT_DIR, CONCURRENCY, LLM_MODEL
from prompts import RELATION_JUDGMENT_PROMPT_TEMPLATE

# Allowed relations
ALLOWED_RELATIONS = {"depends_on", "part_of"}

# Batch size for LLM calls
BATCH_SIZE = int(os.getenv("RELATION_BATCH_SIZE", "8"))

# Cache client
_LLM_CLIENT = get_llm_client()

# Debug knobs
DEBUG = os.getenv("RELATION_DEBUG", "0").strip() == "1"
DEBUG_N = int(os.getenv("RELATION_DEBUG_N", "3"))


# ---------------------------
# File utilities
# ---------------------------

def _ensure_out_dir() -> Path:
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_json_or_jsonl(path: str) -> Any:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        items: List[Any] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _extract_pair_from_output_record(rec: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if not isinstance(rec, dict):
        return None

    a = rec.get("A")
    b = rec.get("B")

    a_name = ""
    b_name = ""

    if isinstance(a, dict):
        a_name = str(a.get("name") or "").strip()
    else:
        a_name = str(a or "").strip()

    if isinstance(b, dict):
        b_name = str(b.get("name") or "").strip()
    else:
        b_name = str(b or "").strip()

    if a_name and b_name:
        return (a_name, b_name)
    return None


def _normalize_pair(a: str, b: str) -> Tuple[str, str]:
    """Normalize pair to (smaller, larger) alphabetically so (A,B) and (B,A) are treated the same."""
    return (min(a, b), max(a, b))


def _load_done_pairs(path: Path) -> set[Tuple[str, str]]:
    """Load already-processed pairs, normalized so (A,B) and (B,A) are the same."""
    done: set[Tuple[str, str]] = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                pair = _extract_pair_from_output_record(rec)
                if pair:
                    # Normalize: (A,B) and (B,A) are the same pair
                    done.add(_normalize_pair(pair[0], pair[1]))
            except Exception:
                continue
    return done


# ---------------------------
# JSON extraction
# ---------------------------

def _extract_first_json_object(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty LLM output.")

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start == -1:
        raise ValueError("No '{' found in output.")

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    raise ValueError("Unbalanced JSON braces in output.")


def _parse_llm_json(text: str) -> Dict[str, Any]:
    raw = _extract_first_json_object(text)
    return json.loads(raw)


# ---------------------------
# Helpers
# ---------------------------

def _lecture_id_from_chunk_id(chunk_id: str) -> str:
    s = str(chunk_id or "")
    if "__" in s:
        return s.split("__", 1)[0]
    return s


def _normalize_role(role: Any) -> str:
    r = str(role or "").strip().lower()
    if r == "definition":
        return "Definition"
    if r == "example":
        return "Example"
    if r == "assumption":
        return "Assumption"
    return "NA"


def _infer_roles_from_pairpacket(pp: Dict[str, Any]) -> Tuple[str, str]:
    rge = pp.get("role_grounded_evidence") or {}

    a_def = (rge.get("A_defined_mentions_B") or {}).get("count", 0) or 0
    a_ex = (rge.get("A_example_mentions_B") or {}).get("count", 0) or 0
    b_def = (rge.get("B_defined_mentions_A") or {}).get("count", 0) or 0

    if int(a_def) > 0:
        a_role = "Definition"
    elif int(a_ex) > 0:
        a_role = "Example"
    else:
        a_role = "NA"

    if int(b_def) > 0:
        b_role = "Definition"
    else:
        b_role = "NA"

    return a_role, b_role


# ---------------------------
# Evidence selection (NEW)
# ---------------------------

def _select_evidence_chunks(pp: Dict[str, Any], *, max_items: int = 3) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Select evidence based on:
    1. If chunk_co_occurrence.count > 0 AND >= cluster count → CHUNK_CO_OCCURRENCE
    2. Else if cluster count > 0 → CLUSTER_CO_OCCURRENCE
    3. Else → NO_EVIDENCE (skip)
    
    Returns:
        mode: str
        evidence_chunks: List[Dict]
        mode_info: Dict with counts and reason
    """
    
    # Get chunk co-occurrence
    chunk_cooc = pp.get("chunk_co_occurrence") or {}
    chunk_count = int(chunk_cooc.get("count") or 0)
    chunk_list = chunk_cooc.get("chunks") or []
    
    # Get cluster co-occurrence
    cluster_cooc = pp.get("cluster_co_occurrence_with_different_chunks") or {}
    cluster_count = int(cluster_cooc.get("count") or 0)
    cluster_list = cluster_cooc.get("clusters") or []
    
    # Base mode info
    mode_info = {
        "chunk_co_occurrence_count": chunk_count,
        "cluster_co_occurrence_count": cluster_count,
    }
    
    # Decision logic
    if chunk_count > 0:
        # Use chunk co-occurrence evidence (A and B appear together in same chunk)
        mode_info["reason"] = f"A and B co-occur in {chunk_count} chunk(s)"
        
        evidence: List[Dict[str, Any]] = []
        for ch in chunk_list[:max_items]:
            if not isinstance(ch, dict):
                continue
            cid = str(ch.get("chunk_id") or "").strip()
            if not cid:
                continue
            evidence.append({
                "source": "chunk_co_occurrence",
                "chunk_id": cid,
                "lecture_id": _lecture_id_from_chunk_id(cid),
                "text": ch.get("text") or "",
            })
        return "CHUNK_CO_OCCURRENCE", evidence, mode_info
    
    elif cluster_count > 0:
        # Use cluster co-occurrence evidence (A and B in same cluster but different chunks)
        mode_info["reason"] = f"A and B appear in different chunks within {cluster_count} shared cluster(s)"
        
        evidence: List[Dict[str, Any]] = []
        
        for cluster in cluster_list[:1]:  # Take first cluster (most relevant)
            if not isinstance(cluster, dict):
                continue
            
            cluster_id = cluster.get("cluster_id")
            label_hint = cluster.get("label_hint") or "misc"
            
            # Add A_chunks
            a_chunks = cluster.get("A_chunks") or []
            for ch in a_chunks[:max_items]:
                if not isinstance(ch, dict):
                    continue
                cid = str(ch.get("chunk_id") or "").strip()
                if not cid:
                    continue
                evidence.append({
                    "source": "cluster_A",
                    "cluster_id": cluster_id,
                    "label_hint": label_hint,
                    "chunk_id": cid,
                    "lecture_id": _lecture_id_from_chunk_id(cid),
                    "text": ch.get("text") or "",
                })
            
            # Add B_chunks
            b_chunks = cluster.get("B_chunks") or []
            for ch in b_chunks[:max_items]:
                if not isinstance(ch, dict):
                    continue
                cid = str(ch.get("chunk_id") or "").strip()
                if not cid:
                    continue
                evidence.append({
                    "source": "cluster_B",
                    "cluster_id": cluster_id,
                    "label_hint": label_hint,
                    "chunk_id": cid,
                    "lecture_id": _lecture_id_from_chunk_id(cid),
                    "text": ch.get("text") or "",
                })
        
        return "CLUSTER_CO_OCCURRENCE", evidence, mode_info
    
    else:
        mode_info["reason"] = "no evidence available"
        return "NO_EVIDENCE", [], mode_info


# ---------------------------
# Prompt building
# ---------------------------

def _format_temporal_block(pp: Dict[str, Any]) -> str:
    t = pp.get("temporal_order") or {}
    a0 = t.get("A_first_introduced_at") or {}
    b0 = t.get("B_first_introduced_at") or {}

    lines: List[str] = []
    if a0:
        lines.append(
            f'- A_first_introduced_at: lecture_index={a0.get("lecture_index")}, '
            f'lecture_id="{a0.get("lecture_id")}", chunk_id="{a0.get("chunk_id")}"'
        )
    if b0:
        lines.append(
            f'- B_first_introduced_at: lecture_index={b0.get("lecture_index")}, '
            f'lecture_id="{b0.get("lecture_id")}", chunk_id="{b0.get("chunk_id")}"'
        )

    return "\n".join(lines).strip() if lines else "- (no temporal info available)"


def _format_mode_block(mode: str, mode_info: Dict[str, Any]) -> str:
    lines = [
        f'- mode = "{mode}"',
        f'- chunk_co_occurrence_count = {mode_info.get("chunk_co_occurrence_count", 0)}',
        f'- cluster_co_occurrence_count = {mode_info.get("cluster_co_occurrence_count", 0)}',
        f'- reason = "{mode_info.get("reason", "")}"',
    ]
    return "\n".join(lines)


def _format_evidence_block(evidence_chunks: List[Dict[str, Any]], mode: str) -> str:
    if not evidence_chunks:
        return "- (no evidence chunks available)"

    lines: List[str] = []
    
    if mode == "CHUNK_CO_OCCURRENCE":
        # A and B appear together in these chunks
        for i, ch in enumerate(evidence_chunks, start=1):
            cid = ch.get("chunk_id")
            lecture_id = ch.get("lecture_id")
            txt = (ch.get("text") or "").strip()
            lines.append(f"[{i}] chunk_id=\"{cid}\", lecture_id=\"{lecture_id}\"")
            lines.append("(A and B both appear in this chunk)")
            lines.append("<chunk>")
            lines.append(txt)
            lines.append("</chunk>")
            lines.append("")
    
    elif mode == "CLUSTER_CO_OCCURRENCE":
        # A and B appear in different chunks but same cluster
        a_chunks = [ch for ch in evidence_chunks if ch.get("source") == "cluster_A"]
        b_chunks = [ch for ch in evidence_chunks if ch.get("source") == "cluster_B"]
        
        cluster_label = evidence_chunks[0].get("label_hint", "misc") if evidence_chunks else "misc"
        lines.append(f"Cluster theme: \"{cluster_label}\"")
        lines.append("")
        
        lines.append("A appears in:")
        for i, ch in enumerate(a_chunks, start=1):
            cid = ch.get("chunk_id")
            lecture_id = ch.get("lecture_id")
            txt = (ch.get("text") or "").strip()
            lines.append(f"[A-{i}] chunk_id=\"{cid}\", lecture_id=\"{lecture_id}\"")
            lines.append("<chunk>")
            lines.append(txt)
            lines.append("</chunk>")
            lines.append("")
        
        lines.append("B appears in:")
        for i, ch in enumerate(b_chunks, start=1):
            cid = ch.get("chunk_id")
            lecture_id = ch.get("lecture_id")
            txt = (ch.get("text") or "").strip()
            lines.append(f"[B-{i}] chunk_id=\"{cid}\", lecture_id=\"{lecture_id}\"")
            lines.append("<chunk>")
            lines.append(txt)
            lines.append("</chunk>")
            lines.append("")
    
    return "\n".join(lines).strip()

def _format_role_block(a_role: str, b_role: str) -> str:
    return f'- A_role = "{a_role}"\n- B_role = "{b_role}"'


def build_prompt_from_pairpacket(pp: Dict[str, Any]) -> Tuple[str, str, str, str, List[Dict[str, Any]], Dict[str, Any]]:
    pair = pp.get("pair") or ["", ""]
    A = str(pair[0]).strip()
    B = str(pair[1]).strip()

    mode, evidence_chunks, mode_info = _select_evidence_chunks(pp, max_items=3)

    # ✅ ADD THIS:
    a_role, b_role = _infer_roles_from_pairpacket(pp)

    prompt = RELATION_JUDGMENT_PROMPT_TEMPLATE.format(
        A=A,
        B=B,
        ROLE_BLOCK=_format_role_block(a_role, b_role),
        TEMPORAL_BLOCK=_format_temporal_block(pp),
        MODE_BLOCK=_format_mode_block(mode, mode_info),
        EVIDENCE_BLOCK=_format_evidence_block(evidence_chunks, mode),
    )
    return A, B, mode, prompt, evidence_chunks, mode_info


# ---------------------------
# Batched LLM call
# ---------------------------

async def _call_llm_batch(prompts: List[str], *, model: str) -> List[str]:
    batch_input = [
        [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
        for prompt in prompts
    ]

    resp = await _LLM_CLIENT.responses.create(
        model=model,
        instructions="",
        input=batch_input,
    )

    if hasattr(resp, "output_texts"):
        return resp.output_texts
    elif hasattr(resp, "output_text"):
        return [resp.output_text]
    else:
        return [str(resp)]


def _normalize_relation(rel: Any) -> Optional[str]:
    if rel is None:
        return None
    s = str(rel).strip().lower()
    if not s or s in ("null", "none"):
        return None
    s = s.replace("-", "_").replace(" ", "_")
    return s if s in ALLOWED_RELATIONS else None


def _extract_relation_and_justification(llm_obj: Dict[str, Any]) -> Tuple[Optional[str], str]:
    rel_out = _normalize_relation(llm_obj.get("relation", None))
    just = llm_obj.get("justification", "")
    just_out = str(just).strip() if just is not None else ""
    return rel_out, just_out


# ---------------------------
# Batched processing
# ---------------------------

async def judge_pairpacket_batch(
    pairpackets: List[Dict[str, Any]],
    *,
    model: str,
) -> List[Dict[str, Any]]:
    batch_data = []
    
    for pp in pairpackets:
        A, B, mode, prompt, evidence_chunks, mode_info = build_prompt_from_pairpacket(pp)
        
        # Skip if no evidence
        if mode == "NO_EVIDENCE":
            continue
        
        a_role, b_role = _infer_roles_from_pairpacket(pp)

        if DEBUG and len(batch_data) < DEBUG_N:
            print(f"[DEBUG] A={A} B={B} mode={mode} chunk_count={mode_info.get('chunk_co_occurrence_count', 0)} cluster_count={mode_info.get('cluster_co_occurrence_count', 0)} reason={mode_info.get('reason', '')}")

        batch_data.append({
            "A": A,
            "B": B,
            "a_role": a_role,
            "b_role": b_role,
            "mode": mode,
            "mode_info": mode_info,
            "prompt": prompt,
            "evidence_chunks": evidence_chunks,
            "temporal_order": pp.get("temporal_order") or {},
        })

    if not batch_data:
        return []

    prompts = [bd["prompt"] for bd in batch_data]

    try:
        responses = await _call_llm_batch(prompts, model=model)
    except Exception as e:
        print(f"Batch LLM call failed: {e}")
        responses = ["{}"] * len(prompts)

    results = []
    for i, (bd, response_text) in enumerate(zip(batch_data, responses)):
        if DEBUG and i < DEBUG_N:
            print("=============== RAW LLM OUTPUT ===============")
            print(f"A={bd['A']} | B={bd['B']} | mode={bd['mode']}")
            print((response_text or "").strip()[:1200])
            print("=============================================\n")

        try:
            parsed = _parse_llm_json(response_text)
            relation, justification = _extract_relation_and_justification(parsed)

            if not justification:
                if relation is None:
                    justification = "No clear relation is supported by the provided evidence."
                else:
                    justification = "Relation selected based on the provided evidence."

            results.append({
                "A": {"name": bd["A"], "role": bd["a_role"]},
                "B": {"name": bd["B"], "role": bd["b_role"]},
                "relation": relation,
                "justification": justification,
                "evidence_chunks": bd["evidence_chunks"],
                "_meta": {
                    "mode": bd["mode"],
                    "mode_info": bd["mode_info"],
                    "temporal_order": bd["temporal_order"],
                },
            })
        except Exception as e:
            results.append({
                "A": {"name": bd["A"], "role": bd["a_role"]},
                "B": {"name": bd["B"], "role": bd["b_role"]},
                "relation": None,
                "justification": "No decision (LLM output invalid).",
                "evidence_chunks": bd["evidence_chunks"],
                "_meta": {
                    "mode": bd["mode"],
                    "mode_info": bd["mode_info"],
                    "temporal_order": bd["temporal_order"],
                    "_error": str(e),
                },
            })

    return results


# ---------------------------
# Main processing
# ---------------------------

async def judge_pairpackets_file(
    in_path: str,
    *,
    out_path: Optional[str] = None,
    model: Optional[str] = None,
    concurrency: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> None:
    out_dir = _ensure_out_dir()
    in_path_p = Path(in_path)

    out_file = Path(out_path) if out_path else (out_dir / "relations.jsonl")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model_name = (model or os.getenv("RELATION_MODEL") or LLM_MODEL or "concepts-default").strip()
    conc = int(concurrency or int(os.getenv("RELATION_CONCURRENCY", str(CONCURRENCY))))
    bs = int(batch_size or BATCH_SIZE)

    pairpackets = _read_json_or_jsonl(str(in_path_p))
    if not isinstance(pairpackets, list):
        raise ValueError(f"Expected JSON array or JSONL list in {in_path}, got {type(pairpackets)}")

    # Load pairs that were already processed in previous runs
    done = _load_done_pairs(out_file)

    todo = []
    skipped_no_evidence = 0
    skipped_duplicate_pair = 0
    
    # Track pairs we're going to process in THIS run (separate from already-done)
    # This ensures (A,B) and (B,A) from the same input file don't both get processed
    seen_this_run: set[Tuple[str, str]] = set()
    
    for pp in pairpackets:
        pair = pp.get("pair") or ["", ""]
        A = str(pair[0]).strip()
        B = str(pair[1]).strip()
        if not A or not B:
            continue
        
        # Normalize pair so (A,B) and (B,A) are treated as the same
        normalized = _normalize_pair(A, B)
        
        # Skip if already processed in a previous run
        if normalized in done:
            skipped_duplicate_pair += 1
            continue
        
        # Skip if we've already queued this pair (in either order) for THIS run
        if normalized in seen_this_run:
            skipped_duplicate_pair += 1
            continue
        
        # Check if has evidence
        chunk_count = (pp.get("chunk_co_occurrence") or {}).get("count", 0) or 0
        cluster_count = (pp.get("cluster_co_occurrence_with_different_chunks") or {}).get("count", 0) or 0
        
        if chunk_count == 0 and cluster_count == 0:
            skipped_no_evidence += 1
            continue
        
        # Mark as seen for this run (prevents processing both (A,B) and (B,A))
        seen_this_run.add(normalized)
        todo.append(pp)

    total = len(pairpackets)
    print(f"Loaded pairpackets = {total}")
    print(f"Already done pairs (previous runs) = {len(done)}")
    print(f"Skipped (duplicate pair, same or reversed order) = {skipped_duplicate_pair}")
    print(f"Skipped (no evidence) = {skipped_no_evidence}")
    print(f"Remaining to process = {len(todo)}")
    print(f"Writing to: {out_file}")
    print(f"Concurrency={conc} BatchSize={bs} Model={model_name}")

    if not todo:
        print("✅ All pairs already processed!")
        return

    write_lock = asyncio.Lock()
    processed = 0

    async def process_batch(batch_pps: List[Dict[str, Any]]):
        nonlocal processed
        records = await judge_pairpacket_batch(batch_pps, model=model_name)

        async with write_lock:
            for rec in records:
                _append_jsonl(out_file, rec)
                processed += 1
                if processed == 1 or processed % 50 == 0:
                    print(f"[relation_judger] wrote {processed}/{len(todo)} records...")

    batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]

    semaphore = asyncio.Semaphore(conc)

    async def process_with_limit(batch):
        async with semaphore:
            await process_batch(batch)

    tasks = [process_with_limit(batch) for batch in batches]
    await asyncio.gather(*tasks)

    print(f"✅ Done. Wrote {processed} new records to {out_file}")


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-based relation judging from pairpackets.jsonl")
    parser.add_argument("--in", dest="in_path", default=str(Path(OUT_DIR) / "pairpackets.jsonl"))
    parser.add_argument("--out", dest="out_path", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)

    args = parser.parse_args()

    asyncio.run(
        judge_pairpackets_file(
            args.in_path,
            out_path=args.out_path,
            model=args.model,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
        )
    )