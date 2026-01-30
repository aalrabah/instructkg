# relation_judger.py (BATCHED VERSION with ROLE-GROUNDED FALLBACK)
from __future__ import annotations

import os
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from adapters import get_llm_client
from config import OUT_DIR, CONCURRENCY, LLM_MODEL
from prompts import RELATION_JUDGMENT_PROMPT_TEMPLATE

ALLOWED_RELATIONS = {"depends_on", "part_of"}

# NEW: Batch size for LLM calls
BATCH_SIZE = int(os.getenv("RELATION_BATCH_SIZE", "8"))  # Adjust based on your GPU memory


# ---------------------------
# File utilities (unchanged)
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


def _load_done_pairs(path: Path) -> set[Tuple[str, str]]:
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
                    done.add(pair)
            except Exception:
                continue
    return done


# ---------------------------
# Robust JSON extraction (unchanged)
# ---------------------------
def _safe_get_output_text(resp: Any) -> str:
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()
    return str(resp).strip()


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
# PairPacket helpers
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
        for key in ("A_defined_mentions_B", "A_example_mentions_B"):
            sup = (rge.get(key) or {}).get("support") or []
            for item in sup:
                if isinstance(item, dict) and item.get("A_role"):
                    a_role = _normalize_role(item.get("A_role"))
                    break
            if a_role != "NA":
                break

    if int(b_def) > 0:
        b_role = "Definition"
    else:
        b_role = "NA"
        sup = (rge.get("B_defined_mentions_A") or {}).get("support") or []
        for item in sup:
            if isinstance(item, dict) and item.get("B_role"):
                b_role = _normalize_role(item.get("B_role"))
                break

    return a_role, b_role


def _select_evidence_chunks(pp: Dict[str, Any], *, max_items: int = 2) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Select evidence chunks with intelligent fallback:
    1. Try co-occurrence if count is high and concepts appear in text
    2. Try cluster if count is high and concepts appear in text
    3. Fallback to role-grounded evidence (where concepts were originally found)
    """
    pair = pp.get("pair") or ["", ""]
    A_name = str(pair[0]).strip().lower()
    B_name = str(pair[1]).strip().lower()
    
    # Helper to check if at least one concept appears in chunk
    def has_concept(text: str) -> bool:
        text_lower = text.lower()
        return A_name in text_lower or B_name in text_lower
    
    # Get co-occurrence count and chunks
    co = pp.get("co_occurrence") or {}
    co_count = int(co.get("count_chunks_together") or 0)

    # Get best cluster count and chunks
    tc = pp.get("theme_coupling") or {}
    overlaps = tc.get("cluster_overlap") or []
    
    best_cluster = None
    cluster_count = 0
    for o in overlaps:
        if not isinstance(o, dict):
            continue
        n = int(o.get("count_chunks_together_in_theme") or 0)
        if n > cluster_count:
            cluster_count = n
            best_cluster = o

    # Try CO_OCCURRENCE first if it has higher count
    if co_count > cluster_count and co_count > 0:
        top = co.get("top_chunks") or []
        out: List[Dict[str, Any]] = []
        for ch in top[:max_items]:
            if not isinstance(ch, dict):
                continue
            cid = str(ch.get("chunk_id") or "").strip()
            if not cid:
                continue
            out.append(
                {
                    "source": "co_occurrence",
                    "chunk_id": cid,
                    "lecture_id": _lecture_id_from_chunk_id(cid),
                    "page_numbers": ch.get("page_numbers") or [],
                    "text": ch.get("text") or "",
                }
            )
        
        # Verify concepts appear
        if any(has_concept(ch.get("text", "")) for ch in out):
            return "CO_OCCURRENCE", out
    
    # Try CLUSTER if it has count
    if cluster_count > 0:
        cluster_texts = []
        if isinstance(best_cluster, dict):
            cluster_texts = best_cluster.get("cluster_texts") or best_cluster.get("cluster_text") or []

        out: List[Dict[str, Any]] = []
        for ct in (cluster_texts or [])[:max_items]:
            if not isinstance(ct, dict):
                continue
            cid = str(ct.get("chunk_id") or "").strip()
            if not cid:
                continue
            out.append(
                {
                    "source": "cluster",
                    "chunk_id": cid,
                    "lecture_id": _lecture_id_from_chunk_id(cid),
                    "page_numbers": [],
                    "text": ct.get("text") or "",
                }
            )
        
        # Verify concepts appear
        if any(has_concept(ch.get("text", "")) for ch in out):
            return "CLUSTER_FALLBACK", out

    # FALLBACK: Use role-grounded evidence (where concepts were originally extracted)
    rge = pp.get("role_grounded_evidence") or {}
    role_chunks: List[Dict[str, Any]] = []
    
    # Collect from all role-grounded sources
    for key in ["A_defined_mentions_B", "A_example_mentions_B", "B_defined_mentions_A"]:
        support = (rge.get(key) or {}).get("support") or []
        for item in support:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("chunk_id") or "").strip()
            if not cid:
                continue
            
            # Use snippet if available, otherwise full text
            text = item.get("snippet") or item.get("text") or ""
            
            role_chunks.append(
                {
                    "source": "role_grounded",
                    "chunk_id": cid,
                    "lecture_id": _lecture_id_from_chunk_id(cid),
                    "page_numbers": item.get("page_numbers") or [],
                    "text": text,
                }
            )
    
    # Deduplicate by chunk_id
    seen = set()
    unique = []
    for ch in role_chunks:
        cid = ch["chunk_id"]
        if cid not in seen:
            seen.add(cid)
            unique.append(ch)
            if len(unique) >= max_items:
                break
    
    if unique:
        return "ROLE_GROUNDED", unique
    
    # Last resort: return empty
    return "NO_EVIDENCE", []


# ---------------------------
# Prompt building
# ---------------------------
def _format_temporal_block(pp: Dict[str, Any]) -> str:
    t = pp.get("temporal_order") or {}
    a0 = t.get("A_first_introduced_at") or {}
    b0 = t.get("B_first_introduced_at") or {}
    gap = t.get("gap_lectures", None)

    lines: List[str] = []
    if a0:
        lines.append(
            f'- A first introduced: lecture_index={a0.get("lecture_index")}, '
            f'lecture_id="{a0.get("lecture_id")}", chunk_id="{a0.get("chunk_id")}"'
        )
    if b0:
        lines.append(
            f'- B first introduced: lecture_index={b0.get("lecture_index")}, '
            f'lecture_id="{b0.get("lecture_id")}", chunk_id="{b0.get("chunk_id")}"'
        )
    if gap is not None:
        lines.append(f"- gap_lectures: {gap}")

    ne = pp.get("negative_evidence") or {}
    if ne:
        lines.append(
            f'- negative_evidence: a_without_b={ne.get("a_without_b")}, '
            f'b_without_a={ne.get("b_without_a")}, a_total={ne.get("a_total")}, b_total={ne.get("b_total")}'
        )

    return "\n".join(lines).strip() if lines else "- (no temporal info available)"


def _format_mode_block(mode: str) -> str:
    if mode == "CO_OCCURRENCE":
        return '- mode = "CO_OCCURRENCE"\n- Both concepts appear in the same chunks.'
    if mode == "CLUSTER_FALLBACK":
        return '- mode = "CLUSTER_FALLBACK"\n- Concepts appear in chunks from the same theme/cluster.'
    if mode == "ROLE_GROUNDED":
        return '- mode = "ROLE_GROUNDED"\n- Evidence from chunks where concepts were originally found (may not co-occur).'
    return '- mode = "NO_EVIDENCE"\n- No textual evidence available; rely on temporal ordering and course structure.'


def _format_evidence_block(evidence_chunks: List[Dict[str, Any]]) -> str:
    if not evidence_chunks:
        return "- (no evidence chunks available)"

    lines: List[str] = []
    for i, ch in enumerate(evidence_chunks, start=1):
        cid = ch.get("chunk_id")
        pages = ch.get("page_numbers") or []
        txt = (ch.get("text") or "").strip()
        lines.append(f"[{i}] chunk_id=\"{cid}\", page_numbers={pages}")
        lines.append("TEXT:")
        lines.append("<chunk>")
        lines.append(txt)
        lines.append("</chunk>")
        lines.append("")
    return "\n".join(lines).strip()


def build_prompt_from_pairpacket(pp: Dict[str, Any]) -> Tuple[str, str, str, str, List[Dict[str, Any]]]:
    pair = pp.get("pair") or ["", ""]
    A = str(pair[0]).strip()
    B = str(pair[1]).strip()

    mode, evidence_chunks = _select_evidence_chunks(pp, max_items=2)

    prompt = RELATION_JUDGMENT_PROMPT_TEMPLATE.format(
        A=A,
        B=B,
        TEMPORAL_BLOCK=_format_temporal_block(pp),
        MODE_BLOCK=_format_mode_block(mode),
        EVIDENCE_BLOCK=_format_evidence_block(evidence_chunks),
    )
    return A, B, mode, prompt, evidence_chunks


# ---------------------------
# Batched LLM call
# ---------------------------
async def _call_llm_batch(prompts: List[str], *, model: str) -> List[str]:
    """
    Call LLM with a batch of prompts.
    Returns list of output texts in the same order.
    """
    client = get_llm_client()
    
    # Build batch input payloads
    batch_input = [
        [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
        for prompt in prompts
    ]
    
    resp = await client.responses.create(
        model=model,
        instructions="",
        input=batch_input,
    )
    
    # Extract texts from batch response
    if hasattr(resp, "output_texts"):
        return resp.output_texts
    elif hasattr(resp, "output_text"):
        return [resp.output_text]
    else:
        return [str(resp)]


def _extract_relation_and_justification(llm_obj: Dict[str, Any]) -> Tuple[Optional[str], str]:
    rel = llm_obj.get("relation", None)
    if rel is None:
        rel_out = None
    else:
        rel_s = str(rel).strip().lower()
        rel_out = rel_s if rel_s in ALLOWED_RELATIONS else None

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
    """
    Process a batch of pairpackets in a single LLM call.
    """
    batch_data = []
    for pp in pairpackets:
        A, B, mode, prompt, evidence_chunks = build_prompt_from_pairpacket(pp)
        a_role, b_role = _infer_roles_from_pairpacket(pp)
        batch_data.append({
            "A": A,
            "B": B,
            "a_role": a_role,
            "b_role": b_role,
            "mode": mode,
            "prompt": prompt,
            "evidence_chunks": evidence_chunks,
            "temporal_order": pp.get("temporal_order") or {},
        })
    
    prompts = [bd["prompt"] for bd in batch_data]
    
    try:
        responses = await _call_llm_batch(prompts, model=model)
    except Exception as e:
        print(f"Batch LLM call failed: {e}")
        responses = ["{}"] * len(prompts)
    
    results = []
    for bd, response_text in zip(batch_data, responses):
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

    done = _load_done_pairs(out_file)
    
    todo = []
    for pp in pairpackets:
        pair = pp.get("pair") or ["", ""]
        A = str(pair[0]).strip()
        B = str(pair[1]).strip()
        if not A or not B:
            continue
        if (A, B) in done:
            continue
        todo.append(pp)
    
    total = len(pairpackets)
    print(f"Loaded pairpackets = {total}")
    print(f"Already done pairs = {len(done)}")
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

    parser = argparse.ArgumentParser(description="LLM-based relation judging from pairpackets.jsonl (BATCHED)")
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