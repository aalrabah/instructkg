import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from adapters import get_llm_client
from collections import defaultdict

from prompts import (
    CONCEPT_EXTRACTION_PROMPT,
    ROLE_CLASSIFICATION_PROMPT,
    RoleTaggerOutput,
    ConceptExtractionOutput,
)

from response_parsing import parse_pydantic_from_llm_text


#load openai api key from env without putting the key because its already there
import re
import os
from dotenv import load_dotenv
load_dotenv()



logger = logging.getLogger(__name__)


# ----------------------------
# Low-level helpers (GPT-5 calls)
# ----------------------------

def _safe_json_loads(raw: str) -> Optional[dict]:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


async def extract_concepts_llm(
    client: Any,
    text: str,
    model: str,
) -> List[str]:

    """
    Extract concepts from a chunk of text using GPT-5.
    Returns a deduplicated list of concepts (strings).
    """
    try:
        resp = await client.responses.create(
            model=model,
            instructions=CONCEPT_EXTRACTION_PROMPT
            + "\n\nReturn ONLY the strict JSON object and nothing else.",
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                }
            ],
        )

        out = parse_pydantic_from_llm_text(resp.output_text, ConceptExtractionOutput)
        if not out or not isinstance(out.concepts, list):
            return []


        # Normalize + dedupe while preserving order
        seen = set()
        concepts: List[str] = []
        for c in out.concepts:
            if not isinstance(c, str):
                continue
            cc = c.strip()
            if not cc:
                continue
            key = cc.lower()
            if key in seen:
                continue
            seen.add(key)
            concepts.append(cc)

        return concepts

    except Exception as e:
        logger.warning(f"Concept extraction failed: {e}")
        return []


async def classify_concept_role_llm(
    client: Any,
    text: str,
    concept: str,
    model: str,
) -> Optional[Dict[str, str]]:

    try:
        resp = await client.responses.create(
            model=model,
            instructions=ROLE_CLASSIFICATION_PROMPT
            + "\n\nReturn ONLY the strict JSON object and nothing else.",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"CHUNK:\n{text}\n\nCONCEPT:\n{concept}"},
                    ],
                }
            ],
        )

        out = parse_pydantic_from_llm_text(resp.output_text, RoleTaggerOutput)
        if not out:
            # ✅ Instead of returning None, return NA with empty snippet
            logger.warning(f"Failed to parse role for concept='{concept}', defaulting to NA")
            return {"role": "na", "snippet": ""}

        # keep your downstream keys: role + snippet
        return {"role": out.role, "snippet": out.snippet.strip()}

    except Exception as e:
        logger.warning(f"Role classification failed for concept='{concept}': {e}")
        # ✅ Return NA instead of None
        return {"role": "na", "snippet": ""}

# --- Batching helpers (HF speedup) ---

def _is_hf_provider() -> bool:
    p = (os.getenv("LLM_PROVIDER", "openai") or "").lower().strip()
    return p in ("hf", "huggingface", "local")


def _resp_texts(resp: Any) -> List[str]:
    # HF batched responses return output_texts; others typically return output_text
    xs = getattr(resp, "output_texts", None)
    if isinstance(xs, list) and xs:
        return [str(x) for x in xs]
    ot = getattr(resp, "output_text", "")
    return [str(ot)]


async def extract_concepts_llm_batch(client: Any, texts: List[str], model: str) -> List[List[str]]:
    """
    Returns concepts per text, aligned with input order.
    Uses 1 batched call on HF; falls back to per-item calls otherwise.
    """
    if not texts:
        return []

    if not _is_hf_provider() or len(texts) == 1:
        return [await extract_concepts_llm(client, text=t, model=model) for t in texts]

    resp = await client.responses.create(
        model=model,
        instructions=CONCEPT_EXTRACTION_PROMPT
        + "\n\nReturn ONLY the strict JSON object and nothing else.",
        input=[
            [{"role": "user", "content": [{"type": "input_text", "text": t}]}]
            for t in texts
        ],
    )
    outs = _resp_texts(resp)

    results: List[List[str]] = []
    for raw in outs:
        out = parse_pydantic_from_llm_text(raw, ConceptExtractionOutput)
        if not out or not isinstance(out.concepts, list):
            results.append([])
            continue

        seen = set()
        concepts: List[str] = []
        for c in out.concepts:
            if not isinstance(c, str):
                continue
            cc = c.strip()
            if not cc:
                continue
            key = cc.lower()
            if key in seen:
                continue
            seen.add(key)
            concepts.append(cc)
        results.append(concepts)

    # Safety: keep alignment
    if len(results) != len(texts):
        results = (results + [[]] * len(texts))[: len(texts)]
    return results


async def classify_concept_role_llm_batch_texts(client: Any, user_texts: List[str], model: str) -> List[Dict[str, str]]:
    """
    user_texts items should be like: f"CHUNK:\\n{chunk}\\n\\nCONCEPT:\\n{concept}"
    Returns list of {"role": "...", "snippet": "..."} aligned with user_texts.
    Uses 1 batched call on HF; falls back otherwise.
    """
    if not user_texts:
        return []

    if not _is_hf_provider() or len(user_texts) == 1:
        # fallback: per-item create call (keeps behavior for OpenAI/Anthropic)
        out_tags: List[Dict[str, str]] = []
        for ut in user_texts:
            resp = await client.responses.create(
                model=model,
                instructions=ROLE_CLASSIFICATION_PROMPT
                + "\n\nReturn ONLY the strict JSON object and nothing else.",
                input=[[{"role": "user", "content": [{"type": "input_text", "text": ut}]}]],
            )
            parsed = parse_pydantic_from_llm_text(resp.output_text, RoleTaggerOutput)
            if not parsed:
                out_tags.append({"role": "na", "snippet": ""})
            else:
                out_tags.append({"role": parsed.role, "snippet": (parsed.snippet or "").strip()})
        return out_tags

    resp = await client.responses.create(
        model=model,
        instructions=ROLE_CLASSIFICATION_PROMPT
        + "\n\nReturn ONLY the strict JSON object and nothing else.",
        input=[
            [{"role": "user", "content": [{"type": "input_text", "text": ut}]}]
            for ut in user_texts
        ],
    )
    outs = _resp_texts(resp)

    tags: List[Dict[str, str]] = []
    for raw in outs:
        out = parse_pydantic_from_llm_text(raw, RoleTaggerOutput)
        if not out:
            tags.append({"role": "na", "snippet": ""})
        else:
            tags.append({"role": out.role, "snippet": (out.snippet or "").strip()})

    if len(tags) != len(user_texts):
        tags = (tags + [{"role": "na", "snippet": ""}] * len(user_texts))[: len(user_texts)]
    return tags

# ----------------------------
# Main pipeline (what you asked for)
# ----------------------------



def concept_to_id(concept: str) -> str:
    s = concept.strip().upper()
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


async def extract_concepts_with_roles_from_chunks(
    chunks: List[Dict[str, Any]],
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:

    """
    Output: mention records (one per concept occurrence), like:
    {
      "concept_id": "LEFT_OUTER_JOIN",
      "concept": c,
      "lecture_id": "...",
      "chunk_id": "...",
      "chunk_index": ...,
      "page_numbers": [...],
      "role": "example" | "definition" | "assumption" | "none",
      "snippet": "..."
    }
    """

    if not model:
        model = os.getenv("LLM_MODEL", "concepts-default")

    client = get_llm_client()
    mentions: List[Dict[str, Any]] = []

    for idx, ch in enumerate(chunks):
        text = (ch.get("text") or "").strip()
        if not text:
            continue

        concepts = await extract_concepts_llm(client, text=text, model=model)


        # (optional debug) if you keep getting empty concepts:
        # if not concepts:
        #     print(f"⚠️ No concepts for chunk_id={ch.get('chunk_id')}")
        #     continue

        for c in concepts:
            tag = await classify_concept_role_llm(client, text=text, concept=c, model=model)
            if tag is None:
                continue

            mentions.append(
                {
                    "concept_id": concept_to_id(c),
                    "concept": c,
                    "lecture_id": ch.get("lecture_id"),
                    "chunk_id": ch.get("chunk_id"),
                    "chunk_index": ch.get("chunk_index", idx),
                    "page_numbers": ch.get("page_numbers") or [],
                    "role": tag["role"].lower(),
                    "snippet": ch.get("text"),
                }
            )

    return mentions

def build_concept_cards(
    mentions: List[Dict[str, Any]],
    chunk_concepts: Dict[str, List[str]],
    lecture_order: Optional[Dict[str, int]] = None,
    top_k_cooc: int = 10,
    evidence_per_role: int = 3,
) -> List[Dict[str, Any]]:
    """
    Deterministically aggregate mention records across ALL lectures into ConceptCards.

    - mentions: output of extract_concepts_with_roles_from_chunks (mention records)
    - chunk_concepts: chunk_id -> list of concept_id present in that chunk
    - lecture_order: optional mapping lecture_id -> lecture_index (if None, inferred by sorted lecture_id)
    """

    # 1) lecture index mapping (deterministic)
    if lecture_order is None:
        lecture_ids = sorted({m["lecture_id"] for m in mentions})
        lecture_order = {lid: i for i, lid in enumerate(lecture_ids)}

    # 2) group mentions by concept_id
    by_concept: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in mentions:
        by_concept[m["concept_id"]].append(m)

    # 3) compute co-occurrence counts using chunk_concepts
    # cooc[a][b] = {"count": int, "lecture_indices": set()}
    cooc: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(lambda: {"count": 0, "lecture_indices": set()}))

    # need chunk -> lecture_index info
    chunk_to_lecture_index: Dict[str, int] = {}
    for m in mentions:
        chunk_to_lecture_index.setdefault(m["chunk_id"], lecture_order.get(m["lecture_id"], 0))

    for chunk_id, concept_ids in chunk_concepts.items():
        if not concept_ids:
            continue
        li = chunk_to_lecture_index.get(chunk_id, None)
        uniq = sorted(set(concept_ids))
        for i in range(len(uniq)):
            a = uniq[i]
            for j in range(i + 1, len(uniq)):
                b = uniq[j]
                cooc[a][b]["count"] += 1
                cooc[b][a]["count"] += 1
                if li is not None:
                    cooc[a][b]["lecture_indices"].add(li)
                    cooc[b][a]["lecture_indices"].add(li)

    # 4) build cards
    cards: List[Dict[str, Any]] = []

    for concept_id, ms in by_concept.items():
        # sort mentions by (lecture_index, chunk_index) to get first intro deterministically
        ms_sorted = sorted(
            ms,
            key=lambda x: (lecture_order.get(x["lecture_id"], 0), x.get("chunk_index", 0))
        )

        first = ms_sorted[0]
        first_li = lecture_order.get(first["lecture_id"], 0)

        # usage signals
        # normalize your 3 roles -> desired keys in output
        # (you can rename keys however you want; I map to your example)
        role_to_bucket = {
            "definition": "defined",
            "example": "example",
            "assumption": "assumed",
            "na": "na",
        }

        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for m in ms_sorted:
            bucket = role_to_bucket.get(m["role"], m["role"])
            buckets[bucket].append(m)

        usage_signals: Dict[str, Any] = {}
        for bucket_name in ["defined", "example", "assumed", "na"]:
            ev = buckets.get(bucket_name, [])
            usage_signals[bucket_name] = {
                "count": len(ev),
                "evidence": [
                    {
                        "lecture_id": e["lecture_id"],
                        "chunk_id": e["chunk_id"],
                        "page_numbers": e.get("page_numbers") or [],
                        "snippet": e["snippet"],
                        "chunk_text": e.get("chunk_text", ""),
                    }
                    for e in ev[:evidence_per_role]
                ],
            }

        # trajectory
        lecture_presence = sorted({lecture_order.get(m["lecture_id"], 0) for m in ms_sorted})
        recurrence_count = len(lecture_presence)
        gaps = []
        if len(lecture_presence) >= 2:
            for a, b in zip(lecture_presence, lecture_presence[1:]):
                if b - a > 1:
                    gaps.append(b - a - 1)

        # top co-occurrences
        neighbors = cooc.get(concept_id, {})
        top_neighbors = sorted(
            neighbors.items(),
            key=lambda kv: (-kv[1]["count"], kv[0])
        )[:top_k_cooc]

        top_cooccurring = [
            {
                "other_concept_id": other_id,
                "count": stats["count"],
                "lecture_indices": sorted(stats["lecture_indices"]),
            }
            for other_id, stats in top_neighbors
        ]

        cards.append(
            {
                "concept_id": concept_id,
                "concept_label": first.get("concept", concept_id.replace("_", " ").title()),
                "canonical_label": concept_id.replace("_", " "),
                "first_introduced_at": {
                    "lecture_index": first_li,
                    "lecture_id": first["lecture_id"],
                    "chunk_id": first["chunk_id"],
                    "page_numbers": first.get("page_numbers") or [],
                    "snippet": first["snippet"],
                },
                "usage_signals": usage_signals,
                "trajectory": {
                    "lecture_presence": lecture_presence,
                    "recurrence_count": recurrence_count,
                    "gaps": gaps,
                },
                "top_cooccurring_concepts": top_cooccurring,
            }
        )

    # deterministic order of cards
    cards.sort(key=lambda c: (c["first_introduced_at"]["lecture_index"], c["concept_id"]))
    return cards