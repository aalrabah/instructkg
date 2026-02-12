from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import json_repair
import numpy as np
from tqdm import tqdm

from vllm import LLM, SamplingParams

from prompts import (
    Output,
    prompt_m1_concept_node_validity_ordinal,
    prompt_m1_concept_triplet_accuracy_ordinal,
)

# --------------------------------------------------
# vLLM structured outputs compatibility layer
# --------------------------------------------------
# Newer vLLM: StructuredOutputsParams + SamplingParams(structured_outputs=...)
# Older vLLM: GuidedDecodingParams + SamplingParams(guided_decoding=...)
try:
    from vllm.sampling_params import StructuredOutputsParams  # vLLM newer API
except Exception:
    StructuredOutputsParams = None  # type: ignore

try:
    from vllm.sampling_params import GuidedDecodingParams  # older API (removed in v0.12+)
except Exception:
    GuidedDecodingParams = None  # type: ignore


def _make_sampling_params(schema: dict, temperature: float, max_tokens: int) -> SamplingParams:
    """
    Build SamplingParams with JSON-structured decoding if supported by the installed vLLM.
    Falls back gracefully if the installed version doesn't support the chosen fields.
    """
    base_kwargs = dict(temperature=temperature, max_tokens=max_tokens)

    # 1) New vLLM path: structured_outputs
    if StructuredOutputsParams is not None:
        try:
            so = StructuredOutputsParams(json=schema)
            return SamplingParams(**base_kwargs, structured_outputs=so)
        except TypeError:
            # SamplingParams doesn't accept structured_outputs in this version
            pass
        except Exception:
            pass

    # 2) Old vLLM path: guided_decoding
    if GuidedDecodingParams is not None:
        try:
            gd = GuidedDecodingParams(json=schema)
            return SamplingParams(**base_kwargs, guided_decoding=gd)
        except TypeError:
            # SamplingParams doesn't accept guided_decoding in this version
            pass
        except Exception:
            pass

    # 3) Fallback: no structured decoding
    return SamplingParams(**base_kwargs)


# --------------------------------------------------
# LLM batch inference
# --------------------------------------------------
def batch_llm_inference(llm: LLM, messages_list: List[List[Dict]], schema: dict,
                        temperature: float = 0.0, max_tokens: int = 2048) -> List[Optional[dict]]:
    params = _make_sampling_params(schema=schema, temperature=temperature, max_tokens=max_tokens)

    # Use keyword argument for compatibility across vLLM versions
    raw = llm.chat(messages_list, sampling_params=params, use_tqdm=False)

    outputs: List[Optional[dict]] = []
    for r in raw:
        # vLLM returns objects with .outputs[0].text in offline mode
        text = r.outputs[0].text
        try:
            json_output = json_repair.loads(text)
            if (isinstance(json_output, list)) and json_output and isinstance(json_output[-1], dict):
                json_output = json_output[-1]
            outputs.append(json_output if isinstance(json_output, dict) else None)
        except Exception as e:
            print(f"⚠ JSON parsing error: {e}")
            outputs.append(None)

    return outputs


# --------------------------------------------------
# Metric evaluators
# --------------------------------------------------
def eval_node_significance(llm: LLM, data: List[dict], course_name: str):
    node_to_excerpts: Dict[str, List[str]] = {}

    for item in data:
        if item.get("relation") is None:
            continue
        for side in ["A", "B"]:
            node = item[side]["name"]
            excerpts = [e["text"] for e in item.get("evidence_chunks", []) if "text" in e]
            node_to_excerpts.setdefault(node, []).extend(excerpts)

    prompts = [
        [{
            "role": "user",
            "content": prompt_m1_concept_node_validity_ordinal(node, ex[:5], course_name)
        }]
        for node, ex in node_to_excerpts.items()
    ]
    print(f"Generated {len(prompts)} prompts for node significance")

    outputs = batch_llm_inference(llm, prompts, Output.model_json_schema())
    if outputs:
        print(f"First output: {outputs[0]}")

    scores = []
    for o in outputs:
        if isinstance(o, dict) and "score" in o:
            try:
                scores.append(float(o["score"]))
            except Exception:
                pass

    if scores:
        return {
            "mean": float(np.mean(scores)) / 2.0,
            "std": float(np.std(scores)) / 2.0
        }
    return None


def eval_triplet_accuracy(llm: LLM, data: List[dict], course_name: str):
    prompts = []

    for item in data:
        edge = {
            "source": item["A"]["name"],
            "relation_type": "None" if item.get("relation") is None else item["relation"],
            "target": item["B"]["name"],
        }
        excerpts = [e["text"] for e in item.get("evidence_chunks", []) if "text" in e]

        prompts.append([{
            "role": "user",
            "content": prompt_m1_concept_triplet_accuracy_ordinal(edge, excerpts[:5], course_name)
        }])

    print(f"Generated {len(prompts)} prompts for triplet accuracy")
    outputs = batch_llm_inference(llm, prompts, Output.model_json_schema())
    if outputs:
        print(f"First output: {outputs[0]}")

    scores = []
    for o in outputs:
        if isinstance(o, dict) and "score" in o:
            try:
                scores.append(float(o["score"]))
            except Exception:
                pass

    if scores:
        return {
            "mean": float(np.mean(scores)) / 2.0,
            "std": float(np.std(scores)) / 2.0
        }
    return None


# --------------------------------------------------
# Process single file
# --------------------------------------------------
def process_file(llm: LLM, path: str, course_name: str, method_name: str):
    """Process a single JSONL file and return results"""
    fname = os.path.basename(path)
    print(f"\n{'='*60}")
    print(f"Processing: {fname}")
    print(f"Method: {method_name}")
    print(f"Course: {course_name}")
    print(f"{'='*60}")

    # Load and parse JSONL with better error handling
    data: List[dict] = []
    try:
        with open(path, "r") as f:
            lines = f.readlines()
            print(f"File has {len(lines)} lines")

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠ JSON error on line {i+1}: {e}")
                    continue

            print(f"Successfully parsed {len(data)} valid JSON objects")

    except FileNotFoundError:
        print(f"✗ File not found: {path}")
        return None
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return None

    if not data:
        print("✗ No valid data loaded - file may be empty or malformed")
        return None

    # Run evaluations
    try:
        print("\n→ Evaluating node significance...")
        ns = eval_node_significance(llm, data, course_name)
        print(f"  Result: {ns}")

        print("\n→ Evaluating triplet accuracy...")
        ta = eval_triplet_accuracy(llm, data, course_name)
        print(f"  Result: {ta}")

        return {"node_significance": ns, "triplet_accuracy": ta}

    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# --------------------------------------------------
# Main evaluation loop
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        default="experiments_outputs",
        help="Root directory containing method folders with JSONL files",
    )
    parser.add_argument(
        "--input_file",
        default=None,
        help="Single JSONL file to evaluate (for testing)",
    )
    parser.add_argument(
        "--course_name",
        default=None,
        help="Course name (required when using --input_file). Options: 'algo', 'anlp', 'sql'",
    )
    parser.add_argument(
        "--method_name",
        default="test_method",
        help="Method name for single file evaluation (default: 'test_method')",
    )
    parser.add_argument("--output_json", default="final_eval_complete.json")
    parser.add_argument("--model_name", default="openai/gpt-oss-120b")
    args = parser.parse_args()

    print(f"Initializing vLLM with model: {args.model_name}")
    llm = LLM(
        model=args.model_name,
        max_model_len=131072,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_num_seqs=400,
        trust_remote_code=True,
    )
    print("✓ Model loaded")

    # SINGLE FILE MODE
    if args.input_file:
        if not args.course_name:
            print("✗ Error: --course_name required when using --input_file")
            print("   Options: 'algo', 'anlp', 'sql'")
            return

        course_map = {
            "algo": "Efficient Algorithms and Intractable Problems",
            "anlp": "Advanced Topics in Natural Language Processing",
            "sql": "Database Systems",
        }
        if args.course_name not in course_map:
            print(f"✗ Error: Invalid course name '{args.course_name}'")
            print(f"   Options: {list(course_map.keys())}")
            return

        full_course_name = course_map[args.course_name]
        result = process_file(llm, args.input_file, full_course_name, args.method_name)

        if result:
            print(f"\n{'='*60}")
            print("RESULTS:")
            print(f"{'='*60}")
            print(json.dumps(result, indent=2))
        return

    # BATCH MODE (original behavior)
    if os.path.exists(args.output_json):
        with open(args.output_json, "r") as f:
            results = json.load(f)
        print("Previous results loaded...")
        print(results)
    else:
        results = {
            "anlp": {"node_significance": {}, "triplet_accuracy": {}},
            "algo": {"node_significance": {}, "triplet_accuracy": {}},
            "sql": {"node_significance": {}, "triplet_accuracy": {}},
        }

    for method in sorted(os.listdir(args.input_root)):
        method_dir = os.path.join(args.input_root, method)
        if not os.path.isdir(method_dir):
            continue

        for path in glob.glob(os.path.join(method_dir, "*.jsonl")):
            fname = os.path.basename(path)
            model = fname.split("_")[-1].replace(".jsonl", "")
            course_code = fname.split("_")[1]

            print(f"\n{'='*60}")
            print(f"Processing: {fname}")
            print(f"Method: {method}")
            print(f"Model: {model}")
            print(f"Course: {course_code}")
            print(f"{'='*60}")

            if (
                (course_code in results)
                and (model in results[course_code]["node_significance"])
                and (method in results[course_code]["node_significance"][model])
            ):
                print("✓ Already evaluated - skipping")
                continue

            if course_code == "algo":
                course_name = "Efficient Algorithms and Intractable Problems"
            elif course_code == "anlp":
                course_name = "Advanced Topics in Natural Language Processing"
            elif course_code == "sql":
                course_name = "Database Systems"
            else:
                print(f"⚠ Unknown course code: {course_code} - skipping")
                continue

            # Load and parse JSONL
            data: List[dict] = []
            try:
                with open(path, "r") as f:
                    lines = f.readlines()
                    print(f"File has {len(lines)} lines")

                    for i, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"⚠ JSON error on line {i+1}: {e}")
                            continue

                    print(f"Successfully parsed {len(data)} valid JSON objects")

            except FileNotFoundError:
                print(f"✗ File not found: {path}")
                continue
            except Exception as e:
                print(f"✗ Error reading file: {e}")
                continue

            if not data:
                print("✗ No valid data loaded - file may be empty or malformed")
                print(f"   Check file: {path}")
                continue

            # Run evaluations
            try:
                print("\n→ Evaluating node significance...")
                ns = eval_node_significance(llm, data, course_name)
                print(f"  Result: {ns}")

                print("\n→ Evaluating triplet accuracy...")
                ta = eval_triplet_accuracy(llm, data, course_name)
                print(f"  Result: {ta}")

                # Store results
                results[course_code]["node_significance"].setdefault(model, {})[method] = ns
                results[course_code]["triplet_accuracy"].setdefault(model, {})[method] = ta

                # Save after each evaluation
                with open(args.output_json, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"\n✓ Saved results → {args.output_json}")

            except Exception as e:
                print(f"\n✗ Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()