from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from typing import List, Dict
import json, json_repair
import glob, os, argparse
import numpy as np
from tqdm import tqdm

from prompts import (
    Output,
    prompt_m1_concept_node_validity_ordinal,
    prompt_m1_concept_triplet_accuracy_ordinal,
)

# --------------------------------------------------
# LLM batch inference
# --------------------------------------------------

def batch_llm_inference(llm, messages_list, schema, temperature=0.0, max_tokens=2048):
    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        structured_outputs=StructuredOutputsParams(json=schema),
    )

    raw = llm.chat(messages_list, params, chat_template_kwargs={"include_reasoning": False})
    outputs = []

    for r in raw:
        text = r.outputs[0].text
        try:
            json_output = json_repair.loads(text)
            if (type(json_output) == list) and (type(json_output[-1]) == dict):
                json_output = json_output[-1]
            outputs.append(json_output)
        except Exception:
            outputs.append(None)

    return outputs


# --------------------------------------------------
# Metric evaluators
# --------------------------------------------------

def eval_node_significance(llm, data, course_name):
    node_to_excerpts = {}

    for item in data:
        if item["relation"] is None:
            continue
        for side in ["A", "B"]:
            node = item[side]["name"]
            excerpts = [e["text"] for e in item["evidence_chunks"]]
            node_to_excerpts.setdefault(node, []).extend(excerpts)

    prompts = [
        [{"role": "user",
          "content": prompt_m1_concept_node_validity_ordinal(node, ex[:5], course_name)}]
        for node, ex in node_to_excerpts.items()
    ]

    outputs = batch_llm_inference(llm, prompts, Output.model_json_schema())
    print(outputs[0])
    scores = [o["score"] for o in outputs if ((isinstance(o, dict)) and ("score" in o))]

    if scores:
        final_out = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores))
        }
    else:
        final_out = None

    return final_out


def eval_triplet_accuracy(llm, data, course_name):
    prompts = []

    for item in data:
        if item["relation"] is None:
            continue
        edge = {
            "source": item["A"]["name"],
            "relation_type": item["relation"],
            "target": item["B"]["name"],
        }
        excerpts = [e["text"] for e in item["evidence_chunks"]]

        prompts.append([{
            "role": "user",
            "content": prompt_m1_concept_triplet_accuracy_ordinal(edge, excerpts[:5], course_name)
        }])

    outputs = batch_llm_inference(llm, prompts, Output.model_json_schema())
    print(outputs[0])
    scores = [o["score"] for o in outputs if ((isinstance(o, dict)) and ("score" in o))]

    if scores:
        final_out = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores))
        }
    else:
        final_out = None

    return final_out


# --------------------------------------------------
# Main evaluation loop
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", default="experiments_outputs")
    parser.add_argument("--output_json", default="final_eval.json")
    parser.add_argument("--model_name", default="openai/gpt-oss-120b")
    args = parser.parse_args()

    llm = LLM(
        model=args.model_name,
        max_model_len=131072,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_num_seqs=400,
    )

    if os.path.exists(args.output_json):
        with open(args.output_json, "r") as f:
            results = json.load(f)
        print("Previous results loaded...")
        print(results)
    else:
        results = {
            "anlp": {
                "node_significance": {},
                "triplet_accuracy": {}
            },
            "algo": {
                "node_significance": {},
                "triplet_accuracy": {}
            },
            "sql": {
                "node_significance": {},
                "triplet_accuracy": {}
            }
        }

    for method in sorted(os.listdir(args.input_root)):
        method_dir = os.path.join(args.input_root, method)
        if not os.path.isdir(method_dir):
            continue

        for path in glob.glob(os.path.join(method_dir, "*.jsonl")):
            fname = os.path.basename(path)
            model = fname.split("_")[-1].replace(".jsonl", "")
            course_code = fname.split("_")[1]
            print("Model:", model)
            print("Course:", course_code)

            if (course_code in results) and (model in results[course_code]["node_significance"]) and (method in results[course_code]["node_significance"][model]):
                print(f"Skipping: {course_code}, {model}, {method}")
                continue

            if course_code == 'algo':
                course_name = "Efficient Algorithms and Intractable Problems"
            elif course_code == 'anlp':
                course_name = "Advanced Topics in Natural Language Processing"
            elif course_code == 'sql':
                course_name = "Database Systems"

            with open(path) as f:
                data = [json.loads(line) for line in f]

            ns = eval_node_significance(llm, data, course_name)
            ta = eval_triplet_accuracy(llm, data, course_name)

            results[course_code]["node_significance"].setdefault(model, {})[method] = ns
            results[course_code]["triplet_accuracy"].setdefault(model, {})[method] = ta

            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved results → {args.output_json}")


if __name__ == "__main__":
    main()