from vllm import SamplingParams, LLM
from vllm.sampling_params import StructuredOutputsParams
import json_repair
from typing import List, Dict
from tqdm import tqdm

def batch_llm_inference(llm , messages_list: List[List[Dict]], schema: dict, temperature: float = 0.0, max_tokens: int = 2048) -> List[dict]:
    """
    Perform batch inference with structured output.
    
    Args:
        llm: vLLM model
        messages_list: List of message sequences (each is a list of message dicts)
        schema: JSON schema for structured output
        temperature: Sampling temperature
        
    Returns:
        List of parsed JSON responses
    """
    sampling_params = SamplingParams(
    max_tokens=max_tokens,
    temperature=temperature,
    top_p=0.95,
    structured_outputs=StructuredOutputsParams(json=schema),
    )
        
    responses = [r.outputs[0].text for r in llm.chat(messages_list, sampling_params, chat_template_kwargs={"include_reasoning": False})]

    # Parse all responses
    parsed_responses = []
    for response in responses:
        try:
            parsed = json_repair.loads(response)
            parsed_responses.append(parsed)
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response text: {response.outputs[0].text}")
            parsed_responses.append(None)
    
    return parsed_responses

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description="Retrieve & analyze interdisciplinary research."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="outputs/main_method",
        help="Path to directory with all results."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/gpt-oss-120b",
        help="LLM model name or path."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison",
        help="Path to output directory."
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Temperature for all LLM generation."
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip processing if output file already exists."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=400,
        help="Max number of samples to run."
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Initialize vLLM model
    print("Loading model...")
    llm = LLM(model=args.model_name, tensor_parallel_size=1, max_model_len=16384, gpu_memory_utilization=0.9, max_num_seqs=400)
    print("Model loaded.\n")

    with open(os.path.join(args.input_dir, "relations_algo_llama3b.jsonl"), "r") as f:
        results = json.load(f)