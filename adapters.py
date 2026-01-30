# adapters.py
import os
import asyncio
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union, Sequence


SENTINEL_MODEL = "concepts-default"

# Optional short-name map (same idea as your other project)
HF_MODELS_MAP: Dict[str, str] = {
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-32B-Instruct",
    # add more if you want
}

# Your llm.py uses OpenAI-style "input" payloads.
# One prompt:  input_payload = [{"role":"user","content":[{"type":"input_text","text":"..."}]}]
# Batch:       input_payloads = [input_payload1, input_payload2, ...]
InputPayload = List[Dict[str, Any]]
BatchInputPayload = List[InputPayload]


def _extract_user_text(input_payload: InputPayload) -> str:
    """
    Your llm.py sends:
      input=[{"role":"user","content":[{"type":"input_text","text": "..."}]}]
    This pulls the text out in a provider-agnostic way.
    """
    try:
        if not input_payload:
            return ""
        content = input_payload[0].get("content", [])
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(str(p["text"]))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts).strip()
    except Exception:
        return str(input_payload).strip()


def _resolve_model(provider: str, requested_model: str) -> str:
    """
    If llm.py passes the sentinel model, swap it for a provider-specific model from env.
    Otherwise, keep requested_model as-is.
    """
    if (requested_model or "").strip() and requested_model != SENTINEL_MODEL:
        return requested_model

    provider = (provider or "").lower().strip()

    if provider == "openai":
        return os.getenv("OPENAI_CONCEPTS_MODEL", "gpt-5-mini-2025-08-07")
    if provider in ("anthropic", "claude"):
        return os.getenv("ANTHROPIC_CONCEPTS_MODEL", "claude-3-5-sonnet-latest")
    if provider == "gemini":
        return os.getenv("GEMINI_CONCEPTS_MODEL", "gemini-1.5-pro")

    # HF: allow LLM_MODEL or HF_CONCEPTS_MODEL
    if provider in ("hf", "huggingface", "local"):
        return os.getenv("HF_CONCEPTS_MODEL", os.getenv("LLM_MODEL", "qwen7b"))

    return requested_model or SENTINEL_MODEL


# -----------------------
# OpenAI (compatible)
# -----------------------
class _OpenAIResponses:
    def __init__(self, client, provider_name: str):
        self._client = client
        self._provider = provider_name

    async def create(self, *, model: str, instructions: str, input: InputPayload, **kwargs):
        model = _resolve_model(self._provider, model)
        resp = await self._client.responses.create(
            model=model,
            instructions=instructions,
            input=input,
            **kwargs,
        )
        return resp


class OpenAICompatClient:
    def __init__(self):
        from openai import AsyncOpenAI
        self._provider = "openai"
        self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.responses = _OpenAIResponses(self._client, self._provider)


# -----------------------
# Anthropic/Claude (compatible)
# -----------------------
class _AnthropicResponses:
    def __init__(self, client, provider_name: str):
        self._client = client
        self._provider = provider_name

    async def create(self, *, model: str, instructions: str, input: InputPayload, **kwargs):
        model = _resolve_model(self._provider, model)
        user_text = _extract_user_text(input)
        max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "800"))

        msg = await self._client.messages.create(
            model=model,
            system=instructions,
            messages=[{"role": "user", "content": user_text}],
            max_tokens=max_tokens,
        )

        # Convert Anthropic blocks -> plain text
        text = ""
        try:
            for block in msg.content:
                t = getattr(block, "text", None)
                if t:
                    text += t
        except Exception:
            text = str(msg)

        return SimpleNamespace(output_text=text.strip())


class AnthropicCompatClient:
    def __init__(self):
        from anthropic import AsyncAnthropic
        self._provider = "anthropic"
        self._client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.responses = _AnthropicResponses(self._client, self._provider)


# -----------------------
# Hugging Face Local (compatible)
# -----------------------
class _HFLocalEngine:
    """
    Lazy-load a transformers text-generation pipeline once.

    Supports:
      - generate(system, user) -> str
      - generate_many(system, [user1, user2, ...]) -> List[str]  (batched pipeline call)
    """
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.pipe = None
        self.tok = None
        self._lock = threading.Lock()  # HF pipelines are not thread-safe (esp. on GPU)

    def load(self):
        if self.pipe is not None:
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        hf_token = os.getenv("HF_TOKEN")  # optional (needed for gated models)
        self.tok = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True,
            token=hf_token,
        )

        # ✅ FIX: decoder-only models + batching need LEFT padding
        try:
            self.tok.padding_side = "left"
        except Exception:
            pass

        # Ensure pad token exists for batching/padding
        if getattr(self.tok, "pad_token_id", None) is None and getattr(self.tok, "eos_token_id", None) is not None:
            try:
                self.tok.pad_token = self.tok.eos_token
            except Exception:
                pass

        force_cpu = os.getenv("HF_FORCE_CPU", "0") == "1"

        if force_cpu:
            mdl = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                token=hf_token,
            ).to("cpu")

            self.pipe = pipeline(
                "text-generation",
                model=mdl,
                tokenizer=self.tok,
                device=-1,  # CPU
                pad_token_id=getattr(self.tok, "eos_token_id", None),
            )
        else:
            mdl = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",  # CUDA/MPS if available
                torch_dtype="auto",
                token=hf_token,
            )
            self.pipe = pipeline(
                "text-generation",
                model=mdl,
                tokenizer=self.tok,
                pad_token_id=getattr(self.tok, "eos_token_id", None),
            )


    def format_prompt(self, system: str, user: str) -> str:
        """
        Prefer tokenizer chat template when available (correct for Llama/Qwen instruct),
        otherwise fall back to your simple template.
        """
        self.load()
        assert self.tok is not None

        # Newer tokenizers for instruct models typically support this:
        if hasattr(self.tok, "apply_chat_template"):
            try:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                return self.tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Fallback: your original simple instruct template
        return (
            f"<|system|>\n{system}\n<|end|>\n"
            f"<|user|>\n{user}\n<|end|>\n<|assistant|>\n"
        )

    def _pipe_call(self, prompts: Union[str, List[str]], **gen_kwargs) -> Any:
        """
        Calls the pipeline in a lock to avoid concurrent GPU/CPU pipeline usage.
        Tries to request return_full_text=False to get only the completion.
        """
        assert self.pipe is not None

        # Try to get only the generated continuation (much easier to post-process)
        try:
            return self.pipe(prompts, return_full_text=False, **gen_kwargs)
        except TypeError:
            # Older transformers may not support return_full_text in this pipeline call
            return self.pipe(prompts, **gen_kwargs)

    def _normalize_outputs(self, outputs: Any, prompts: Optional[Sequence[str]] = None) -> List[str]:
        """
        Normalize pipeline outputs to List[str] completions.
        Handles:
          - list[dict]
          - list[list[dict]] (when num_return_sequences>1)
        """
        texts: List[str] = []

        if outputs is None:
            return texts

        # If batch
        if isinstance(outputs, list):
            for item in outputs:
                if isinstance(item, list) and item:
                    # num_return_sequences case: take first
                    item = item[0]
                if isinstance(item, dict) and "generated_text" in item:
                    texts.append(str(item["generated_text"]))
                else:
                    texts.append(str(item))
            return [t.strip() for t in texts]

        # Single weird case
        if isinstance(outputs, dict) and "generated_text" in outputs:
            return [str(outputs["generated_text"]).strip()]

        return [str(outputs).strip()]

    def generate_many(self, system: str, users: List[str]) -> List[str]:
        """
        Batched generation: one pipeline call with list[str].
        """
        self.load()
        assert self.pipe is not None

        prompts = [self.format_prompt(system, u) for u in users]

        max_new_tokens = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))
        temperature = float(os.getenv("HF_TEMPERATURE", "0.1"))
        top_p = float(os.getenv("HF_TOP_P", "1.0"))
        batch_size = int(os.getenv("HF_BATCH_SIZE", "4"))

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0),
            batch_size=batch_size,
            pad_token_id=getattr(self.tok, "eos_token_id", None),
        )

        with self._lock:
            outputs = self._pipe_call(prompts, **gen_kwargs)

        texts = self._normalize_outputs(outputs, prompts=prompts)
        return texts

    def generate(self, system: str, user: str) -> str:
        """
        Single generation (still uses the same pipeline).
        """
        texts = self.generate_many(system, [user])
        return texts[0].strip() if texts else ""


class _HFResponses:
    def __init__(self, provider_name: str):
        self._provider = provider_name
        self._engine: Optional[_HFLocalEngine] = None
        self._model_id: Optional[str] = None

    def _ensure_engine(self, requested_model: str):
        resolved = _resolve_model(self._provider, requested_model)
        # Map short names -> full HF ids
        model_id = HF_MODELS_MAP.get(resolved, resolved)

        # If model changed, rebuild engine
        if self._engine is None or self._model_id != model_id:
            self._model_id = model_id
            self._engine = _HFLocalEngine(model_id=model_id)

    async def create(
        self,
        *,
        model: str,
        instructions: str,
        input: Union[InputPayload, BatchInputPayload],
        **kwargs,
    ):
        """
        HF batching support:

        1) Single prompt (existing behavior):
             input = [{"role":"user","content":[{"type":"input_text","text":"..."}]}]
           returns: SimpleNamespace(output_text="...")

        2) Batch prompts (NEW):
             input = [input_payload1, input_payload2, ...]
           returns: SimpleNamespace(output_texts=[...], output_text="first item")
           (output_text kept for backward-compat; use output_texts for the batch.)
        """
        self._ensure_engine(model)
        assert self._engine is not None

        # Optional escape hatch that mirrors HF docs naming:
        # client.responses.create(..., text_inputs=[...])  (HF only)
        text_inputs = kwargs.pop("text_inputs", None)
        if text_inputs is not None:
            if isinstance(text_inputs, str):
                user_texts = [text_inputs]
            elif isinstance(text_inputs, list):
                user_texts = [str(x) for x in text_inputs]
            else:
                user_texts = [str(text_inputs)]

            def _run_text_inputs():
                return self._engine.generate_many(instructions, user_texts)

            texts = await asyncio.to_thread(_run_text_inputs)
            return SimpleNamespace(
                output_texts=[t.strip() for t in texts],
                output_text=(texts[0].strip() if texts else ""),
            )

        # Detect batch: input is a list whose first element is itself a list (payload)
        is_batch = bool(input) and isinstance(input, list) and isinstance(input[0], list)

        if is_batch:
            payloads: BatchInputPayload = input  # type: ignore[assignment]
            user_texts = [_extract_user_text(p) for p in payloads]

            def _run_batch():
                return self._engine.generate_many(instructions, user_texts)

            texts = await asyncio.to_thread(_run_batch)
            return SimpleNamespace(
                output_texts=[t.strip() for t in texts],
                output_text=(texts[0].strip() if texts else ""),
            )

        # Single prompt
        payload: InputPayload = input  # type: ignore[assignment]
        user_text = _extract_user_text(payload)

        def _run_one():
            return self._engine.generate(instructions, user_text)

        text = await asyncio.to_thread(_run_one)
        return SimpleNamespace(output_text=text.strip())


class HFCompatClient:
    def __init__(self):
        self._provider = "hf"
        self.responses = _HFResponses(self._provider)


# -----------------------
# Gemini (placeholder)
# -----------------------
# Once you pick the exact Gemini SDK you installed, we can implement this similarly.
# class GeminiCompatClient:
#     ...


def get_llm_client():
    provider = (os.getenv("LLM_PROVIDER", "openai") or "openai").lower()

    if provider == "openai":
        return OpenAICompatClient()
    if provider in ("anthropic", "claude"):
        return AnthropicCompatClient()
    if provider in ("hf", "huggingface", "local"):
        return HFCompatClient()

    raise ValueError(f"Unsupported LLM_PROVIDER={provider}")
