# adapters.py
from __future__ import annotations

import os
import asyncio
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

SENTINEL_MODEL = "concepts-default"

HF_MODELS_MAP: Dict[str, str] = {
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-32B-Instruct",
}

# OpenAI-style payloads
InputPayload = List[Dict[str, Any]]
BatchInputPayload = List[InputPayload]


# -----------------------
# Helpers
# -----------------------
def _extract_user_text(input_payload: InputPayload) -> str:
    """
    Expected:
      [{"role":"user","content":[{"type":"input_text","text":"..."}]}]
    """
    try:
        if not input_payload:
            return ""
        first = input_payload[0] if isinstance(input_payload[0], dict) else {}
        content = first.get("content", [])
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict):
                if "text" in p:
                    parts.append(str(p["text"]))
                elif "content" in p:
                    parts.append(str(p["content"]))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts).strip()
    except Exception:
        return str(input_payload).strip()


def _resolve_model(provider: str, requested_model: str) -> str:
    # If caller passed a real model, use it
    if (requested_model or "").strip() and requested_model != SENTINEL_MODEL:
        return requested_model

    provider = (provider or "").lower().strip()

    if provider == "openai":
        return os.getenv("OPENAI_CONCEPTS_MODEL", "gpt-5-mini-2025-08-07")
    if provider in ("anthropic", "claude"):
        return os.getenv("ANTHROPIC_CONCEPTS_MODEL", "claude-3-5-sonnet-latest")
    if provider == "gemini":
        return os.getenv("GEMINI_CONCEPTS_MODEL", "gemini-1.5-pro")

    # HF/local/vllm
    if provider in ("hf", "huggingface", "local", "vllm"):
        return os.getenv("HF_CONCEPTS_MODEL", os.getenv("LLM_MODEL", "qwen7b"))

    return requested_model or SENTINEL_MODEL


def _is_batch_input(inp: Any) -> bool:
    # batch looks like: [payload1, payload2, ...] where each payload is a list[dict]
    return bool(inp) and isinstance(inp, list) and isinstance(inp[0], list)


# -----------------------
# OpenAI (compatible)
# -----------------------
class _OpenAIResponses:
    def __init__(self, client, provider_name: str):
        self._client = client
        self._provider = provider_name

    async def create(
        self,
        *,
        model: str,
        instructions: str,
        input: Union[InputPayload, BatchInputPayload],
        **kwargs,
    ):
        model = _resolve_model(self._provider, model)

        # If a batch is passed, run per-item (keeps your interface stable)
        if _is_batch_input(input):
            payloads: BatchInputPayload = input  # type: ignore[assignment]
            outs: List[str] = []

            async def _one(payload: InputPayload) -> str:
                r = await self._client.responses.create(
                    model=model,
                    instructions=instructions,
                    input=payload,
                    **kwargs,
                )
                t = getattr(r, "output_text", None)
                return (t.strip() if isinstance(t, str) else str(r).strip())

            outs = await asyncio.gather(*[_one(p) for p in payloads])
            return SimpleNamespace(output_texts=outs, output_text=(outs[0] if outs else ""))

        # Single
        return await self._client.responses.create(
            model=model,
            instructions=instructions,
            input=input,  # type: ignore[arg-type]
            **kwargs,
        )


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

    async def _one(self, *, model: str, instructions: str, payload: InputPayload) -> str:
        user_text = _extract_user_text(payload)
        max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "800"))

        msg = await self._client.messages.create(
            model=model,
            system=instructions,
            messages=[{"role": "user", "content": user_text}],
            max_tokens=max_tokens,
        )

        # flatten blocks -> str
        text = ""
        try:
            for block in msg.content:
                t = getattr(block, "text", None)
                if t:
                    text += t
        except Exception:
            text = str(msg)

        return text.strip()

    async def create(
        self,
        *,
        model: str,
        instructions: str,
        input: Union[InputPayload, BatchInputPayload],
        **kwargs,
    ):
        model = _resolve_model(self._provider, model)

        if _is_batch_input(input):
            payloads: BatchInputPayload = input  # type: ignore[assignment]
            outs = await asyncio.gather(*[self._one(model=model, instructions=instructions, payload=p) for p in payloads])
            return SimpleNamespace(output_texts=outs, output_text=(outs[0] if outs else ""))

        text = await self._one(model=model, instructions=instructions, payload=input)  # type: ignore[arg-type]
        return SimpleNamespace(output_text=text)


class AnthropicCompatClient:
    def __init__(self):
        from anthropic import AsyncAnthropic

        self._provider = "anthropic"
        self._client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.responses = _AnthropicResponses(self._client, self._provider)


# -----------------------
# vLLM Local Engine (NEW - replaces HuggingFace transformers)
# -----------------------
class _VLLMLocalEngine:
    """
    vLLM-based inference engine for fast local inference.
    """
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.llm = None
        self.tokenizer = None
        self._load_lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        with self._load_lock:
            if self._loaded:
                return

            from vllm import LLM
            from transformers import AutoTokenizer

            hf_token = os.getenv("HF_TOKEN")
            trust_remote_code = os.getenv("HF_TRUST_REMOTE_CODE", "1") == "1"
            
            # vLLM settings
            gpu_memory_utilization = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
            max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
            
            print(f"[vLLM] Loading model: {self.model_id}")
            print(f"[vLLM] GPU memory utilization: {gpu_memory_utilization}")
            print(f"[vLLM] Max model length: {max_model_len}")
            
            self.llm = LLM(
                model=self.model_id,
                trust_remote_code=trust_remote_code,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=True,  # Disable CUDA graphs to avoid async issues
            )
            
            # Load tokenizer for chat template
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=hf_token,
                trust_remote_code=trust_remote_code,
            )
            
            self._loaded = True
            print(f"[vLLM] Model loaded successfully!")

    def format_prompt(self, system: str, user: str) -> str:
        """Format prompt using chat template."""
        self.load()
        assert self.tokenizer is not None

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                msgs = [
                    {"role": "system", "content": system or ""},
                    {"role": "user", "content": user or ""},
                ]
                return self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Fallback format
        return (
            f"<|system|>\n{system}\n<|end|>\n"
            f"<|user|>\n{user}\n<|end|>\n<|assistant|>\n"
        )

    def generate_many(self, system: str, users: List[str]) -> List[str]:
        """Generate responses for multiple user inputs."""
        self.load()
        assert self.llm is not None

        from vllm import SamplingParams

        # Format all prompts
        prompts = [self.format_prompt(system, u) for u in users]

        # Sampling parameters from env
        max_new_tokens = int(os.getenv("VLLM_MAX_NEW_TOKENS", os.getenv("HF_MAX_NEW_TOKENS", "512")))
        temperature = float(os.getenv("VLLM_TEMPERATURE", os.getenv("HF_TEMPERATURE", "0.1")))
        top_p = float(os.getenv("VLLM_TOP_P", os.getenv("HF_TOP_P", "1.0")))

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Generate (vLLM handles batching internally)
        try:
            outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        except Exception as e:
            print(f"[vLLM] Generate failed: {e}")
            return [""] * len(users)

        # Extract generated text
        results: List[str] = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            results.append(generated_text)

        return results

    def generate(self, system: str, user: str) -> str:
        """Generate response for single user input."""
        outs = self.generate_many(system, [user])
        return outs[0] if outs else ""


class _VLLMResponses:
    def __init__(self, provider_name: str):
        self._provider = provider_name
        self._engine: Optional[_VLLMLocalEngine] = None
        self._model_id: Optional[str] = None

    def _ensure_engine(self, requested_model: str) -> None:
        resolved = _resolve_model(self._provider, requested_model)
        model_id = HF_MODELS_MAP.get(resolved, resolved)

        if self._engine is None or self._model_id != model_id:
            self._model_id = model_id
            self._engine = _VLLMLocalEngine(model_id)

    async def create(
        self,
        *,
        model: str,
        instructions: str,
        input: Union[InputPayload, BatchInputPayload],
        **kwargs,
    ):
        self._ensure_engine(model)
        assert self._engine is not None

        # Optional escape hatch: text_inputs=[...]
        text_inputs = kwargs.pop("text_inputs", None)
        if text_inputs is not None:
            if isinstance(text_inputs, str):
                user_texts = [text_inputs]
            elif isinstance(text_inputs, list):
                user_texts = [str(x) for x in text_inputs]
            else:
                user_texts = [str(text_inputs)]

            def _run_text_inputs() -> List[str]:
                return self._engine.generate_many(instructions, user_texts)

            texts = await asyncio.to_thread(_run_text_inputs)
            texts = [t.strip() for t in texts]
            return SimpleNamespace(output_texts=texts, output_text=(texts[0] if texts else ""))

        # Batch
        if _is_batch_input(input):
            payloads: BatchInputPayload = input  # type: ignore[assignment]
            user_texts = [_extract_user_text(p) for p in payloads]

            def _run_batch() -> List[str]:
                return self._engine.generate_many(instructions, user_texts)

            texts = await asyncio.to_thread(_run_batch)
            texts = [t.strip() for t in texts]
            return SimpleNamespace(output_texts=texts, output_text=(texts[0] if texts else ""))

        # Single
        payload: InputPayload = input  # type: ignore[assignment]
        user_text = _extract_user_text(payload)

        def _run_one() -> str:
            return self._engine.generate(instructions, user_text)

        text = await asyncio.to_thread(_run_one)
        return SimpleNamespace(output_text=str(text).strip())


class VLLMCompatClient:
    def __init__(self):
        self._provider = "vllm"
        self.responses = _VLLMResponses(self._provider)


# -----------------------
# Legacy HuggingFace (kept for backwards compatibility)
# -----------------------
class _HFLocalEngine:
    """
    Legacy HuggingFace transformers pipeline.
    Use VLLMCompatClient for faster inference.
    """
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.pipe = None
        self.tok = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()

    def load(self) -> None:
        if self.pipe is not None:
            return

        with self._load_lock:
            if self.pipe is not None:
                return

            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch

            hf_token = os.getenv("HF_TOKEN")
            trust_remote_code = os.getenv("HF_TRUST_REMOTE_CODE", "0") == "1"

            self.tok = AutoTokenizer.from_pretrained(
                self.model_id,
                use_fast=True,
                token=hf_token,
                trust_remote_code=trust_remote_code,
            )

            try:
                self.tok.padding_side = "left"
            except Exception:
                pass

            if getattr(self.tok, "pad_token_id", None) is None and getattr(self.tok, "eos_token_id", None) is not None:
                try:
                    self.tok.pad_token = self.tok.eos_token
                except Exception:
                    pass

            force_cpu = os.getenv("HF_FORCE_CPU", "0") == "1"
            has_cuda = bool(getattr(torch, "cuda", None)) and torch.cuda.is_available()
            use_cpu = force_cpu or (not has_cuda)

            if use_cpu:
                mdl = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype="auto",
                    token=hf_token,
                    trust_remote_code=trust_remote_code,
                )
                self.pipe = pipeline(
                    "text-generation",
                    model=mdl,
                    tokenizer=self.tok,
                    device=-1,
                    pad_token_id=getattr(self.tok, "eos_token_id", None),
                )
            else:
                device_map = os.getenv("HF_DEVICE_MAP", "auto")
                mdl = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map=device_map,
                    torch_dtype="auto",
                    token=hf_token,
                    trust_remote_code=trust_remote_code,
                )
                self.pipe = pipeline(
                    "text-generation",
                    model=mdl,
                    tokenizer=self.tok,
                    pad_token_id=getattr(self.tok, "eos_token_id", None),
                )

    def format_prompt(self, system: str, user: str) -> str:
        self.load()
        assert self.tok is not None

        if hasattr(self.tok, "apply_chat_template"):
            try:
                msgs = [
                    {"role": "system", "content": system or ""},
                    {"role": "user", "content": user or ""},
                ]
                return self.tok.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        return (
            f"<|system|>\n{system}\n<|end|>\n"
            f"<|user|>\n{user}\n<|end|>\n<|assistant|>\n"
        )

    def _pipe_call(self, prompts: Union[str, List[str]], **gen_kwargs) -> Any:
        assert self.pipe is not None
        try:
            return self.pipe(prompts, return_full_text=False, **gen_kwargs)
        except TypeError:
            return self.pipe(prompts, **gen_kwargs)

    def _normalize_outputs(self, outputs: Any) -> List[str]:
        if outputs is None:
            return []

        if isinstance(outputs, list):
            out: List[str] = []
            for item in outputs:
                if isinstance(item, list) and item:
                    item = item[0]
                if isinstance(item, dict) and "generated_text" in item:
                    out.append(str(item["generated_text"]).strip())
                else:
                    out.append(str(item).strip())
            return out

        if isinstance(outputs, dict) and "generated_text" in outputs:
            return [str(outputs["generated_text"]).strip()]

        return [str(outputs).strip()]

    def generate_many(self, system: str, users: List[str]) -> List[str]:
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

        return self._normalize_outputs(outputs)

    def generate(self, system: str, user: str) -> str:
        outs = self.generate_many(system, [user])
        return outs[0] if outs else ""


class _HFResponses:
    def __init__(self, provider_name: str):
        self._provider = provider_name
        self._engine: Optional[_HFLocalEngine] = None
        self._model_id: Optional[str] = None

    def _ensure_engine(self, requested_model: str) -> None:
        resolved = _resolve_model(self._provider, requested_model)
        model_id = HF_MODELS_MAP.get(resolved, resolved)

        if self._engine is None or self._model_id != model_id:
            self._model_id = model_id
            self._engine = _HFLocalEngine(model_id)

    async def create(
        self,
        *,
        model: str,
        instructions: str,
        input: Union[InputPayload, BatchInputPayload],
        **kwargs,
    ):
        self._ensure_engine(model)
        assert self._engine is not None

        text_inputs = kwargs.pop("text_inputs", None)
        if text_inputs is not None:
            if isinstance(text_inputs, str):
                user_texts = [text_inputs]
            elif isinstance(text_inputs, list):
                user_texts = [str(x) for x in text_inputs]
            else:
                user_texts = [str(text_inputs)]

            def _run_text_inputs() -> List[str]:
                return self._engine.generate_many(instructions, user_texts)

            texts = await asyncio.to_thread(_run_text_inputs)
            texts = [t.strip() for t in texts]
            return SimpleNamespace(output_texts=texts, output_text=(texts[0] if texts else ""))

        if _is_batch_input(input):
            payloads: BatchInputPayload = input
            user_texts = [_extract_user_text(p) for p in payloads]

            def _run_batch() -> List[str]:
                return self._engine.generate_many(instructions, user_texts)

            texts = await asyncio.to_thread(_run_batch)
            texts = [t.strip() for t in texts]
            return SimpleNamespace(output_texts=texts, output_text=(texts[0] if texts else ""))

        payload: InputPayload = input
        user_text = _extract_user_text(payload)

        def _run_one() -> str:
            return self._engine.generate(instructions, user_text)

        text = await asyncio.to_thread(_run_one)
        return SimpleNamespace(output_text=str(text).strip())


class HFCompatClient:
    def __init__(self):
        self._provider = "hf"
        self.responses = _HFResponses(self._provider)


# -----------------------
# Client factory
# -----------------------
def get_llm_client():
    provider = (os.getenv("LLM_PROVIDER", "openai") or "openai").lower().strip()

    if provider == "openai":
        return OpenAICompatClient()
    if provider in ("anthropic", "claude"):
        return AnthropicCompatClient()
    if provider == "vllm":
        return VLLMCompatClient()
    if provider in ("hf", "huggingface", "local"):
        # Default to vLLM for hf/local providers (faster)
        # Set LLM_PROVIDER=hf_legacy to use old transformers pipeline
        return VLLMCompatClient()
    if provider == "hf_legacy":
        return HFCompatClient()

    raise ValueError(f"Unsupported LLM_PROVIDER={provider}")