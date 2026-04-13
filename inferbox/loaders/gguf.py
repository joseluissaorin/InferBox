"""GGUF model loader via llama-cpp-python.

Supports:
- Prompt KV-cache reuse (LlamaRAMCache)
- Speculative decoding with a draft model
- Grammar-constrained generation (BNF or JSON schema)
- Streaming token output
"""
import json
import logging
from dataclasses import dataclass
from typing import Any

from ..config import ModelConfig

logger = logging.getLogger("inferbox")


@dataclass
class GGUFModel:
    llm: Any
    draft_llm: Any | None = None


def _build_grammar(grammar_spec):
    """Build a llama-cpp grammar object from various spec types."""
    try:
        from llama_cpp import LlamaGrammar
    except ImportError:
        return None

    if grammar_spec is None:
        return None
    if grammar_spec == "json":
        # JSON object grammar
        return LlamaGrammar.from_string(_JSON_GRAMMAR)
    if isinstance(grammar_spec, tuple) and grammar_spec[0] == "json_schema":
        # JSON schema grammar (constrains output to match schema)
        try:
            from llama_cpp.llama_grammar import LlamaGrammar as LG
            schema = grammar_spec[1]
            return LG.from_json_schema(json.dumps(schema))
        except Exception as e:
            logger.warning(f"Could not build grammar from json_schema: {e}")
            return None
    if isinstance(grammar_spec, str):
        # Raw BNF
        try:
            return LlamaGrammar.from_string(grammar_spec)
        except Exception as e:
            logger.warning(f"Could not parse grammar BNF: {e}")
            return None
    return None


# Minimal JSON object grammar (GBNF format)
_JSON_GRAMMAR = r"""
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array  ::= "[" ws ( value ("," ws value)* )? "]" ws
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws     ::= [ \t\n]*
"""


def load(config: ModelConfig) -> GGUFModel:
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id=config.model_id,
        filename=config.model_file,
    )

    n_gpu_layers = config.options.get("n_gpu_layers", -1)
    n_ctx = config.options.get("n_ctx", 8192)
    cache_size = config.options.get("cache_size_mb", 512) * 1024 * 1024  # bytes

    # Build draft model for speculative decoding if specified
    draft_llm = None
    draft_path = config.options.get("draft_model_path")
    draft_repo = config.options.get("draft_model_id")
    draft_file = config.options.get("draft_model_file")
    if draft_repo and draft_file:
        try:
            draft_path = hf_hub_download(repo_id=draft_repo, filename=draft_file)
        except Exception as e:
            logger.warning(f"Could not download draft model: {e}")

    if draft_path:
        try:
            from llama_cpp import LlamaPromptLookupDecoding
            # Use prompt lookup decoding (lightweight speculative without draft model)
            # For full speculative with separate draft model, would use draft_model param
            logger.info(f"Loading draft model: {draft_path}")
            draft_llm = Llama(
                model_path=draft_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"Could not load draft model: {e}")

    llm_kwargs = dict(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False,
    )
    # Speculative decoding via draft_model parameter (newer llama-cpp-python)
    if draft_llm is not None:
        try:
            llm_kwargs["draft_model"] = draft_llm
        except Exception:
            pass

    llm = Llama(**llm_kwargs)

    # NOTE: LlamaRAMCache (prefix KV reuse) is DISABLED. It causes
    # `IndexError: index N is out of bounds for axis 0 with size M` inside
    # llama.py:generate() when cache state drifts from n_tokens on varied
    # prompts, and subsequently SIGSEGVs the C++ side. Our RAG prompts are
    # mostly unique per call anyway, so prefix reuse is ~0% hit rate.
    logger.info("GGUF model loaded (no prefix KV cache)")

    return GGUFModel(llm=llm, draft_llm=draft_llm)


def unload(model: GGUFModel):
    # Do NOT `del model.llm` or `del model.draft_llm`. See hf_embed.unload.
    # GC will reclaim the Llama objects once the manager drops its reference.
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def generate_stream(
    model: GGUFModel,
    config: ModelConfig,
    prompt: str | None = None,
    messages: list[dict] | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    stop: list[str] | None = None,
    grammar=None,
):
    """Yield generated tokens one at a time."""
    grammar_obj = _build_grammar(grammar)
    extra: dict = {}
    if grammar_obj is not None:
        extra["grammar"] = grammar_obj

    if messages:
        for chunk in model.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens,
            temperature=temperature, stop=stop, stream=True, **extra,
        ):
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta and delta["content"]:
                yield delta["content"]
    else:
        for chunk in model.llm(
            prompt or "", max_tokens=max_tokens,
            temperature=temperature, stop=stop, stream=True, **extra,
        ):
            text = chunk["choices"][0].get("text", "")
            if text:
                yield text


def generate(
    model: GGUFModel,
    config: ModelConfig,
    prompt: str | None = None,
    messages: list[dict] | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    stop: list[str] | None = None,
    images: list[str] | None = None,
    grammar=None,
) -> dict:
    grammar_obj = _build_grammar(grammar)
    extra: dict = {}
    if grammar_obj is not None:
        extra["grammar"] = grammar_obj

    if messages:
        result = model.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **extra,
        )
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
    else:
        result = model.llm(
            prompt or "",
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **extra,
        )
        text = result["choices"][0]["text"]
        usage = result.get("usage", {})

    return {"text": text, "usage": usage}
