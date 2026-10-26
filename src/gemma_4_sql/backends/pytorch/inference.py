"""Pytorch-specific inference logic."""

from __future__ import annotations

import logging
import math
from typing import Any

from gemma_4_sql.backends.common_inference import (
    BoundedLRUCache,
    normalize_logprob_confidence,
    run_inference_wrapper,
)
from gemma_4_sql.backends.common_multimodal import prepare_multimodal_inputs
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError
from gemma_4_sql.type_hints import InferenceResultDict

logger = logging.getLogger(__name__)

torch = optional_import("torch")
AutoModelForCausalLM = optional_attr("transformers", "AutoModelForCausalLM")
AutoTokenizer = optional_attr("transformers", "AutoTokenizer")

_MODEL_CACHE: BoundedLRUCache[tuple[Any, Any]] = BoundedLRUCache(maxsize=4)


def _run_generation(
    model_name: str,
    prompt: str,
    beam_width: int,
    max_length: int,
    *,
    backend_alias: str = "pytorch",
    **kwargs: object,
) -> tuple[str, float]:
    """Execute the inference logic across native or Hugging Face PyTorch pipelines.

    Args:
        model_name: The name or path of the target model.
        prompt: The input natural language prompt.
        beam_width: The number of beams for beam search.
        max_length: The maximum length of new tokens to generate.
        backend_alias: The backend alias ('pytorch' or 'pytorch_native').
        **kwargs: Optional extra arguments including model config and adapter path.

    Returns:
        A tuple of (generated SQL query string, confidence score).

    Raises:
        InferenceError: If generation yields an empty sequence or SQL decoding fails.

    """

    def _to_torch_tensor(arr: Any) -> Any:
        """Convert feature array into a batched float32 PyTorch tensor."""
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

    (prompt, pixel_values, audio_values) = prepare_multimodal_inputs(
        prompt,
        dict(kwargs),
        _to_torch_tensor,
    )

    if backend_alias == "pytorch_native":
        from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM as NativeGemma4
        from gemma_4_sql.tokenization import SQLTokenizer

        tokenizer = None
        input_ids = None
        if AutoTokenizer is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs.input_ids
            except (OSError, ValueError, RuntimeError, KeyError, AttributeError):
                tokenizer = None

        if tokenizer is None or input_ids is None:
            sql_tok = SQLTokenizer()
            tokens = sql_tok.encode(prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long)

        cfg_in: Any = kwargs.get("config")
        native_cache_key = f"native:{model_name}:{id(cfg_in)}:{id(getattr(NativeGemma4, 'from_pretrained', None))}"
        cached_native = _MODEL_CACHE.get_model(native_cache_key)
        model = cached_native[0] if cached_native is not None else NativeGemma4.from_pretrained(model_name, config=cfg_in)
        model.eval()
        _MODEL_CACHE.put_model(native_cache_key, (model, tokenizer))
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_length}
        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values
        if audio_values is not None:
            gen_kwargs["audio_values"] = audio_values

        with torch.no_grad():
            output_ids = model.generate(input_ids, **gen_kwargs)

        gen_tokens = output_ids[0][input_ids.shape[-1] :]
        if len(gen_tokens) == 0:
            raise InferenceError("PyTorch native generation yielded an empty sequence.")

        if tokenizer is not None and hasattr(tokenizer, "decode"):
            sql = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        else:
            tok_list: list[int] = [int(t) for t in gen_tokens.tolist()]
            sql = SQLTokenizer().decode(tok_list).strip()

        if not sql:
            raise InferenceError("PyTorch native generation decoded into an empty SQL query string.")

        with torch.no_grad():
            fwd_out: Any = model(output_ids)
            logits: Any = fwd_out[0] if isinstance(fwd_out, tuple) else getattr(fwd_out, "logits", fwd_out)
            prompt_len = int(input_ids.shape[-1])
            step_logits = logits[0, prompt_len - 1 : -1, :]
            log_probs = torch.nn.functional.log_softmax(step_logits, dim=-1)
            selected_log_probs = log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)
            logprob_sum = float(selected_log_probs.sum().item())
        confidence_score = normalize_logprob_confidence(logprob_sum, len(gen_tokens))
        return (sql, confidence_score)

    adapter_key = str(kwargs.get("adapter_path") or kwargs.get("lora_path") or "")
    hf_cache_key = f"hf:{model_name}:{adapter_key}:{id(getattr(AutoTokenizer, 'from_pretrained', None))}:{id(getattr(AutoModelForCausalLM, 'from_pretrained', None))}"
    cached_hf = _MODEL_CACHE.get_model(hf_cache_key)
    if cached_hf is not None and not adapter_key:
        model, tokenizer = cached_hf
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        if "adapter_path" in kwargs or "lora_path" in kwargs:
            adapter_path = str(kwargs.get("adapter_path") or kwargs.get("lora_path"))
            try:
                peft_model_cls = optional_attr("peft", "PeftModel")
                if peft_model_cls is None:
                    raise ImportError("peft.PeftModel is not available.")
                model = peft_model_cls.from_pretrained(model, adapter_path)
            except (ImportError, ValueError, RuntimeError, OSError) as e:
                logger.warning("Could not load adapter from %s: %s", adapter_path, e)
        _MODEL_CACHE.put_model(hf_cache_key, (model, tokenizer))

    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(getattr(model, "device", "cpu"))
    extra_gen: dict[str, Any] = {}
    pv_any: Any = pixel_values
    av_any: Any = audio_values
    if pv_any is not None:
        extra_gen["pixel_values"] = pv_any.to(getattr(model, "device", "cpu")) if hasattr(pv_any, "to") else pv_any
    if av_any is not None:
        extra_gen["audio_values"] = av_any.to(getattr(model, "device", "cpu")) if hasattr(av_any, "to") else av_any

    outputs = model.generate(
        **inputs,
        **extra_gen,
        max_new_tokens=max_length,
        num_beams=beam_width,
        early_stopping=True,
        output_scores=True,
        return_dict_in_generate=True,
    )
    sequences = outputs.sequences
    input_length = inputs.input_ids.shape[-1] if hasattr(inputs, "input_ids") and hasattr(inputs.input_ids, "shape") else 0
    if input_length > 0 and hasattr(sequences[0], "__getitem__") and len(sequences[0]) >= input_length:
        generated_tokens = sequences[0][input_length:]
        sql = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    else:
        generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
        sql = generated_text[len(prompt) :].strip() if generated_text.startswith(prompt) else generated_text.strip()

    if not sql:
        raise InferenceError("PyTorch generation yielded an empty SQL sequence.")

    raw_conf = float(outputs.sequences_scores[0].item()) if hasattr(outputs, "sequences_scores") else 0.8
    if raw_conf <= 0.0:
        confidence_score = max(0.0, min(1.0, math.exp(raw_conf)))
    else:
        confidence_score = max(0.0, min(1.0, raw_conf))

    return (sql, confidence_score)


def generate_sql(
    model_name: str,
    prompt: str,
    beam_width: int = 3,
    max_length: int = 50,
    **kwargs: object,
) -> InferenceResultDict:
    """Generate a SQL query from a natural language prompt using PyTorch.

    Args:
        model_name: The name or path of the model to use.
        prompt: The natural language prompt.
        beam_width: Number of beams for search.
        max_length: Maximum number of tokens to generate.
        **kwargs: Additional parameters.

    Returns:
        A dictionary containing the generated SQL and metadata.

    Raises:
        DependencyMissingError: If PyTorch dependencies are missing.

    """
    backend_alias = str(kwargs.get("backend_alias", kwargs.get("backend", "pytorch")))
    if torch is None or (backend_alias != "pytorch_native" and (AutoModelForCausalLM is None or AutoTokenizer is None)):
        raise DependencyMissingError("PyTorch dependencies are missing.")

    gen_kwargs = dict(kwargs)
    gen_kwargs.pop("backend_alias", None)
    gen_kwargs.pop("backend", None)

    def _execute() -> tuple[str, float]:
        """Run PyTorch native or HuggingFace generation."""
        logger.info("Generating with %s", model_name)
        return _run_generation(
            model_name,
            prompt,
            beam_width,
            max_length,
            backend_alias=backend_alias,
            **gen_kwargs,
        )

    return run_inference_wrapper(backend_alias, model_name, prompt, beam_width, _execute, **kwargs)
