"""Pytorch-specific inference logic."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, cast

from gemma_4_sql.exceptions import DependencyMissingError, InferenceError

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)

try:
    import torch as _torch
    from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
    from transformers import AutoTokenizer as _AutoTokenizer

    torch: Any = _torch
    AutoModelForCausalLM: Any = _AutoModelForCausalLM
    AutoTokenizer: Any = _AutoTokenizer
except (ImportError, AttributeError):
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


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
    if backend_alias == "pytorch_native":
        from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM as NativeGemma4
        from gemma_4_sql.tokenization import SQLTokenizer

        image_path = kwargs.get("image_path")
        audio_path = kwargs.get("audio_path")
        pixel_values = kwargs.get("pixel_values")
        audio_values = kwargs.get("audio_values")

        if image_path is not None or audio_path is not None:
            from gemma_4_sql.backends.common_multimodal import (
                format_multimodal_prompt,
                process_audio,
                process_image,
            )

            formatted = format_multimodal_prompt(
                prompt,
                has_image=image_path is not None or pixel_values is not None,
                has_audio=audio_path is not None or audio_values is not None,
            )
            prompt = formatted["prompt"]

            if image_path is not None and pixel_values is None:
                img_res = process_image(cast(Any, image_path))
                pixel_values = torch.tensor(img_res["pixel_values"], dtype=torch.float32).unsqueeze(0)

            if audio_path is not None and audio_values is None:
                aud_res = process_audio(cast(Any, audio_path))
                audio_values = torch.tensor(aud_res["audio_values"], dtype=torch.float32).unsqueeze(0)

        tokenizer = None
        if AutoTokenizer is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs.input_ids
            except (OSError, ValueError, RuntimeError, KeyError, AttributeError):
                tokenizer = None

        if tokenizer is None:
            sql_tok = SQLTokenizer()
            tokens = sql_tok.encode(prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long)

        model = NativeGemma4.from_pretrained(model_name, config=cast(Any, kwargs.get("config")))
        model.eval()
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
            sql = SQLTokenizer().decode(gen_tokens.tolist()).strip()

        if not sql:
            raise InferenceError("PyTorch native generation decoded into an empty SQL query string.")

        confidence_score = 0.95
        return (sql, confidence_score)

    image_path = kwargs.get("image_path")
    audio_path = kwargs.get("audio_path")
    pixel_values = kwargs.get("pixel_values")
    audio_values = kwargs.get("audio_values")

    if image_path is not None or audio_path is not None:
        from gemma_4_sql.backends.common_multimodal import (
            format_multimodal_prompt,
            process_audio,
            process_image,
        )

        formatted = format_multimodal_prompt(
            prompt,
            has_image=image_path is not None or pixel_values is not None,
            has_audio=audio_path is not None or audio_values is not None,
        )
        prompt = formatted["prompt"]

        if image_path is not None and pixel_values is None:
            img_res = process_image(cast(Any, image_path))
            pixel_values = torch.tensor(img_res["pixel_values"], dtype=torch.float32).unsqueeze(0)

        if audio_path is not None and audio_values is None:
            aud_res = process_audio(cast(Any, audio_path))
            audio_values = torch.tensor(aud_res["audio_values"], dtype=torch.float32).unsqueeze(0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    if "adapter_path" in kwargs or "lora_path" in kwargs:
        adapter_path = str(kwargs.get("adapter_path") or kwargs.get("lora_path"))
        try:
            peft_pkg = __import__("peft", fromlist=["PeftModel"])
            model = peft_pkg.PeftModel.from_pretrained(model, adapter_path)
        except (ImportError, ValueError, RuntimeError, OSError) as e:
            logger.warning("Could not load adapter from %s: %s", adapter_path, e)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    extra_gen: dict[str, Any] = {}
    if pixel_values is not None:
        extra_gen["pixel_values"] = pixel_values.to(model.device) if hasattr(pixel_values, "to") else pixel_values
    if audio_values is not None:
        extra_gen["audio_values"] = audio_values.to(model.device) if hasattr(audio_values, "to") else audio_values

    outputs = cast(Any, model).generate(
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
    **kwargs: JSONValue,
) -> JSONDict:
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
    confidence_score = 0.0
    if torch is None or (backend_alias != "pytorch_native" and (AutoModelForCausalLM is None or AutoTokenizer is None)):
        raise DependencyMissingError("PyTorch dependencies are missing.")

    try:
        logger.info("Generating with %s", model_name)
        gen_kwargs = dict(kwargs)
        gen_kwargs.pop("backend_alias", None)
        gen_kwargs.pop("backend", None)
        gen_kwargs.pop("test_mode", None)
        (sql, confidence_score) = _run_generation(
            model_name,
            prompt,
            beam_width,
            max_length,
            backend_alias=backend_alias,
            **gen_kwargs,
        )
        status = "success"
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Generation failed: ")
        sql = ""
        confidence_score = 0.0
        status = f"failed: {e!s}"

    return {
        "backend": backend_alias,
        "model": model_name,
        "prompt": prompt,
        "sql": sql,
        "status": status,
        "beam_width": beam_width,
        "confidence_score": confidence_score,
    }
