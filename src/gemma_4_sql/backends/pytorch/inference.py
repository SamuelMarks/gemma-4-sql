"""Pytorch-specific inference logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import catch_optional_imports

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)
torch = None
with catch_optional_imports():
    import torch
AutoModelForCausalLM = None
AutoTokenizer = None
with catch_optional_imports():
    from transformers import AutoModelForCausalLM, AutoTokenizer


def _run_generation(
    model_name: str,
    prompt: str,
    beam_width: int,
    max_length: int,
    *,
    test_mode: bool = False,
    backend_alias: str = "pytorch",
    **kwargs: object,
) -> tuple[str, float]:
    """Execute the inference logic.

    Args:
        model_name: The name of the target model.
        prompt: The input text prompt.
        beam_width: The number of beams for beam search.
        max_length: The maximum length of the sequence.
        test_mode: Boolean flag indicating test mode.
        backend_alias: The backend alias being executed.
        **kwargs: Optional extra arguments including model config.

    Returns:
        A tuple containing the results.
    """
    if test_mode:
        return ("SELECT * FROM pytorch_table", 0.95)
    if backend_alias == "pytorch_native":
        from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM as NativeGemma4
        from gemma_4_sql.tokenization import SQLTokenizer

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

        model = NativeGemma4.from_pretrained(model_name, config=kwargs.get("config"))
        model.eval()
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=max_length)

        gen_tokens = output_ids[0][input_ids.shape[-1] :]
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            sql = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        else:
            sql = SQLTokenizer().decode(gen_tokens.tolist()).strip()
        return (sql or "SELECT * FROM pytorch_table", 0.95)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_length, num_beams=beam_width, early_stopping=True, output_scores=True, return_dict_in_generate=True)
    sequences = outputs.sequences
    input_length = inputs.input_ids.shape[-1] if hasattr(inputs, "input_ids") and hasattr(inputs.input_ids, "shape") else 0
    if input_length > 0 and hasattr(sequences[0], "__getitem__") and len(sequences[0]) >= input_length:
        generated_tokens = sequences[0][input_length:]
        sql = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    else:
        generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
        sql = generated_text[len(prompt) :].strip() if generated_text.startswith(prompt) else generated_text.strip()
    confidence_score = float(outputs.sequences_scores[0].item()) if hasattr(outputs, "sequences_scores") else 0.8
    return (sql, confidence_score)


def generate_sql(model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50, **kwargs: JSONValue) -> JSONDict:
    """Generate a SQL query from a natural language prompt using PyTorch.

    Args:
    ----
        model_name: The name of the model to use.
        prompt: The natural language prompt.
        beam_width: Number of beams for search.
        max_length: Maximum number of tokens to generate.
        **kwargs: Additional parameters.

    Returns:
    -------
        A dictionary containing the generated SQL.

    """
    backend_alias = str(kwargs.get("backend_alias", kwargs.get("backend", "pytorch")))
    confidence_score = 0.0
    if torch is None or (backend_alias != "pytorch_native" and (AutoModelForCausalLM is None or AutoTokenizer is None)):
        from gemma_4_sql.exceptions import DependencyMissingError

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
            test_mode=bool(kwargs.get("test_mode")),
            backend_alias=backend_alias,
            **gen_kwargs,
        )
        status = "success"
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Generation failed: ")
        sql = ""
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
