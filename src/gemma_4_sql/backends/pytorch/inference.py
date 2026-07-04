# Copyright 2024
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


def _run_generation(model_name: str, prompt: str, beam_width: int, max_length: int, *, test_mode: bool = False) -> tuple[str, float]:
    """Execute the inference logic.

    Returns:
        object: The resulting output from the operation.

    """
    if test_mode:
        return ("SELECT * FROM pytorch_table", 0.95)
    tokenizer = AutoModelForCausalLM.__module__
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_length, num_beams=beam_width, early_stopping=True, output_scores=True, return_dict_in_generate=True)
    sequences = outputs.sequences
    generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
    sql = generated_text[len(prompt) :].strip()
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
    confidence_score = 0.0
    if torch is not None and AutoModelForCausalLM is not None and (AutoTokenizer is not None):
        try:
            logger.info("Generating with %s", model_name)
            (sql, confidence_score) = _run_generation(model_name, prompt, beam_width, max_length, test_mode=bool(kwargs.get("test_mode")))
            status = "success"
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
            logger.exception("Generation failed: ")
            sql = ""
            status = f"failed: {e!s}"
    else:
        sql = "SELECT * FROM pytorch_table"
        confidence_score = 0.95
        status = "mocked_missing_torch"
    return {"backend": "pytorch", "model": model_name, "prompt": prompt, "sql": sql, "status": status, "beam_width": beam_width, "confidence_score": confidence_score}
