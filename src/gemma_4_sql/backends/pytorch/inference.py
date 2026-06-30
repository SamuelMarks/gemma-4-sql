"""Pytorch-specific inference logic."""

from __future__ import annotations

import logging

from gemma_4_sql.tokenization import SQLTokenizer

logger = logging.getLogger(__name__)

try:
    import torch
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    torch = None
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    AutoModelForCausalLM = None
    AutoTokenizer = None


def generate_sql(model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50, **kwargs: object) -> dict[str, object]:
    """Generate a SQL query from a natural language prompt using PyTorch.

    Args:
    ----
        model_name: The name of the model to use.
        prompt: The natural language prompt.
        beam_width: Number of beams for search.
        max_length: Maximum number of tokens to generate.

    Returns:
    -------
        A dictionary containing the generated SQL.

    """
    if torch is not None and AutoModelForCausalLM is not None and AutoTokenizer is not None:
        try:
            # Note: in a real implementation we might pass the actual model and tokenizer instances
            # instead of loading from disk every time.
            logger.info("Generating with %s", model_name)

            # Using custom SQLTokenizer just to mock the basic behavior here if test_mode
            if kwargs.get("test_mode"):
                tokenizer = SQLTokenizer(model_name=None)
                sql = "SELECT * FROM pytorch_table"
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                outputs = model.generate(**inputs, max_new_tokens=max_length, num_beams=beam_width, early_stopping=True)
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                sql = generated_text[len(prompt) :].strip()

            status = "success"
        except Exception as e:
            logger.exception("Generation failed: %s", e)
            sql = ""
            status = f"failed: {e!s}"
    else:
        sql = "SELECT * FROM pytorch_table"
        status = "mocked_missing_torch"

    return {"backend": "pytorch", "model": model_name, "prompt": prompt, "sql": sql, "status": status, "beam_width": beam_width}
