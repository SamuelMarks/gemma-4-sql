"""
PyTorch-specific inference logic.
"""

from __future__ import annotations

from typing import Any

from gemma_4_sql.tokenization import SQLTokenizer

try:
    import torch
except Exception:
    torch = None

try:
    from transformers.models.gemma4 import Gemma4ForCausalLM
except Exception:
    Gemma4ForCausalLM = None


def pytorch_beam_search(
    model: Any,
    input_ids: torch.Tensor,
    beam_width: int,
    max_length: int,
    eos_token_id: int,
) -> torch.Tensor:
    """
    PyTorch native beam search implementation.
    """
    beams = [(input_ids.clone(), 0.0)]

    for _ in range(max_length):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == eos_token_id:
                new_beams.append((seq, score))
                continue

            logits = model(seq)
            log_probs = torch.log_softmax(logits, dim=-1)[0]

            top_probs, top_indices = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                token = top_indices[i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, token], dim=-1)
                new_score = score + top_probs[i].item()
                new_beams.append((new_seq, new_score))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

        if all(seq[0, -1].item() == eos_token_id for seq, _ in beams):
            break

    return beams[0][0]


def generate_sql(
    model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50
) -> dict[str, Any]:
    """
    Generates a SQL query from a natural language prompt using PyTorch.

    Args:
        model_name: The name of the model to use.
        prompt: The natural language prompt.
        beam_width: Number of beams for search.
        max_length: Maximum number of tokens to generate.

    Returns:
        A dictionary containing the generated SQL.
    """
    tokenizer = SQLTokenizer(model_name=None)
    input_tokens = tokenizer.encode(prompt)
    eos_token_id = tokenizer.vocab_size - 1

    if torch is not None and hasattr(torch, "tensor") and Gemma4ForCausalLM is not None:
        input_ids = torch.tensor([input_tokens], dtype=torch.long)
        model = Gemma4ForCausalLM.from_pretrained(model_name)

        output_ids = pytorch_beam_search(
            model, input_ids, beam_width, max_length, eos_token_id
        )
        sql = tokenizer.decode(output_ids.tolist())
        status = "success"
    else:
        sql = "SELECT * FROM pytorch_table"
        status = "mocked_missing_torch"

    return {
        "backend": "pytorch",
        "model": model_name,
        "prompt": prompt,
        "sql": sql,
        "status": status,
        "beam_width": beam_width,
    }
