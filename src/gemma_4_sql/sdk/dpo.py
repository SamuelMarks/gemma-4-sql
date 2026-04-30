"""
SDK interface for DPO (Direct Preference Optimization).
"""

from __future__ import annotations

from typing import Any

from gemma_4_sql.backends.jax.dpo import run_dpo as run_dpo_jax
from gemma_4_sql.backends.keras.dpo import run_dpo as run_dpo_keras
from gemma_4_sql.backends.maxtext.dpo import run_dpo as run_dpo_maxtext
from gemma_4_sql.backends.pytorch.dpo import run_dpo as run_dpo_pytorch

def run_dpo(
    model_name: str, dataset: str, backend: str = "pytorch", beta: float = 0.1
) -> dict[str, Any]:
    """
    Runs Direct Preference Optimization (DPO).

    Args:
        model_name: Name of the model.
        dataset: Name of the dataset.
        backend: The execution backend ('jax', 'keras', 'maxtext', 'pytorch').
        beta: Temperature parameter for the DPO loss.

    Returns:
        A dict with execution status and metrics.

    Raises:
        ValueError: If an unknown backend is provided.
    """
    if backend == "pytorch":
        return run_dpo_pytorch(model_name, dataset, beta)
    elif backend == "jax":
        return run_dpo_jax(model_name, dataset, beta)
    elif backend == "keras":
        return run_dpo_keras(model_name, dataset, beta)
    elif backend == "maxtext":
        return run_dpo_maxtext(model_name, dataset, beta)
    else:
        raise ValueError(f"Unknown backend: {backend}")
