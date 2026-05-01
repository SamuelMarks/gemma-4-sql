"""
SDK interface for DPO (Direct Preference Optimization).
"""

from __future__ import annotations

from typing import Any



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
    from gemma_4_sql.sdk.registry import get_backend
    return get_backend(backend).run_dpo(model_name, dataset, beta)
