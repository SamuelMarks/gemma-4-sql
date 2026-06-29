"""SDK interface for DPO (Direct Preference Optimization)."""

from __future__ import annotations


def run_dpo(model_name: str, dataset: str, backend: str = "pytorch", beta: float = 0.1) -> dict[str, object]:
    """Run Direct Preference Optimization (DPO).

    Args:
    ----
        model_name: Name of the model.
        dataset: Name of the dataset.
        backend: The execution backend ('jax', 'keras', 'maxtext', 'pytorch').
        beta: Temperature parameter for the DPO loss.

    Returns:
    -------
        A dict with execution status and metrics.

    Raises:
    ------
        ValueError: If an unknown backend is provided.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).run_dpo(model_name, dataset, beta)
