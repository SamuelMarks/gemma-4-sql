"""SDK interface for DPO (Direct Preference Optimization)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict


def run_dpo(model_name: str, dataset: str, backend: str = "pytorch", beta: float = 0.1) -> JSONDict:
    """Run Direct Preference Optimization (DPO).

    Args:
        model_name: The name of the target model.
        dataset: The name or path of the dataset.
        backend: The backend framework to use.
        beta: The beta parameter controlling the KL penalty.

    Returns:
        A dictionary containing the results.
    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    DPOConfig = __import__("gemma_4_sql.type_hints", fromlist=["DPOConfig"]).DPOConfig
    return get_backend(backend).run_dpo(DPOConfig(model_name=model_name, dataset=dataset, beta=beta))
