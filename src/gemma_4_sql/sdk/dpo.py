"""SDK interface for DPO (Direct Preference Optimization)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue


def run_dpo(
    model_name: str,
    dataset: str,
    backend: str = "pytorch",
    beta: float = 0.1,
    epochs: int = 1,
    learning_rate: float = 1e-05,
    batch_size: int = 2,
    **kwargs: JSONValue,
) -> JSONDict:
    """Run Direct Preference Optimization (DPO).

    Args:
        model_name: The name of the target model.
        dataset: The name or path of the dataset.
        backend: The backend framework to use.
        beta: The beta parameter controlling the KL penalty.
        epochs: Number of DPO training epochs.
        learning_rate: Learning rate for optimizer.
        batch_size: Batch size for DPO dataloader.
        **kwargs: Additional backend-specific parameters.

    Returns:
        A dictionary containing the results.
    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    DPOConfig = __import__("gemma_4_sql.type_hints", fromlist=["DPOConfig"]).DPOConfig
    test_mode = bool(kwargs.get("test_mode"))
    return get_backend(backend).run_dpo(
        DPOConfig(
            model_name=model_name,
            dataset=dataset,
            beta=beta,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            test_mode=test_mode,
        )
    )
