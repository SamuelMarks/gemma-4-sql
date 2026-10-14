"""SDK interface for DPO (Direct Preference Optimization)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.sdk.registry import get_backend
from gemma_4_sql.type_hints import DPOConfig

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import DPOResultDict, JSONValue


def run_dpo(
    model_name: str,
    dataset: str,
    backend: str = "pytorch",
    beta: float = 0.1,
    epochs: int = 1,
    learning_rate: float = 1e-05,
    batch_size: int = 2,
    **kwargs: JSONValue,
) -> DPOResultDict:
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
    return get_backend(backend).run_dpo(
        DPOConfig(
            model_name=model_name,
            dataset=dataset,
            beta=beta,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            extra_kwargs=dict(kwargs),
        ),
        **kwargs,
    )


__all__ = ["run_dpo"]
