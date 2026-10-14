"""Common training epoch iteration utilities across backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

T_Batch = TypeVar("T_Batch")


def generic_run_training_epochs(
    epochs: int,
    dataloader: Iterable[T_Batch],
    process_batch_fn: Callable[[T_Batch], float],
) -> float:
    """Run a generic supervised training loop across epochs and batches.

    Args:
        epochs: Number of full passes over the dataset.
        dataloader: Iterable yielding training batches.
        process_batch_fn: Callback executing one optimization step and returning scalar loss.

    Returns:
        Mean batch loss from the final training epoch.

    """
    final_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            loss = process_batch_fn(batch)
            epoch_loss += loss
            num_batches += 1
        final_loss = epoch_loss / max(1, num_batches)
    return final_loss
