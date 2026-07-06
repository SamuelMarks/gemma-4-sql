"""Provide module docstring."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

T_Data = TypeVar("T_Data")
T_Batch = TypeVar("T_Batch")

if TYPE_CHECKING:
    from collections.abc import Callable


def generic_run_training_epochs(epochs: int, dataloader: T_Data, process_batch_fn: Callable[[T_Batch], float]) -> float:
    """A generic training loop for iterating over epochs and batches.

    Args:
        epochs: The integer value for epochs.
        dataloader: The dataloader.
        process_batch_fn: The process batch fn.

    Returns:
        The computed float value.
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
