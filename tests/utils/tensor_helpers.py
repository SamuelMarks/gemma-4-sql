"""Tensor comparison and fixture generation utilities for tests."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import mlx.core as mx
import torch


@runtime_checkable
class FlattenableTensorProtocol(Protocol):
    """Protocol for tensors supporting `.shape`, `.flatten()`, and `.tolist()`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Provide tensor dimensions."""
        ...

    def flatten(self) -> FlattenableTensorProtocol:
        """Return a 1D flattened tensor view."""
        ...

    def tolist(self) -> list[float] | Sequence[float]:
        """Convert flattened tensor to a list of floats."""
        ...


type ComparableTensor = FlattenableTensorProtocol | Sequence[float] | Sequence[Sequence[float]]


def create_dummy_weight_matrix(rows: int, cols: int, seed: float = 0.42) -> list[list[float]]:
    """Generate a deterministic 2D matrix of floating-point values for testing.

    Args:
        rows: Number of rows in the matrix.
        cols: Number of columns in the matrix.
        seed: Initial floating-point seed value for pseudorandom simulation.

    Returns:
        2D nested list representing the generated weight matrix.

    """
    matrix: list[list[float]] = []
    val = seed
    for _r in range(rows):
        row: list[float] = []
        for _c in range(cols):
            val = (val * 1.337 + 0.123) % 2.0 - 1.0
            row.append(round(val, 4))
        matrix.append(row)
    return matrix


def assert_sequences_close(
    actual: Sequence[float],
    expected: Sequence[float],
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> None:
    """Assert that two numerical sequences are element-wise equal within tolerance.

    Args:
        actual: The actual output numerical sequence.
        expected: The reference expected numerical sequence.
        rtol: Relative tolerance limit.
        atol: Absolute tolerance limit.

    Raises:
        AssertionError: If lengths differ or any element diverges beyond tolerance.

    """
    if len(actual) != len(expected):
        raise AssertionError(f"Length mismatch: {len(actual)} vs {len(expected)}")
    for i, (a, b) in enumerate(zip(actual, expected)):
        diff = abs(a - b)
        limit = atol + rtol * abs(b)
        if diff > limit:
            raise AssertionError(f"Divergence at index {i}: actual={a} != expected={b} (diff={diff} > limit={limit})")


def compute_snr(original: Sequence[float], reconstructed: Sequence[float]) -> float:
    """Calculate Signal-to-Noise Ratio (SNR) in decibels between signals.

    Args:
        original: Uncompressed or target numerical signal.
        reconstructed: Quantized or reconstructed numerical signal.

    Returns:
        SNR in decibels (B).

    """
    if len(original) != len(reconstructed):
        raise ValueError("Sequence lengths must match for SNR computation.")
    signal_power = sum(x * x for x in original)
    noise_power = sum((x - y) * (x - y) for x, y in zip(original, reconstructed))
    if noise_power == 0.0:
        return float("inf")
    if signal_power == 0.0:
        return 0.0
    return 10.0 * math.log10(signal_power / noise_power)


def assert_tensors_close(
    actual: ComparableTensor,
    expected: ComparableTensor,
    backend: str = "pytorch",
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> None:
    """Compare tensor values flexibly across frameworks.

    Args:
        actual: The actual tensor.
        expected: The expected tensor.
        backend: Backend framework name.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Raises:
        AssertionError: If shapes differ or elements diverge.
        ValueError: If `backend` is not recognized.

    """
    if isinstance(actual, FlattenableTensorProtocol) and isinstance(expected, FlattenableTensorProtocol) and backend in {"pytorch", "jax", "mlx"}:
        assert list(actual.shape) == list(expected.shape)
        actual_list = [float(x) for x in actual.flatten().tolist()]
        expected_list = [float(x) for x in expected.flatten().tolist()]
        assert_sequences_close(actual_list, expected_list, rtol=rtol, atol=atol)
    elif backend == "jax":
        actual_jax = jnp.asarray(actual, dtype=jnp.float32)
        expected_jax = jnp.asarray(expected, dtype=jnp.float32)
        assert list(actual_jax.shape) == list(expected_jax.shape)
        actual_flat = [float(x) for x in actual_jax.reshape(-1).tolist()]
        expected_flat = [float(x) for x in expected_jax.reshape(-1).tolist()]
        assert_sequences_close(actual_flat, expected_flat, rtol=rtol, atol=atol)
    elif backend == "mlx":
        assert not isinstance(actual, FlattenableTensorProtocol) and not isinstance(expected, FlattenableTensorProtocol)
        actual_mx = mx.array(list(actual), dtype=mx.float32)
        expected_mx = mx.array(list(expected), dtype=mx.float32)
        assert list(actual_mx.shape) == list(expected_mx.shape)
        assert bool(mx.allclose(actual_mx, expected_mx, rtol=rtol, atol=atol).item())
    elif backend == "pytorch":
        actual_pt = torch.as_tensor(actual, dtype=torch.float32)
        expected_pt = torch.as_tensor(expected, dtype=torch.float32)
        assert list(actual_pt.shape) == list(expected_pt.shape)
        actual_flat = [float(x) for x in actual_pt.reshape(-1).tolist()]
        expected_flat = [float(x) for x in expected_pt.reshape(-1).tolist()]
        assert_sequences_close(actual_flat, expected_flat, rtol=rtol, atol=atol)
    else:
        raise ValueError(f"Unknown backend for tensor comparison: {backend}")
