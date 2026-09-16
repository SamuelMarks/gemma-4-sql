"""Tensor comparison and fixture generation utilities for tests."""

from __future__ import annotations

import math
from collections.abc import Sequence


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
    for r in range(rows):
        row: list[float] = []
        for c in range(cols):
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
        SNR in decibels (dB).
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
