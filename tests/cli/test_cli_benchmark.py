"""Provide module docstring."""

import argparse

from gemma_4_sql.cli import benchmark_cmd


def test_benchmark_cmd() -> object:
    """Initialize function test_benchmark_cmd."""
    args = argparse.Namespace(
        model="gemma-4",
        hardware="gpu",
        batch_size=1,
        backend="jax",
        dtype="bfloat16",
        mode="prefill",
        max_new_tokens=128,
        warmup_steps=5,
    )
    benchmark_cmd(args)
