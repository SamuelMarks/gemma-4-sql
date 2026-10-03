"""Provide module docstring."""

import argparse

import pytest

from gemma_4_sql.cli_benchmark import benchmark_cmd


def test_benchmark_cmd(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda **kwargs: {"status": "ok"})
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
