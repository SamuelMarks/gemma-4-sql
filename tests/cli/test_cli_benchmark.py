"""Unit and behavioral tests for CLI benchmark subcommand."""

from __future__ import annotations

import argparse
import json

import pytest

from gemma_4_sql.cli_benchmark import benchmark_cmd
from gemma_4_sql.exceptions import BenchmarkError
from gemma_4_sql.type_hints import BenchmarkResultDict, JSONDict, JSONValue


def test_benchmark_cmd(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify `benchmark_cmd` forwards all CLI arguments to `sdk.benchmark` and prints JSON output."""
    captured_kwargs: JSONDict = {}

    def _fake_benchmark(**kwargs: JSONValue) -> BenchmarkResultDict:
        """Capture forwarded benchmark arguments and return a structured result."""
        captured_kwargs.update(kwargs)
        return {
            "backend": str(kwargs["backend"]),
            "model": str(kwargs["model_name"]),
            "hardware": str(kwargs["hardware"]),
            "batch_size": int(str(kwargs["batch_size"])),
            "tokens_per_sec": 512.0,
            "latency_ms": 12.5,
            "memory_mb": 1024.0,
            "status": "success",
        }

    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", _fake_benchmark)
    args = argparse.Namespace(
        model="gemma-4",
        hardware="gpu",
        batch_size=4,
        backend="jax",
        dtype="bfloat16",
        mode="prefill",
        max_new_tokens=128,
        warmup_steps=5,
        image_path="diagram.png",
        audio_path=None,
        modality="vision",
    )
    benchmark_cmd(args)

    assert captured_kwargs["model_name"] == "gemma-4"
    assert captured_kwargs["hardware"] == "gpu"
    assert captured_kwargs["batch_size"] == 4
    assert captured_kwargs["backend"] == "jax"
    assert captured_kwargs["dtype"] == "bfloat16"
    assert captured_kwargs["mode"] == "prefill"
    assert captured_kwargs["max_new_tokens"] == 128
    assert captured_kwargs["warmup_steps"] == 5
    assert captured_kwargs["image_path"] == "diagram.png"
    assert captured_kwargs["modality"] == "vision"

    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["status"] == "success"
    assert parsed["tokens_per_sec"] == 512.0
    assert parsed["backend"] == "jax"

    def _fail_benchmark(**_kwargs: JSONValue) -> BenchmarkResultDict:
        """Raise BenchmarkError to verify CLI error formatting."""
        raise BenchmarkError("tpu benchmark error", backend="jax", model_name="gemma-4")

    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", _fail_benchmark)
    assert benchmark_cmd(args) == 1
    err_parsed = json.loads(capsys.readouterr().out)
    assert err_parsed["status_code"] == "failed"
    assert "tpu benchmark error" in err_parsed["error_message"]
