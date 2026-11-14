"""Unit and behavioral tests for CLI benchmark subcommand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from gemma_4_sql.cli_benchmark import benchmark_cmd, benchmark_matrix_cmd
from gemma_4_sql.cli_misc import extract_hardware_cli_kwargs, write_atomic_output_json
from gemma_4_sql.exceptions import BenchmarkError, HardwareUnavailableError
from gemma_4_sql.type_hints import BenchmarkMatrixResultDict, BenchmarkResultDict, JSONDict, JSONValue


def test_benchmark_cmd(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify `benchmark_cmd` forwards all CLI arguments to `sdk.benchmark` and prints JSON output."""
    captured_kwargs: JSONDict = {}

    def _fake_benchmark(**kwargs: JSONValue) -> BenchmarkResultDict:
        """Capture forwarded benchmark arguments and return a structured result.

        Returns:
            Simulated benchmark result dictionary.

        """
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
        """Raise BenchmarkError to verify CLI error formatting.

        Raises:
            BenchmarkError: Always raised to test CLI error handling.

        """
        raise BenchmarkError("tpu benchmark error", backend="jax", model_name="gemma-4")

    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", _fail_benchmark)
    assert benchmark_cmd(args) == 1
    err_parsed = json.loads(capsys.readouterr().out)
    assert err_parsed["status_code"] == "failed"
    assert "tpu benchmark error" in err_parsed["error_message"]


def test_benchmark_cmd_extended_flags_and_matrix_cmd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify extended CLI benchmark flags and `benchmark_matrix_cmd`."""
    captured: JSONDict = {}

    def _ok_bench(**kwargs: JSONValue) -> BenchmarkResultDict:
        """Record forwarded kwargs and return success.

        Returns:
            Simulated `BenchmarkResultDict`.

        """
        captured.update(kwargs)
        return {
            "backend": str(kwargs["backend"]),
            "model": str(kwargs["model_name"]),
            "hardware": str(kwargs["hardware"]),
            "batch_size": int(str(kwargs["batch_size"])),
            "tokens_per_sec": 900.0,
            "latency_ms": 2.0,
            "memory_mb": 512.0,
            "status": "success",
        }

    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", _ok_bench)
    out_file = tmp_path / "nested" / "bench.json"
    ext_args = argparse.Namespace(
        model="google/gemma-4-E4B",
        hardware="tpu-v6e",
        batch_size=8,
        backend="jax_native",
        dtype="bfloat16",
        mode="decode",
        max_new_tokens=64,
        warmup_steps=2,
        num_runs=15,
        strict_hardware=True,
        tensor_parallel_size=4,
        output_json=str(out_file),
        image_path=None,
        audio_path=None,
        modality="text",
    )
    assert benchmark_cmd(ext_args) == 0
    assert captured["num_runs"] == 15
    assert captured["strict_hardware"] is True
    assert captured["tensor_parallel_size"] == 4
    assert out_file.is_file()
    assert json.loads(out_file.read_text(encoding="utf-8"))["tokens_per_sec"] == 900.0
    capsys.readouterr()

    # Non-Namespace custom BenchmarkArgsProtocol branch (without output_json)
    class _MinimalBenchArgs:
        """Minimal `BenchmarkArgsProtocol` double without `output_json`."""

        model: str = "m"
        hardware: str = "cpu"
        batch_size: int = 1
        backend: str = "jax"
        dtype: str = "bfloat16"
        mode: str = "prefill"
        max_new_tokens: int = 16
        warmup_steps: int = 1
        image_path: str | None = None
        audio_path: str | None = None
        modality: str = "text"

    assert benchmark_cmd(_MinimalBenchArgs()) == 0
    capsys.readouterr()

    # Direct cli_misc helper branches (dict payload + explicit str path + hardware kwargs)
    direct_path = tmp_path / "direct.json"
    write_atomic_output_json(str(direct_path), {"ok": True})
    assert json.loads(direct_path.read_text(encoding="utf-8")) == {"ok": True}
    write_atomic_output_json(None, {"ignored": True})
    hw_ns = argparse.Namespace(hardware="tpu-v5e", strict_hardware=True)
    assert extract_hardware_cli_kwargs(hw_ns) == {"hardware": "tpu-v5e", "strict_hardware": True}

    # benchmark_matrix_cmd: JSON mode, Markdown mode, "all" backends, invalid CSV, and GemmaSQLError
    def _fake_matrix(**kwargs: JSONValue) -> BenchmarkMatrixResultDict:
        """Return simulated `BenchmarkMatrixResultDict`.

        Returns:
            Simulated `BenchmarkMatrixResultDict`.

        """
        return {
            "model": str(kwargs["model_name"]),
            "hardware": str(kwargs["hardware"]),
            "dtype": str(kwargs["dtype"]),
            "status": "success",
            "status_code": "success",
            "cells": [],
            "markdown_table": "| Backend | Tok/s |\n| --- | --- |",
        }

    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark_matrix", _fake_matrix)
    mat_json = tmp_path / "matrix.json"
    mat_args = argparse.Namespace(
        model="google/gemma-4-E4B",
        hardware="tpu-v6e",
        backends="all",
        modes="prefill,decode",
        batch_sizes="1,4",
        dtype="bfloat16",
        max_new_tokens=32,
        warmup_steps=1,
        num_runs=3,
        strict_hardware=True,
        markdown=True,
        output_json=str(mat_json),
    )
    assert benchmark_matrix_cmd(mat_args) == 0
    assert "| Backend | Tok/s |" in capsys.readouterr().out
    assert mat_json.is_file()

    mat_args.backends = "jax_native,maxtext"
    mat_args.markdown = False
    mat_args.output_json = None
    assert benchmark_matrix_cmd(mat_args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "success"

    mat_args.batch_sizes = "not_an_int"
    assert benchmark_matrix_cmd(mat_args) == 1
    capsys.readouterr()

    mat_args.batch_sizes = "1"

    def _fail_matrix(**_kwargs: JSONValue) -> BenchmarkMatrixResultDict:
        """Raise HardwareUnavailableError.

        Raises:
            HardwareUnavailableError: Always raised to test CLI error exit code.

        """
        raise HardwareUnavailableError("tpu missing", hardware="tpu-v6e")

    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark_matrix", _fail_matrix)
    assert benchmark_matrix_cmd(mat_args) == 1
    capsys.readouterr()
