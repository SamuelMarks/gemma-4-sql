"""Benchmarking SDK module including 7-backend TPU comparison matrix orchestration."""

from __future__ import annotations

from collections.abc import Sequence

from gemma_4_sql.constants import TPU_COMPARISON_BACKENDS
from gemma_4_sql.exceptions import ExecutionStatus, GemmaSQLError, is_failed_status
from gemma_4_sql.sdk.registry import get_backend
from gemma_4_sql.type_hints import (
    BenchmarkMatrixCellDict,
    BenchmarkMatrixResultDict,
    BenchmarkResultDict,
    JSONValue,
)


def benchmark(
    model_name: str,
    hardware: str,
    batch_size: int,
    backend: str,
    dtype: str = "bfloat16",
    mode: str = "prefill",
    max_new_tokens: int = 128,
    warmup_steps: int = 5,
    **kwargs: JSONValue,
) -> BenchmarkResultDict:
    """Benchmark a Text-to-SQL model on a target hardware accelerator.

    Args:
        model_name: Hugging Face model identifier or preset name.
        hardware: Target hardware accelerator (e.g., `'tpu-v5p'`, `'tpu-v6e'`, `'gpu'`, `'cpu'`).
        batch_size: Number of sequences processed per benchmark batch.
        backend: Backend framework or comparison alias (`'hf_transformers'`, `'keras_jax'`, etc.).
        dtype: Numerical precision policy (e.g., `'bfloat16'`, `'float16'`, `'float32'`).
        mode: Benchmark execution mode (`'prefill'`, `'decode'`, `'generate'`, or `'end-to-end'`).
        max_new_tokens: Maximum number of tokens generated in non-prefill modes.
        warmup_steps: Number of untimed warmup iterations before latency measurement.
        **kwargs: Additional parameters including `num_runs`, `strict_hardware`, and multimodal options.

    Returns:
        Structured `BenchmarkResultDict` containing throughput (`tokens_per_sec`), latency, memory, and status.

    """
    return get_backend(backend).benchmark_model(
        model_name,
        hardware,
        batch_size,
        dtype=dtype,
        mode=mode,
        max_new_tokens=max_new_tokens,
        warmup_steps=warmup_steps,
        backend_alias=backend,
        **kwargs,
    )


def format_benchmark_matrix_markdown(result: BenchmarkMatrixResultDict) -> str:
    """Render a `BenchmarkMatrixResultDict` into a GitHub-flavored Markdown comparison table.

    Args:
        result: Completed multi-backend `BenchmarkMatrixResultDict`.

    Returns:
        Formatted Markdown table string comparing throughput, latency, compile time, HBM, and speedups.

    """
    lines = [
        f"### Gemma 4 Benchmark Matrix (`{result['model']}` on `{result['hardware']}`, `{result['dtype']}`)",
        "",
        "| Backend | Mode | Batch | Tok/s | Latency (ms) | Compile (ms) | Peak HBM (MB) | vs HF | vs JAX | Status |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for cell in result["cells"]:
        res = cell["result"]
        comp_ms = res.get("compile_latency_ms")
        comp_str = f"{comp_ms:.2f}" if comp_ms is not None else "n/a"
        hbm_mb = res.get("peak_hbm_mb")
        hbm_str = f"{hbm_mb:.1f}" if hbm_mb is not None else f"{res['memory_mb']:.1f}"
        vs_hf = cell.get("speedup_vs_hf_transformers")
        vs_hf_str = f"{vs_hf:.2f}x" if vs_hf is not None else "n/a"
        vs_jax = cell.get("speedup_vs_jax_native")
        vs_jax_str = f"{vs_jax:.2f}x" if vs_jax is not None else "n/a"
        lines.append(
            f"| `{cell['backend']}` | `{cell['mode']}` | {cell['batch_size']} | "
            f"{res['tokens_per_sec']:.2f} | {res['latency_ms']:.2f} | {comp_str} | "
            f"{hbm_str} | {vs_hf_str} | {vs_jax_str} | `{res['status']}` |"
        )
    return chr(10).join(lines)


def benchmark_matrix(
    model_name: str,
    hardware: str = "tpu",
    *,
    backends: Sequence[str] | None = None,
    modes: Sequence[str] = ("prefill", "decode"),
    batch_sizes: Sequence[int] = (1,),
    dtype: str = "bfloat16",
    **kwargs: JSONValue,
) -> BenchmarkMatrixResultDict:
    """Execute a multi-backend TPU/GPU/CPU comparison matrix across `backends`, `modes`, and `batch_sizes`.

    Args:
        model_name: Target Gemma 4 model identifier.
        hardware: Target hardware accelerator (`'tpu'`, `'tpu-v6e'`, `'gpu'`, `'cpu'`).
        backends: Sequence of backend aliases (defaults to all 7 `TPU_COMPARISON_BACKENDS`).
        modes: Sequence of benchmark modes (`'prefill'`, `'decode'`, `'end-to-end'`).
        batch_sizes: Sequence of batch sizes to profile.
        dtype: Numerical precision policy (`'bfloat16'`, `'float16'`, `'float32'`).
        **kwargs: Additional options forwarded to `benchmark` (`num_runs`, `warmup_steps`, `strict_hardware`).

    Returns:
        `BenchmarkMatrixResultDict` containing per-cell results, relative speedups, and a Markdown summary.

    Raises:
        GemmaSQLError: If `strict_hardware=True` or `raise_on_error=True` and any benchmark cell fails.

    """
    active_backends = tuple(backends) if backends is not None else TPU_COMPARISON_BACKENDS
    strict_hw = bool(kwargs.get("strict_hardware", False))
    raise_on_err = bool(kwargs.get("raise_on_error", False))
    max_new_tokens = int(str(kwargs.pop("max_new_tokens", 128)))
    warmup_steps = int(str(kwargs.pop("warmup_steps", 5)))

    raw_cells: list[tuple[str, str, int, BenchmarkResultDict]] = []
    any_failed = False

    for mode in modes:
        for bsz in batch_sizes:
            for backend_name in active_backends:
                cell_res: BenchmarkResultDict
                try:
                    cell_res = benchmark(
                        model_name,
                        hardware,
                        bsz,
                        backend_name,
                        dtype=dtype,
                        mode=mode,
                        max_new_tokens=max_new_tokens,
                        warmup_steps=warmup_steps,
                        **kwargs,
                    )
                except GemmaSQLError as err:
                    if strict_hw or raise_on_err:
                        raise
                    any_failed = True
                    cell_res = {
                        "backend": backend_name,
                        "backend_variant": backend_name,
                        "model": model_name,
                        "hardware": hardware,
                        "batch_size": bsz,
                        "tokens_per_sec": 0.0,
                        "latency_ms": 0.0,
                        "memory_mb": 0.0,
                        "status": f"failed: {err!s}",
                        "status_code": ExecutionStatus.FAILED.value,
                        "error_message": str(err),
                    }
                if is_failed_status(cell_res):
                    any_failed = True
                raw_cells.append((backend_name, mode, bsz, cell_res))

    hf_tps: dict[tuple[str, int], float] = {}
    jax_tps: dict[tuple[str, int], float] = {}
    for b_name, mode, bsz, c_res in raw_cells:
        tps = float(c_res["tokens_per_sec"])
        if tps > 0.0:
            if b_name in {"hf_transformers", "pytorch_hf"} and (mode, bsz) not in hf_tps:
                hf_tps[mode, bsz] = tps
            if b_name in {"jax_native", "jax"} and (mode, bsz) not in jax_tps:
                jax_tps[mode, bsz] = tps

    cells: list[BenchmarkMatrixCellDict] = []
    for b_name, mode, bsz, c_res in raw_cells:
        tps = float(c_res["tokens_per_sec"])
        base_hf = hf_tps.get((mode, bsz))
        base_jax = jax_tps.get((mode, bsz))
        cell_entry: BenchmarkMatrixCellDict = {
            "backend": b_name,
            "mode": mode,
            "batch_size": bsz,
            "result": c_res,
            "speedup_vs_hf_transformers": (tps / base_hf) if (tps > 0.0 and base_hf is not None) else None,
            "speedup_vs_jax_native": (tps / base_jax) if (tps > 0.0 and base_jax is not None) else None,
        }
        cells.append(cell_entry)

    overall_status = ExecutionStatus.FAILED.value if any_failed else ExecutionStatus.SUCCESS.value
    matrix_out: BenchmarkMatrixResultDict = {
        "model": model_name,
        "hardware": hardware,
        "dtype": dtype,
        "status": overall_status,
        "status_code": overall_status,
        "cells": cells,
    }
    matrix_out["markdown_table"] = format_benchmark_matrix_markdown(matrix_out)
    return matrix_out


__all__ = [
    "benchmark",
    "benchmark_matrix",
    "format_benchmark_matrix_markdown",
]
