"""Benchmarking and 7-backend comparison matrix CLI commands."""

from __future__ import annotations

import argparse
import json
from typing import Protocol, runtime_checkable

from gemma_4_sql.cli_misc import (
    OutputJSONCLIArgsProtocol,
    StrictHardwareCLIArgsProtocol,
    print_cli_error,
    write_atomic_output_json,
)
from gemma_4_sql.constants import TPU_COMPARISON_BACKENDS
from gemma_4_sql.exceptions import GemmaSQLError, is_failed_status
from gemma_4_sql.sdk import benchmark, benchmark_matrix
from gemma_4_sql.type_hints import JSONValue


@runtime_checkable
class _NumRunsCLIArgsProtocol(Protocol):
    """Protocol for parsed benchmark CLI arguments exposing `.num_runs`."""

    @property
    def num_runs(self) -> int | None:
        """Provide timed iteration count."""
        ...


@runtime_checkable
class _TensorParallelCLIArgsProtocol(Protocol):
    """Protocol for parsed benchmark CLI arguments exposing `.tensor_parallel_size`."""

    @property
    def tensor_parallel_size(self) -> int | None:
        """Provide tensor parallelism degree."""
        ...


class BenchmarkArgsProtocol(Protocol):
    """Protocol for parsed `benchmark` CLI arguments."""

    model: str
    hardware: str
    batch_size: int
    backend: str
    dtype: str
    mode: str
    max_new_tokens: int
    warmup_steps: int
    image_path: str | None
    audio_path: str | None
    modality: str


class BenchmarkMatrixArgsProtocol(Protocol):
    """Protocol for parsed `benchmark-matrix` CLI arguments."""

    model: str
    hardware: str
    backends: str
    modes: str
    batch_sizes: str
    dtype: str
    max_new_tokens: int
    warmup_steps: int
    num_runs: int
    strict_hardware: bool
    markdown: bool
    output_json: str | None


def benchmark_cmd(args: argparse.Namespace | BenchmarkArgsProtocol) -> int:
    """Benchmark a model on target hardware.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if benchmarking fails.

    """
    extra: dict[str, JSONValue] = {}
    if isinstance(args, _NumRunsCLIArgsProtocol) and args.num_runs is not None:
        extra["num_runs"] = int(args.num_runs)
    if isinstance(args, StrictHardwareCLIArgsProtocol) and args.strict_hardware:
        extra["strict_hardware"] = True
    if isinstance(args, _TensorParallelCLIArgsProtocol) and args.tensor_parallel_size is not None:
        extra["tensor_parallel_size"] = int(args.tensor_parallel_size)

    try:
        res = benchmark(
            model_name=args.model,
            hardware=args.hardware,
            batch_size=args.batch_size,
            backend=args.backend,
            dtype=args.dtype,
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            warmup_steps=args.warmup_steps,
            image_path=args.image_path,
            audio_path=args.audio_path,
            modality=args.modality,
            **extra,
        )
    except GemmaSQLError as err:
        return print_cli_error(
            err,
            args.backend,
            model=args.model,
            hardware=args.hardware,
            batch_size=args.batch_size,
        )
    formatted = json.dumps(res, indent=2)
    if isinstance(args, (argparse.Namespace, OutputJSONCLIArgsProtocol)):
        write_atomic_output_json(args, formatted)
    print(formatted)
    return 1 if is_failed_status(res) else 0


def benchmark_matrix_cmd(args: argparse.Namespace | BenchmarkMatrixArgsProtocol) -> int:
    """Execute a multi-backend TPU/GPU/CPU comparison matrix and emit JSON or Markdown output.

    Args:
        args: Parsed command-line arguments for `benchmark-matrix`.

    Returns:
        Exit code `0` on success, or `1` if any cell or hardware validation fails.

    """
    backends_list = (
        list(TPU_COMPARISON_BACKENDS)
        if args.backends.strip().lower() in {"", "all"}
        else [b.strip() for b in args.backends.split(",") if b.strip()]
    )
    modes_list = [m.strip() for m in args.modes.split(",") if m.strip()] or ["prefill"]
    try:
        batch_sizes_list = [int(b.strip()) for b in args.batch_sizes.split(",") if b.strip()] or [1]
    except ValueError as err:
        from gemma_4_sql.exceptions import ConfigurationError

        cfg_err = ConfigurationError(f"Invalid --batch-sizes CSV {args.batch_sizes!r}: {err}")
        return print_cli_error(cfg_err, "matrix", model=args.model, hardware=args.hardware)

    try:
        res = benchmark_matrix(
            model_name=args.model,
            hardware=args.hardware,
            backends=backends_list,
            modes=modes_list,
            batch_sizes=batch_sizes_list,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            warmup_steps=args.warmup_steps,
            num_runs=args.num_runs,
            strict_hardware=args.strict_hardware,
        )
    except GemmaSQLError as err:
        return print_cli_error(err, "matrix", model=args.model, hardware=args.hardware)

    formatted_json = json.dumps(res, indent=2)
    write_atomic_output_json(args, formatted_json)
    if args.markdown and "markdown_table" in res:
        print(res["markdown_table"])
    else:
        print(formatted_json)
    return 1 if is_failed_status(res) else 0


__all__ = [
    "BenchmarkArgsProtocol",
    "BenchmarkMatrixArgsProtocol",
    "benchmark_cmd",
    "benchmark_matrix_cmd",
]
