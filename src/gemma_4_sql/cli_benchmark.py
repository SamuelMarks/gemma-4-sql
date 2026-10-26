"""Benchmarking CLI commands."""

from __future__ import annotations

import argparse
import json
from typing import Protocol

from gemma_4_sql.exceptions import is_failed_status
from gemma_4_sql.sdk import benchmark


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


def benchmark_cmd(args: argparse.Namespace | BenchmarkArgsProtocol) -> int:
    """Benchmark a model on target hardware.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if benchmarking fails.

    """
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
    )
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0
