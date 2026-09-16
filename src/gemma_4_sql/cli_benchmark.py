"""Benchmarking CLI commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from gemma_4_sql.sdk import benchmark

if TYPE_CHECKING:
    import argparse


def benchmark_cmd(args: argparse.Namespace) -> None:
    """Benchmark a model on target hardware.

    Args:
        args: Parsed command-line arguments containing command-specific options.
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
        image_path=getattr(args, "image_path", None),
        audio_path=getattr(args, "audio_path", None),
        modality=getattr(args, "modality", "text"),
    )
    print(json.dumps(res, indent=2))
