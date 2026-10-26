"""CLI commands for miscellaneous tasks."""

from __future__ import annotations

import argparse
import json
import logging

from gemma_4_sql.exceptions import is_failed_status
from gemma_4_sql.sdk.export import export_model
from gemma_4_sql.sdk.logging import log_metrics
from gemma_4_sql.sdk.quantize import quantize_model
from gemma_4_sql.sdk.rag import build_rag_prompt, extract_schema_entities, retrieve_relevant_schema
from gemma_4_sql.tokenization import SQLTokenizer

logger = logging.getLogger(__name__)


def tokenize_cmd(args: argparse.Namespace) -> int:
    """Run tokenization.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if `--decode` receives invalid token IDs.

    """
    tokenizer = SQLTokenizer(model_name=args.hf_model, vocab_size=args.vocab_size)
    if args.decode:
        try:
            tokens = [int(t.strip()) for t in args.decode.split(",")]
            decoded = tokenizer.decode(tokens)
            print(str(decoded))
        except ValueError as e:
            logger.warning("Invalid decode tokens: %s", e)
            return 1
    elif args.encode:
        encoded = tokenizer.encode(args.encode)
        print(json.dumps(encoded))
    return 0


def quantize_cmd(args: argparse.Namespace) -> int:
    """Run quantization.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if quantization fails.

    """
    res = quantize_model(args.model, args.method, args.backend)
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0


def export_cmd(args: argparse.Namespace) -> int:
    """Run model export.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if export fails.

    """
    res = export_model(args.model, args.path, args.backend)
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0


def rag_cmd(args: argparse.Namespace) -> None:
    """Build a RAG-augmented prompt or extract schema context.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    """
    if args.action == "extract":
        schema = extract_schema_entities(args.ddl)
        print(json.dumps(schema, indent=2))
    elif args.action == "retrieve":
        schema = extract_schema_entities(args.ddl)
        retrieved = retrieve_relevant_schema(args.prompt, schema)
        print(str(retrieved))
    else:
        rag_prompt = build_rag_prompt(
            prompt=args.prompt,
            ddl=args.ddl,
            image_path=args.image_path,
            audio_path=args.audio_path,
        )
        print(str(rag_prompt))


def log_metrics_cmd(args: argparse.Namespace) -> int:
    """Log training metrics.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if metric parsing or logging fails.

    """
    metrics_dict: dict[str, float] = {}
    if args.metrics:
        for item in args.metrics.split(","):
            stripped = item.strip()
            if not stripped:
                continue
            if "=" not in stripped:
                logger.warning("Skipping malformed metric entry (expected key=value): %s", stripped)
                continue
            (k, v) = stripped.split("=", 1)
            try:
                metrics_dict[k.strip()] = float(v.strip())
            except ValueError as e:
                logger.warning("Invalid float value in metric entry '%s': %s", stripped, e)
                return 1
    res = log_metrics(metrics=metrics_dict, step=args.step, log_dir=args.log_dir, backend=args.backend)
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0
