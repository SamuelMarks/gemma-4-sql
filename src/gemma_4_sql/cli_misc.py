"""CLI commands for miscellaneous tasks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.sdk.export import export_model
from gemma_4_sql.sdk.logging import log_metrics
from gemma_4_sql.sdk.quantize import quantize_model
from gemma_4_sql.sdk.rag import build_rag_prompt, extract_schema_entities, retrieve_relevant_schema
from gemma_4_sql.tokenization import SQLTokenizer

if TYPE_CHECKING:
    import argparse
logger = logging.getLogger(__name__)


def tokenize_cmd(args: argparse.Namespace) -> None:
    """Run tokenization.


    Args:
        args: Parsed command-line arguments containing command-specific options."""
    tokenizer = SQLTokenizer(model_name=args.hf_model, vocab_size=args.vocab_size)
    if args.decode:
        try:
            tokens = [int(t.strip()) for t in args.decode.split(",")]
            tokenizer.decode(tokens)
        except ValueError as e:
            logger.warning("Invalid decode tokens: %s", e)
    elif args.encode:
        tokenizer.encode(args.encode)


def quantize_cmd(args: argparse.Namespace) -> None:
    """Run quantization.


    Args:
        args: Parsed command-line arguments containing command-specific options."""
    res = quantize_model(args.model, args.method, args.backend)
    logger.info("Quantization completed: %s", res)


def export_cmd(args: argparse.Namespace) -> None:
    """Run model export.


    Args:
        args: Parsed command-line arguments containing command-specific options."""
    res = export_model(args.model, args.path, args.backend)
    logger.info("Export completed: %s", res)


def rag_cmd(args: argparse.Namespace) -> None:
    """Build a RAG-augmented prompt or extract schema context.


    Args:
        args: Parsed command-line arguments containing command-specific options."""
    if getattr(args, "action", "build") == "extract":
        extract_schema_entities(args.ddl)
    elif getattr(args, "action", "build") == "retrieve":
        schema = extract_schema_entities(args.ddl)
        retrieve_relevant_schema(args.prompt, schema)
    else:
        build_rag_prompt(prompt=args.prompt, ddl=args.ddl)


def log_metrics_cmd(args: argparse.Namespace) -> None:
    """Log training metrics.

    Args:
        args: Parsed command-line arguments containing command-specific options."""
    metrics_dict = {}
    if args.metrics:
        for m in args.metrics.split(","):
            (k, v) = m.split("=")
            metrics_dict[k.strip()] = float(v.strip())
    res = log_metrics(metrics=metrics_dict, step=args.step, log_dir=args.log_dir, backend=args.backend)
    logger.info("Logging completed: %s", res)
