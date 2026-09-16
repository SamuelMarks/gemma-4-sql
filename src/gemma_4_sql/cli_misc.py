"""CLI commands for miscellaneous tasks."""

from __future__ import annotations

import json
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
        args: Parsed command-line arguments containing command-specific options.
    """
    tokenizer = SQLTokenizer(model_name=args.hf_model, vocab_size=args.vocab_size)
    if args.decode:
        try:
            tokens = [int(t.strip()) for t in args.decode.split(",")]
            decoded = tokenizer.decode(tokens)
            print(str(decoded))
        except ValueError as e:
            logger.warning("Invalid decode tokens: %s", e)
    elif args.encode:
        encoded = tokenizer.encode(args.encode)
        print(json.dumps(encoded))


def quantize_cmd(args: argparse.Namespace) -> None:
    """Run quantization.

    Args:
        args: Parsed command-line arguments containing command-specific options.
    """
    res = quantize_model(args.model, args.method, args.backend)
    print(json.dumps(res, indent=2))


def export_cmd(args: argparse.Namespace) -> None:
    """Run model export.

    Args:
        args: Parsed command-line arguments containing command-specific options.
    """
    res = export_model(args.model, args.path, args.backend)
    print(json.dumps(res, indent=2))


def rag_cmd(args: argparse.Namespace) -> None:
    """Build a RAG-augmented prompt or extract schema context.

    Args:
        args: Parsed command-line arguments containing command-specific options.
    """
    if getattr(args, "action", "build") == "extract":
        schema = extract_schema_entities(args.ddl)
        print(json.dumps(schema, indent=2))
    elif getattr(args, "action", "build") == "retrieve":
        schema = extract_schema_entities(args.ddl)
        retrieved = retrieve_relevant_schema(args.prompt, schema)
        print(str(retrieved))
    else:
        rag_prompt = build_rag_prompt(prompt=args.prompt, ddl=args.ddl)
        image_path = getattr(args, "image_path", None)
        audio_path = getattr(args, "audio_path", None)
        if image_path is not None or audio_path is not None:
            from gemma_4_sql.backends.common_multimodal import format_multimodal_prompt

            rag_prompt = format_multimodal_prompt(
                rag_prompt,
                has_image=image_path is not None,
                has_audio=audio_path is not None,
            )["prompt"]
        print(str(rag_prompt))


def log_metrics_cmd(args: argparse.Namespace) -> None:
    """Log training metrics.

    Args:
        args: Parsed command-line arguments containing command-specific options.
    """
    metrics_dict = {}
    if args.metrics:  # pragma: no cover
        for m in args.metrics.split(","):
            (k, v) = m.split("=")
            metrics_dict[k.strip()] = float(v.strip())
    res = log_metrics(metrics=metrics_dict, step=args.step, log_dir=args.log_dir, backend=args.backend)
    print(json.dumps(res, indent=2))
