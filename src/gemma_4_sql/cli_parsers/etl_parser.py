"""CLI argument parsers for ETL pipelines."""

from __future__ import annotations

import argparse
from typing import Callable

from gemma_4_sql.cli_etl import etl_posttrain_cmd, etl_pretrain_cmd, etl_sft_cmd
from gemma_4_sql.constants import DEFAULT_POSTTRAIN_DATASET, DEFAULT_PRETRAIN_DATASET, DEFAULT_SFT_DATASET


def add_etl_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help_text: str,
    default_dataset: str,
    cmd_func: Callable[[argparse.Namespace], int | None],
) -> argparse.ArgumentParser:
    """Add an individual ETL subcommand parser.

    Args:
        subparsers: Subparser collection to attach the parser to.
        name: Name of the subparser command (e.g., 'pretrain', 'sft', 'posttrain').
        help_text: Descriptive help string for the subparser.
        default_dataset: Default dataset name or identifier.
        cmd_func: Callback function executed when this command is chosen.

    Returns:
        The configured ArgumentParser instance for this ETL command.
    """
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("--dataset", default=default_dataset, help="Hugging Face dataset name.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--backend", default="jax", help="Backend to use.")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed sharding.")
    parser.add_argument("--tokenizer", default=None, help="Hugging Face tokenizer model name.")
    parser.add_argument("--duckdb-path", default=None, help="Optional path to DuckDB database.")
    parser.add_argument("--duckdb-table", default=None, help="Optional DuckDB table name.")
    parser.add_argument("--image-path", default=None, help="Optional path to schema image or directory.")
    parser.add_argument("--audio-path", default=None, help="Optional path to audio query file or directory.")
    parser.add_argument("--modality", default="text", choices=["text", "vision", "audio", "multimodal"], help="Explicit modality selector.")
    parser.set_defaults(func=cmd_func)
    return parser


def add_etl_parsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    """Register the root ETL parser and all associated subcommands.

    Args:
        subparsers: Top-level subparser collection to attach the ETL parser to.

    Returns:
        The configured parent ArgumentParser instance for 'etl'.
    """
    parser_etl = subparsers.add_parser("etl", help="Run ETL to prepare SQL training datasets.")
    etl_subparsers = parser_etl.add_subparsers(dest="etl_command", required=True)
    add_etl_subparser(etl_subparsers, "pretrain", "Run ETL for pretraining SQL datasets.", DEFAULT_PRETRAIN_DATASET, etl_pretrain_cmd)
    add_etl_subparser(etl_subparsers, "sft", "Run ETL for SFT SQL datasets.", DEFAULT_SFT_DATASET, etl_sft_cmd)
    add_etl_subparser(etl_subparsers, "posttrain", "Run ETL for post-training SQL datasets.", DEFAULT_POSTTRAIN_DATASET, etl_posttrain_cmd)
    return parser_etl
