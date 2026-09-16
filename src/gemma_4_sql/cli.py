"""Main CLI entrypoint for gemma-4-sql."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from gemma_4_sql.cli_benchmark import benchmark_cmd
from gemma_4_sql.cli_db import db_execute_cmd, embed_duckdb_cmd
from gemma_4_sql.cli_etl import etl_posttrain_cmd, etl_pretrain_cmd, etl_sft_cmd
from gemma_4_sql.cli_misc import export_cmd, log_metrics_cmd, quantize_cmd, rag_cmd, tokenize_cmd
from gemma_4_sql.cli_parsers import (
    add_benchmark_parser,
    add_db_parsers,
    add_etl_parsers,
    add_etl_subparser,
    add_evaluate_parsers,
    add_generate_agent_parsers,
    add_inference_parsers,
    add_misc_parsers,
    add_peft_quantize_parsers,
    add_rag_log_parsers,
    add_serve_export_parsers,
    add_tokenize_parser,
    add_training_parsers,
    add_training_subparser,
)
from gemma_4_sql.cli_serve import agent_cmd, chat_cmd, evaluate_cmd, few_shot_cmd, generate_cmd, serve_cmd
from gemma_4_sql.cli_train import dpo_cmd, peft_cmd, posttrain_cmd, pretrain_cmd, sft_cmd, train_cmd

# Backward-compatible internal aliases
_add_etl_subparser = add_etl_subparser
_add_etl_parsers = add_etl_parsers
_add_peft_quantize_parsers = add_peft_quantize_parsers
_add_training_subparser = add_training_subparser
_add_training_parsers = add_training_parsers
_add_evaluate_parsers = add_evaluate_parsers
_add_serve_export_parsers = add_serve_export_parsers
_add_generate_agent_parsers = add_generate_agent_parsers
_add_inference_parsers = add_inference_parsers
_add_tokenize_execute_parsers = add_db_parsers
_add_misc_parsers = add_misc_parsers

__all__ = [
    "_add_etl_parsers",
    "_add_etl_subparser",
    "_add_evaluate_parsers",
    "_add_generate_agent_parsers",
    "_add_inference_parsers",
    "_add_misc_parsers",
    "_add_peft_quantize_parsers",
    "_add_serve_export_parsers",
    "_add_tokenize_execute_parsers",
    "_add_training_parsers",
    "_add_training_subparser",
    "add_benchmark_parser",
    "add_db_parsers",
    "add_etl_parsers",
    "add_etl_subparser",
    "add_evaluate_parsers",
    "add_generate_agent_parsers",
    "add_inference_parsers",
    "add_misc_parsers",
    "add_peft_quantize_parsers",
    "add_rag_log_parsers",
    "add_serve_export_parsers",
    "add_tokenize_parser",
    "add_training_parsers",
    "add_training_subparser",
    "agent_cmd",
    "benchmark_cmd",
    "build_root_parser",
    "chat_cmd",
    "cli",
    "db_execute_cmd",
    "dpo_cmd",
    "embed_duckdb_cmd",
    "etl_posttrain_cmd",
    "etl_pretrain_cmd",
    "etl_sft_cmd",
    "evaluate_cmd",
    "export_cmd",
    "few_shot_cmd",
    "generate_cmd",
    "log_metrics_cmd",
    "peft_cmd",
    "posttrain_cmd",
    "pretrain_cmd",
    "quantize_cmd",
    "rag_cmd",
    "serve_cmd",
    "sft_cmd",
    "tokenize_cmd",
    "train_cmd",
]


def build_root_parser() -> argparse.ArgumentParser:
    """Construct and configure the root ArgumentParser with all subcommands.

    Returns:
        The root ArgumentParser with all registered command parsers.
    """
    parser = argparse.ArgumentParser(description="CLI for gemma-4-sql dataset generation and model training.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_etl_parsers(subparsers)
    add_training_parsers(subparsers)
    add_evaluate_parsers(subparsers)
    add_inference_parsers(subparsers)
    add_misc_parsers(subparsers)
    return parser


def cli(args: Sequence[str] | None = None) -> int:
    """Run main CLI entrypoint.

    Standard exit codes:
        0: Successful execution.
        1: General runtime error.
        2: Command-line parsing / usage error.
        3: Database execution failure.

    Args:
        args: Optional list or sequence of command-line argument strings.

    Returns:
        Integer exit code (0 for success, non-zero on failure).
    """
    parser = build_root_parser()
    parsed_args = parser.parse_args(args)
    res = parsed_args.func(parsed_args)
    return int(res) if isinstance(res, int) else 0


if __name__ == "__main__":
    import sys

    sys.exit(cli())
