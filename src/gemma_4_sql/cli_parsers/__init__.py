"""Modular CLI argument parser package for gemma-4-sql."""

from __future__ import annotations

from gemma_4_sql.cli_parsers.benchmark_parser import (
    add_benchmark_parser,
    add_misc_parsers,
    add_rag_log_parsers,
    add_tokenize_parser,
)
from gemma_4_sql.cli_parsers.db_parser import add_db_parsers
from gemma_4_sql.cli_parsers.etl_parser import add_etl_parsers, add_etl_subparser
from gemma_4_sql.cli_parsers.serve_parser import (
    add_evaluate_parsers,
    add_generate_agent_parsers,
    add_inference_parsers,
    add_serve_export_parsers,
)
from gemma_4_sql.cli_parsers.train_parser import (
    add_peft_quantize_parsers,
    add_training_parsers,
    add_training_subparser,
)

__all__ = [
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
]
