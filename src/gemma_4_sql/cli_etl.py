"""CLI commands for ETL processing."""

from __future__ import annotations

import argparse
import logging

from gemma_4_sql.exceptions import is_failed_status
from gemma_4_sql.sdk.etl import etl_posttrain, etl_pretrain, etl_sft
from gemma_4_sql.type_hints import ETLConfig

logger = logging.getLogger(__name__)


def etl_pretrain_cmd(args: argparse.Namespace) -> int:
    """Run ETL for pretraining.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if ETL fails.

    """
    config = ETLConfig(dataset_name=args.dataset, split=args.split, batch_size=args.batch_size, distributed=args.distributed, tokenizer_name=args.tokenizer, duckdb_path=args.duckdb_path, duckdb_table=args.duckdb_table)
    res = etl_pretrain(config, backend=args.backend)
    return 1 if is_failed_status(res) else 0


def etl_sft_cmd(args: argparse.Namespace) -> int:
    """Run ETL for SFT.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if ETL fails.

    """
    config = ETLConfig(dataset_name=args.dataset, split=args.split, batch_size=args.batch_size, distributed=args.distributed, tokenizer_name=args.tokenizer, duckdb_path=args.duckdb_path, duckdb_table=args.duckdb_table)
    res = etl_sft(config, backend=args.backend)
    return 1 if is_failed_status(res) else 0


def etl_posttrain_cmd(args: argparse.Namespace) -> int:
    """Run ETL for post-training.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if ETL fails.

    """
    config = ETLConfig(dataset_name=args.dataset, split=args.split, batch_size=args.batch_size, distributed=args.distributed, tokenizer_name=args.tokenizer, duckdb_path=args.duckdb_path, duckdb_table=args.duckdb_table)
    res = etl_posttrain(config, backend=args.backend)
    return 1 if is_failed_status(res) else 0
