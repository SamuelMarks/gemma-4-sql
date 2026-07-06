"""CLI commands for ETL processing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.sdk.etl import etl_posttrain, etl_pretrain, etl_sft
from gemma_4_sql.type_hints import ETLConfig

if TYPE_CHECKING:
    import argparse
logger = logging.getLogger(__name__)


def etl_pretrain_cmd(args: argparse.Namespace) -> None:
    """Run ETL for pretraining.

    Args:
        args: Additional positional arguments.
    """
    config = ETLConfig(dataset_name=args.dataset, split=args.split, batch_size=args.batch_size, distributed=args.distributed, tokenizer_name=args.tokenizer, duckdb_path=args.duckdb_path, duckdb_table=args.duckdb_table)
    etl_pretrain(config, backend=args.backend)


def etl_sft_cmd(args: argparse.Namespace) -> None:
    """Run ETL for SFT.

    Args:
        args: Additional positional arguments.
    """
    config = ETLConfig(dataset_name=args.dataset, split=args.split, batch_size=args.batch_size, distributed=args.distributed, tokenizer_name=args.tokenizer, duckdb_path=args.duckdb_path, duckdb_table=args.duckdb_table)
    etl_sft(config, backend=args.backend)


def etl_posttrain_cmd(args: argparse.Namespace) -> None:
    """Run ETL for post-training."""
    config = ETLConfig(dataset_name=args.dataset, split=args.split, batch_size=args.batch_size, distributed=args.distributed, tokenizer_name=args.tokenizer, duckdb_path=args.duckdb_path, duckdb_table=args.duckdb_table)
    etl_posttrain(config, backend=args.backend)
