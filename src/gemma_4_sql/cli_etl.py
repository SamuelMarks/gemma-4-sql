"""CLI commands for ETL processing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.sdk.etl import etl_posttrain, etl_pretrain, etl_sft

if TYPE_CHECKING:
    import argparse
logger = logging.getLogger(__name__)


def etl_pretrain_cmd(args: argparse.Namespace) -> None:
    """Run ETL for pretraining."""
    etl_pretrain(dataset_name=args.dataset, split=args.split, batch_size=args.batch_size, backend=args.backend, distributed=args.distributed, tokenizer_name=args.tokenizer, duckdb_path=args.duckdb_path, duckdb_table=args.duckdb_table)


def etl_sft_cmd(args: argparse.Namespace) -> None:
    """Run ETL for SFT."""
    etl_sft(dataset_name=args.dataset, split=args.split, batch_size=args.batch_size, backend=args.backend, distributed=args.distributed, tokenizer_name=args.tokenizer, duckdb_path=args.duckdb_path, duckdb_table=args.duckdb_table)


def etl_posttrain_cmd(args: argparse.Namespace) -> None:
    """Run ETL for post-training."""
    etl_posttrain(dataset_name=args.dataset, split=args.split, batch_size=args.batch_size, backend=args.backend, distributed=args.distributed, tokenizer_name=args.tokenizer, duckdb_path=args.duckdb_path, duckdb_table=args.duckdb_table)
