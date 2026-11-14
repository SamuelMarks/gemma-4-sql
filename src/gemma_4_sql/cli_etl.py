"""CLI commands for ETL processing."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from gemma_4_sql.cli_misc import print_cli_error
from gemma_4_sql.cli_protocols import ETLArgsProtocol
from gemma_4_sql.exceptions import GemmaSQLError, is_failed_status
from gemma_4_sql.sdk.etl import etl_posttrain, etl_pretrain, etl_sft
from gemma_4_sql.type_hints import DataloaderResultDict, ETLConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class ETLMultimodalCLIArgsProtocol(Protocol):
    """Protocol for parsed ETL CLI arguments carrying multimodal column and file path flags."""

    @property
    def image_column(self) -> str | None:
        """Provide optional image column name."""
        ...

    @property
    def audio_column(self) -> str | None:
        """Provide optional audio column name."""
        ...

    @property
    def modality(self) -> str | None:
        """Provide explicit modality selector."""
        ...


@runtime_checkable
class ETLMediaPathsCLIArgsProtocol(Protocol):
    """Protocol for parsed ETL CLI arguments carrying `--image-path` and `--audio-path` flags."""

    @property
    def image_path(self) -> str | None:
        """Provide optional image file path."""
        ...

    @property
    def audio_path(self) -> str | None:
        """Provide optional audio file path."""
        ...


def _print_etl_result(res: DataloaderResultDict) -> None:
    """Print JSON-serializable ETL status summary (excluding live loader instance).

    Args:
        res: `DataloaderResultDict` returned by SDK ETL function.

    """
    summary = {k: v for k, v in res.items() if k != "loader"}
    print(json.dumps(summary, indent=2))


def _format_etl_error(err: GemmaSQLError, config: ETLConfig, backend: str) -> int:
    """Print structured JSON error output for a failed ETL CLI command and return exit code 1.

    Args:
        err: Caught `GemmaSQLError` instance.
        config: `ETLConfig` used for the run.
        backend: Target backend name.

    Returns:
        Exit code `1`.

    """
    return print_cli_error(
        err,
        backend,
        dataset=config.dataset_name,
        split=config.split,
        batch_size=config.batch_size,
    )


def _run_etl_stage(args: ETLArgsProtocol | argparse.Namespace, stage_fn: Callable[..., DataloaderResultDict]) -> int:
    """Construct an `ETLConfig` from CLI args, invoke `stage_fn`, and print the JSON result.

    Args:
        args: Parsed command-line arguments containing command-specific options.
        stage_fn: Target SDK ETL function (`etl_pretrain`, `etl_sft`, or `etl_posttrain`).

    Returns:
        Exit code `0` on success, or `1` if ETL fails.

    """
    modality = str((args.modality if isinstance(args, ETLMultimodalCLIArgsProtocol) else "text") or "text")
    raw_img_path = args.image_path if isinstance(args, ETLMediaPathsCLIArgsProtocol) else None
    raw_aud_path = args.audio_path if isinstance(args, ETLMediaPathsCLIArgsProtocol) else None
    config = ETLConfig(
        dataset_name=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        distributed=args.distributed,
        tokenizer_name=args.tokenizer,
        duckdb_path=args.duckdb_path,
        duckdb_table=args.duckdb_table,
        image_column=args.image_column if isinstance(args, ETLMultimodalCLIArgsProtocol) else None,
        audio_column=args.audio_column if isinstance(args, ETLMultimodalCLIArgsProtocol) else None,
        image_path=None if modality == "audio" else raw_img_path,
        audio_path=None if modality == "vision" else raw_aud_path,
        modality=modality,
    )
    try:
        res = stage_fn(config, backend=args.backend)
    except GemmaSQLError as err:
        return _format_etl_error(err, config, args.backend)
    _print_etl_result(res)
    return 1 if is_failed_status(res) else 0


def etl_pretrain_cmd(args: ETLArgsProtocol | argparse.Namespace) -> int:
    """Run ETL for pretraining.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if ETL fails.

    """
    return _run_etl_stage(args, etl_pretrain)


def etl_sft_cmd(args: ETLArgsProtocol | argparse.Namespace) -> int:
    """Run ETL for SFT.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if ETL fails.

    """
    return _run_etl_stage(args, etl_sft)


def etl_posttrain_cmd(args: ETLArgsProtocol | argparse.Namespace) -> int:
    """Run ETL for post-training.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if ETL fails.

    """
    return _run_etl_stage(args, etl_posttrain)
