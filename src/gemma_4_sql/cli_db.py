"""CLI commands for database execution and DuckDB UDF embedding."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.cli_misc import parse_db_kwargs as _parse_db_kwargs
from gemma_4_sql.exceptions import GemmaSQLError
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine
from gemma_4_sql.sdk.duckdb_extension import DuckDBUDFConnProtocol, embed_in_duckdb
from gemma_4_sql.type_hints import JSONValue

logger = logging.getLogger(__name__)


@runtime_checkable
class _ClosableDuckDBConnProtocol(DuckDBUDFConnProtocol, Protocol):
    """Protocol for a DuckDB connection supporting UDF registration and `.close()`."""

    def close(self) -> None:
        """Close the DuckDB connection."""
        ...


@runtime_checkable
class _DuckDBConnectModuleProtocol(Protocol):
    """Protocol for `duckdb` module exposing `connect` and `Error`."""

    @property
    def Error(self) -> type[Exception] | int | str | None:
        """Provide `duckdb.Error` exception class."""
        ...

    def connect(self, database: str = ":memory:") -> _ClosableDuckDBConnProtocol:
        """Open a DuckDB connection.

        Args:
            database: Database file path or `':memory:'`.

        Returns:
            Opened DuckDB connection.

        """
        ...


def db_execute_cmd(args: argparse.Namespace) -> int:
    """Execute a SQL query against the `LiveDatabaseEngine`.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, `1` on invalid `--db-kwargs` JSON or configuration error, or `3` on database failure.

    """
    parsed_kwargs = _parse_db_kwargs(args.db_kwargs)
    if parsed_kwargs is None:
        return 1
    db_kwargs: dict[str, JSONValue] = dict(parsed_kwargs)
    try:
        engine = LiveDatabaseEngine(db_path=args.db_path, ddl=args.ddl, db_type=args.db_type, db_kwargs=db_kwargs)
    except GemmaSQLError as err:
        print(json.dumps({"success": False, "results": [], "error": str(err)}, indent=2))
        return 1
    (success, results, error) = engine.execute_with_feedback(args.query)
    engine.close()
    output = {
        "success": success,
        "results": results,
        "error": error,
    }
    print(json.dumps(output, indent=2))
    return 0 if success else 3


def embed_duckdb_cmd(args: argparse.Namespace) -> int:
    """Embed Gemma as a UDF in DuckDB and execute an optional prompt safely.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if DuckDB is unavailable or DDL/query execution fails.

    """
    duckdb_mod = optional_import("duckdb", _DuckDBConnectModuleProtocol)
    if duckdb_mod is None:
        return 1
    err_attr = duckdb_mod.Error
    duckdb_err: type[Exception] = (
        err_attr if err_attr is not None and not isinstance(err_attr, (int, str)) else RuntimeError
    )
    try:
        conn = duckdb_mod.connect(args.db_path)
    except Exception as err:
        if not isinstance(err, (duckdb_err, RuntimeError, OSError)):
            raise
        print(json.dumps({"success": False, "error": str(err)}, indent=2))
        return 1

    try:
        if args.ddl:
            conn.execute(args.ddl)
        embed_in_duckdb(
            conn=conn,
            model_name=args.model,
            backend=args.backend,
            db_path=args.db_path,
            max_retries=args.max_retries,
        )
        if args.prompt:
            rows = conn.execute("SELECT ask_gemma(?)", [args.prompt]).fetchall()
            for row in rows:
                print(str(row[0]))
    except Exception as err:
        if not isinstance(err, (duckdb_err, GemmaSQLError, RuntimeError, ValueError, OSError)):
            raise
        print(json.dumps({"success": False, "error": str(err)}, indent=2))
        return 1
    finally:
        conn.close()
    return 0
