"""CLI commands for database execution."""

from __future__ import annotations

import argparse
import json
import logging

from gemma_4_sql.backends.lazy_loader import LazyLoader
from gemma_4_sql.cli_serve import _parse_db_kwargs
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine
from gemma_4_sql.sdk.duckdb_extension import embed_in_duckdb

logger = logging.getLogger(__name__)


def db_execute_cmd(args: argparse.Namespace) -> int:
    """Execute a SQL query against the LiveDatabaseEngine.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, `1` on invalid `--db-kwargs` JSON, or `3` on database failure.

    """
    parsed_kwargs = _parse_db_kwargs(args.db_kwargs)
    if parsed_kwargs is None:
        return 1
    db_kwargs: dict[str, object] = dict(parsed_kwargs)
    engine = LiveDatabaseEngine(db_path=args.db_path, ddl=args.ddl, db_type=args.db_type, db_kwargs=db_kwargs)
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
    """Embed Gemma as a UDF in DuckDB and execute a prompt.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if DuckDB is unavailable.

    """
    duckdb = LazyLoader("duckdb").get_module()
    if duckdb is None:
        return 1
    conn = duckdb.connect(args.db_path)
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
    conn.close()
    return 0
