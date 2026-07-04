# Copyright 2024
"""CLI commands for database execution."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.lazy_loader import LazyLoader
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine
from gemma_4_sql.sdk.duckdb_extension import embed_in_duckdb

if TYPE_CHECKING:
    import argparse
logger = logging.getLogger(__name__)


def db_execute_cmd(args: argparse.Namespace) -> None:
    """Execute a SQL query against the LiveDatabaseEngine."""
    db_kwargs = {}
    if getattr(args, "db_kwargs", ""):
        db_kwargs = json.loads(args.db_kwargs)
    engine = LiveDatabaseEngine(db_path=args.db_path, ddl=args.ddl, db_type=args.db_type, db_kwargs=db_kwargs)
    (_success, _results, _error) = engine.execute_with_feedback(args.query)
    engine.close()


def embed_duckdb_cmd(args: argparse.Namespace) -> None:
    """Embed Gemma as a UDF in DuckDB and execute a prompt."""
    duckdb = LazyLoader("duckdb").get_module()
    if duckdb is None:
        return
    conn = duckdb.connect(args.db_path)
    if args.ddl:
        conn.execute(args.ddl)
    embed_in_duckdb(conn=conn, model_name=args.model, backend=args.backend, db_path=args.db_path, max_retries=args.max_retries)
    if args.prompt:
        conn.execute("SELECT ask_gemma(?)", [args.prompt]).fetchall()
    conn.close()
