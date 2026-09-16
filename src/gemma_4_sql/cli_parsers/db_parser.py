"""CLI argument parsers for database query execution and DuckDB extensions."""

from __future__ import annotations

import argparse

from gemma_4_sql.cli_db import db_execute_cmd, embed_duckdb_cmd


def add_db_parsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register database query execution and DuckDB embedding subparsers.

    Args:
        subparsers: Subparser collection to attach commands to.
    """
    parser_execute = subparsers.add_parser("execute", help="Execute SQL against a live database.")
    parser_execute.add_argument("--query", required=True, help="SQL query to execute.")
    parser_execute.add_argument("--db-path", default=":memory:", help="Path to database.")
    parser_execute.add_argument("--db-type", default="sqlite", help="Type of database.")
    parser_execute.add_argument("--db-kwargs", default="", help="JSON string of DB kwargs.")
    parser_execute.add_argument("--ddl", default="", help="DDL string to initialize the schema.")
    parser_execute.set_defaults(func=db_execute_cmd)

    parser_embed = subparsers.add_parser("embed-duckdb", help="Embed Gemma as a UDF in DuckDB.")
    parser_embed.add_argument("--model", default="gemma-4", help="Model name.")
    parser_embed.add_argument("--db-path", default=":memory:", help="DuckDB database path.")
    parser_embed.add_argument("--prompt", default="", help="Prompt to execute via the UDF.")
    parser_embed.add_argument("--ddl", default="", help="Optional DDL to setup the schema.")
    parser_embed.add_argument("--backend", default="jax", help="Backend to use.")
    parser_embed.add_argument("--max-retries", type=int, default=3, help="Max self-correction attempts.")
    parser_embed.add_argument("--test-mode", action="store_true", help="Run embed-duckdb in fast test mode.")
    parser_embed.set_defaults(func=embed_duckdb_cmd)
