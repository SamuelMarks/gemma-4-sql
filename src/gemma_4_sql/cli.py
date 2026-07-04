# Copyright 2024
"""Main CLI entrypoint for gemma-4-sql."""

from __future__ import annotations

import argparse

from gemma_4_sql.cli_benchmark import benchmark_cmd
from gemma_4_sql.cli_db import db_execute_cmd, embed_duckdb_cmd
from gemma_4_sql.cli_etl import etl_posttrain_cmd, etl_pretrain_cmd, etl_sft_cmd
from gemma_4_sql.cli_misc import export_cmd, log_metrics_cmd, quantize_cmd, rag_cmd, tokenize_cmd
from gemma_4_sql.cli_serve import agent_cmd, chat_cmd, evaluate_cmd, few_shot_cmd, generate_cmd, serve_cmd
from gemma_4_sql.cli_train import dpo_cmd, peft_cmd, posttrain_cmd, pretrain_cmd, sft_cmd, train_cmd
from gemma_4_sql.constants import DEFAULT_POSTTRAIN_DATASET, DEFAULT_PRETRAIN_DATASET, DEFAULT_SFT_DATASET


def _add_etl_subparser(subparsers: argparse._SubParsersAction, name: str, help_text: str, default_dataset: str, cmd_func: object) -> None:
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("--dataset", default=default_dataset, help="Hugging Face dataset name.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--backend", default="jax", help="Backend to use.")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed sharding.")
    parser.add_argument("--tokenizer", default=None, help="Hugging Face tokenizer model name.")
    parser.add_argument("--duckdb-path", default=None, help="Optional path to DuckDB database.")
    parser.add_argument("--duckdb-table", default=None, help="Optional DuckDB table name.")
    parser.set_defaults(func=cmd_func)


def _add_etl_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Execute the add etl parsers operation."""
    parser_etl = subparsers.add_parser("etl", help="Run ETL to prepare SQL training datasets.")
    etl_subparsers = parser_etl.add_subparsers(dest="etl_command", required=True)
    _add_etl_subparser(etl_subparsers, "pretrain", "Run ETL for pretraining SQL datasets.", DEFAULT_PRETRAIN_DATASET, etl_pretrain_cmd)
    _add_etl_subparser(etl_subparsers, "sft", "Run ETL for SFT SQL datasets.", DEFAULT_SFT_DATASET, etl_sft_cmd)
    _add_etl_subparser(etl_subparsers, "posttrain", "Run ETL for post-training SQL datasets.", DEFAULT_POSTTRAIN_DATASET, etl_posttrain_cmd)


def _add_peft_quantize_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Execute the add peft quantize parsers operation."""
    parser_dpo = subparsers.add_parser("dpo", help="Run Direct Preference Optimization (DPO).")
    parser_dpo.add_argument("--model", default="gemma-4", help="Model name.")
    parser_dpo.add_argument("--dataset", default="dummy_dataset", help="Training dataset.")
    parser_dpo.add_argument("--beta", type=float, default=0.1, help="DPO temperature parameter.")
    parser_dpo.add_argument("--backend", default="jax", help="Backend to use.")
    parser_dpo.set_defaults(func=dpo_cmd)
    parser_peft = subparsers.add_parser("peft", help="Apply PEFT / LoRA configuration to a model.")
    parser_peft.add_argument("--model", default="gemma-4", help="Model name.")
    parser_peft.add_argument("--target-modules", default="q_proj,v_proj", help="Target modules.")
    parser_peft.add_argument("--lora-r", type=int, default=8, help="LoRA attention dimension.")
    parser_peft.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha parameter.")
    parser_peft.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout probability.")
    parser_peft.add_argument("--backend", default="jax", help="Backend to use.")
    parser_peft.set_defaults(func=peft_cmd)
    parser_quantize = subparsers.add_parser("quantize", help="Quantize a model.")
    parser_quantize.add_argument("--model", default="gemma-4", help="Model name.")
    parser_quantize.add_argument("--method", default="int8", choices=["int8", "awq", "gptq", "gguf"], help="Quantization method.")
    parser_quantize.add_argument("--backend", default="pytorch", help="Backend to use.")
    parser_quantize.set_defaults(func=quantize_cmd)


def _add_training_subparser(subparsers: argparse._SubParsersAction, name: str, help_text: str, backend: str, cmd_func: object) -> None:
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("--model", default="gemma-4", help="Model name.")
    parser.add_argument("--dataset", default="dummy_dataset", help="Training dataset.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--backend", default=backend, help="Backend to use.")
    parser.add_argument("--distributed-strategy", default="none", choices=["none", "ddp", "fsdp"], help="Distributed strategy.")
    parser.set_defaults(func=cmd_func)


def _add_training_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Execute the add training parsers operation."""
    _add_training_subparser(subparsers, "train", "Train a new model from scratch.", "jax", train_cmd)
    _add_training_subparser(subparsers, "pretrain", "Pretrain an existing model.", "maxtext", pretrain_cmd)
    _add_training_subparser(subparsers, "sft", "Supervised fine-tune an existing model.", "jax", sft_cmd)
    _add_training_subparser(subparsers, "posttrain", "Post-train an existing model.", "keras", posttrain_cmd)
    _add_peft_quantize_parsers(subparsers)


def _add_evaluate_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Execute the add evaluate parsers operation."""
    parser_evaluate = subparsers.add_parser("evaluate", help="Evaluate a trained model.")
    parser_evaluate.add_argument("--model", default="gemma-4", help="Model name.")
    parser_evaluate.add_argument("--dataset", default="test-data", help="Dataset to evaluate on.")
    parser_evaluate.add_argument("--backend", default="jax", help="Backend to use.")
    parser_evaluate.add_argument("--db-path", default=":memory:", help="Path to SQLite db for evaluation.")
    parser_evaluate.add_argument("--db-type", default="sqlite", help="Type of database backend.")
    parser_evaluate.add_argument("--db-kwargs", default="", help="JSON string of DB kwargs.")
    parser_evaluate.add_argument("--ddl", default="", help="DDL string to setup the evaluation schema.")
    parser_evaluate.add_argument("--predictions", default=None, help="Semicolon separated mock predictions.")
    parser_evaluate.add_argument("--truths", default=None, help="Semicolon separated mock truths.")
    parser_evaluate.set_defaults(func=evaluate_cmd)
    parser_few_shot = subparsers.add_parser("few-shot", help="Build a dynamic few-shot prompt.")
    parser_few_shot.add_argument("--model", default="gemma-4", help="Model name.")
    parser_few_shot.add_argument("--prompt", required=True, help="New user prompt.")
    parser_few_shot.add_argument("--examples", default="[]", help="JSON string representing few-shot examples.")
    parser_few_shot.add_argument("--backend", default="jax", help="Backend to use.")
    parser_few_shot.set_defaults(func=few_shot_cmd)
    parser_chat = subparsers.add_parser("chat", help="Execute a turn in a multi-turn conversational SQL chat.")
    parser_chat.add_argument("--model", default="gemma-4", help="Model name.")
    parser_chat.add_argument("--prompt", required=True, help="New user prompt.")
    parser_chat.add_argument("--history", default="[]", help="JSON string representing history.")
    parser_chat.add_argument("--backend", default="jax", help="Backend to use.")
    parser_chat.set_defaults(func=chat_cmd)


def _add_serve_export_parsers(subparsers: argparse._SubParsersAction) -> None:
    parser_serve = subparsers.add_parser("serve", help="Serve a model using continuous batching.")
    parser_serve.add_argument("--model", default="gemma-4", help="Model name.")
    parser_serve.add_argument("--port", type=int, default=8000, help="Port to bind to.")
    parser_serve.add_argument("--max-batch-size", type=int, default=256, help="Maximum batch size.")
    parser_serve.add_argument("--backend", default="pytorch", help="Backend to use.")
    parser_serve.set_defaults(func=serve_cmd)

    parser_export = subparsers.add_parser("export", help="Export and save a trained model.")
    parser_export.add_argument("--model", default="gemma-4", help="Model name.")
    parser_export.add_argument("--path", default="./checkpoints", help="Export destination path.")
    parser_export.add_argument("--backend", default="jax", help="Backend to use.")
    parser_export.set_defaults(func=export_cmd)


def _add_generate_agent_parsers(subparsers: argparse._SubParsersAction) -> None:
    parser_generate = subparsers.add_parser("generate", help="Generate SQL from text using a trained model.")
    parser_generate.add_argument("--model", default="gemma-4", help="Model name.")
    parser_generate.add_argument("--prompt", required=True, help="Natural language prompt.")
    parser_generate.add_argument("--backend", default="jax", help="Backend to use.")
    parser_generate.add_argument("--beam-width", type=int, default=3, help="Number of beams for generation.")
    parser_generate.add_argument("--max-length", type=int, default=50, help="Maximum generation length.")
    parser_generate.add_argument("--show-confidence", action="store_true", help="Display the model's confidence score.")
    parser_generate.set_defaults(func=generate_cmd)

    parser_agent = subparsers.add_parser("agent", help="Run agentic self-correction loop.")
    parser_agent.add_argument("--model", default="gemma-4", help="Model name.")
    parser_agent.add_argument("--prompt", required=True, help="Natural language prompt.")
    parser_agent.add_argument("--db-path", default=":memory:", help="Path to database for execution.")
    parser_agent.add_argument("--db-type", default="sqlite", help="Type of database backend.")
    parser_agent.add_argument("--db-kwargs", default="", help="JSON string of DB kwargs.")
    parser_agent.add_argument("--ddl", default="", help="DDL string to setup the evaluation schema.")
    parser_agent.add_argument("--max-retries", type=int, default=3, help="Max retries.")
    parser_agent.add_argument("--min-confidence", type=float, default=0.0, help="Min confidence.")
    parser_agent.add_argument("--backend", default="jax", help="Backend to use.")
    parser_agent.set_defaults(func=agent_cmd)


def _add_inference_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Execute the add inference parsers operation."""
    _add_serve_export_parsers(subparsers)
    _add_generate_agent_parsers(subparsers)

    parser_rag = subparsers.add_parser("rag", help="Build a RAG prompt or extract schema context.")
    parser_rag.add_argument("--action", default="build", choices=["build", "extract", "retrieve"], help="Action to perform.")
    parser_rag.add_argument("--prompt", default="", help="Natural language prompt.")
    parser_rag.add_argument("--ddl", required=True, help="DDL string to extract schema context from.")
    parser_rag.set_defaults(func=rag_cmd)

    parser_log = subparsers.add_parser("log", help="Log metrics to the backend.")
    parser_log.add_argument("--step", type=int, default=0, help="Training step.")
    parser_log.add_argument("--metrics", default="", help="Comma separated key=value metrics.")
    parser_log.add_argument("--log-dir", default="logs", help="Directory to save TensorBoard logs.")
    parser_log.add_argument("--backend", default="jax", help="Backend to use.")
    parser_log.set_defaults(func=log_metrics_cmd)


def _add_tokenize_execute_parsers(subparsers: argparse._SubParsersAction) -> None:
    parser_tokenize = subparsers.add_parser("tokenize", help="Encode or decode text using SQLTokenizer.")
    parser_tokenize.add_argument("--encode", type=str, help="Text to encode.")
    parser_tokenize.add_argument("--decode", type=str, help="Comma-separated tokens to decode.")
    parser_tokenize.add_argument("--hf-model", type=str, default=None, help="Hugging Face model name.")
    parser_tokenize.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size.")
    parser_tokenize.set_defaults(func=tokenize_cmd)

    parser_execute = subparsers.add_parser("execute", help="Execute SQL against a live database.")
    parser_execute.add_argument("--query", required=True, help="SQL query to execute.")
    parser_execute.add_argument("--db-path", default=":memory:", help="Path to database.")
    parser_execute.add_argument("--db-type", default="sqlite", help="Type of database.")
    parser_execute.add_argument("--db-kwargs", default="", help="JSON string of DB kwargs.")
    parser_execute.add_argument("--ddl", default="", help="DDL string to initialize the schema.")
    parser_execute.set_defaults(func=db_execute_cmd)


def _add_misc_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Execute the add misc parsers operation."""
    _add_tokenize_execute_parsers(subparsers)

    parser_embed = subparsers.add_parser("embed-duckdb", help="Embed Gemma as a UDF in DuckDB.")
    parser_embed.add_argument("--model", default="gemma-4", help="Model name.")
    parser_embed.add_argument("--db-path", default=":memory:", help="DuckDB database path.")
    parser_embed.add_argument("--prompt", default="", help="Prompt to execute via the UDF.")
    parser_embed.add_argument("--ddl", default="", help="Optional DDL to setup the schema.")
    parser_embed.add_argument("--backend", default="jax", help="Backend to use.")
    parser_embed.add_argument("--max-retries", type=int, default=3, help="Max self-correction attempts.")
    parser_embed.set_defaults(func=embed_duckdb_cmd)

    parser_benchmark = subparsers.add_parser("benchmark", help="Benchmark a model on target hardware.")
    parser_benchmark.add_argument("--model", default="gemma-4", help="Model name.")
    parser_benchmark.add_argument("--hardware", default="gpu", choices=["gpu", "tpu", "cpu"], help="Target hardware.")
    parser_benchmark.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmark.")
    parser_benchmark.add_argument("--backend", default="jax", help="Backend to use.")
    parser_benchmark.set_defaults(func=benchmark_cmd)


def cli(args: list[str] | None = None) -> None:
    """Run main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="CLI for gemma-4-sql dataset generation and model training.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_etl_parsers(subparsers)
    _add_training_parsers(subparsers)
    _add_evaluate_parsers(subparsers)
    _add_inference_parsers(subparsers)
    _add_misc_parsers(subparsers)
    parsed_args = parser.parse_args(args)
    parsed_args.func(parsed_args)


if __name__ == "__main__":
    cli()
