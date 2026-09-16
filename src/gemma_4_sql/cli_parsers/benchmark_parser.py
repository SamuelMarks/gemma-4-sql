"""CLI argument parsers for benchmarking, tokenization, RAG contextualization, and logging."""

from __future__ import annotations

import argparse

from gemma_4_sql.cli_benchmark import benchmark_cmd
from gemma_4_sql.cli_misc import log_metrics_cmd, rag_cmd, tokenize_cmd
from gemma_4_sql.cli_parsers.db_parser import add_db_parsers


def add_tokenize_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register tokenization inspection subparser.

    Args:
        subparsers: Subparser collection to attach commands to.
    """
    parser_tokenize = subparsers.add_parser("tokenize", help="Encode or decode text using SQLTokenizer.")
    parser_tokenize.add_argument("--encode", type=str, help="Text to encode.")
    parser_tokenize.add_argument("--decode", type=str, help="Comma-separated tokens to decode.")
    parser_tokenize.add_argument("--hf-model", type=str, default=None, help="Hugging Face model name.")
    parser_tokenize.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size.")
    parser_tokenize.set_defaults(func=tokenize_cmd)


def add_rag_log_parsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register RAG prompt builder and metric logging subparsers.

    Args:
        subparsers: Subparser collection to attach commands to.
    """
    parser_rag = subparsers.add_parser("rag", help="Build a RAG prompt or extract schema context.")
    parser_rag.add_argument("--action", default="build", choices=["build", "extract", "retrieve"], help="Action to perform.")
    parser_rag.add_argument("--prompt", default="", help="Natural language prompt.")
    parser_rag.add_argument("--ddl", required=True, help="DDL string to extract schema context from.")
    parser_rag.add_argument("--image-path", default=None, help="Path to schema diagram, ERD image, or table screenshot.")
    parser_rag.add_argument("--audio-path", default=None, help="Path to recorded natural language query audio file.")
    parser_rag.add_argument("--modality", default="text", choices=["text", "vision", "audio", "multimodal"], help="Explicit modality selector.")
    parser_rag.set_defaults(func=rag_cmd)

    parser_log = subparsers.add_parser("log", help="Log metrics to the backend.")
    parser_log.add_argument("--step", type=int, default=0, help="Training step.")
    parser_log.add_argument("--metrics", default="", help="Comma separated key=value metrics.")
    parser_log.add_argument("--log-dir", default="logs", help="Directory to save TensorBoard logs.")
    parser_log.add_argument("--backend", default="jax", help="Backend to use.")
    parser_log.set_defaults(func=log_metrics_cmd)


def add_benchmark_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register hardware throughput and latency benchmark subparser.

    Args:
        subparsers: Subparser collection to attach commands to.
    """
    parser_benchmark = subparsers.add_parser("benchmark", help="Benchmark a model on target hardware.")
    parser_benchmark.add_argument("--model", default="gemma-4", help="Model name.")
    parser_benchmark.add_argument("--hardware", default="gpu", choices=["gpu", "tpu", "cpu"], help="Target hardware.")
    parser_benchmark.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmark.")
    parser_benchmark.add_argument("--backend", default="jax", help="Backend to use.")
    parser_benchmark.add_argument("--dtype", default="bfloat16", help="Precision to use.")
    parser_benchmark.add_argument("--mode", default="prefill", choices=["prefill", "decode", "end-to-end"], help="Benchmark mode.")
    parser_benchmark.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens for generation.")
    parser_benchmark.add_argument("--warmup-steps", type=int, default=5, help="Number of warmup steps.")
    parser_benchmark.add_argument("--image-path", default=None, help="Path to schema diagram, ERD image, or table screenshot.")
    parser_benchmark.add_argument("--audio-path", default=None, help="Path to recorded natural language query audio file.")
    parser_benchmark.add_argument("--modality", default="text", choices=["text", "vision", "audio", "multimodal"], help="Explicit modality selector.")
    parser_benchmark.set_defaults(func=benchmark_cmd)


def add_misc_parsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register tokenization, database, benchmark, RAG, and logging subparsers.

    Args:
        subparsers: Subparser collection to attach commands to.
    """
    add_tokenize_parser(subparsers)
    add_db_parsers(subparsers)
    add_benchmark_parser(subparsers)
    add_rag_log_parsers(subparsers)
