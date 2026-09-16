"""CLI argument parsers for serving, generation, chat, evaluation, and agents."""

from __future__ import annotations

import argparse

from gemma_4_sql.cli_misc import export_cmd
from gemma_4_sql.cli_serve import agent_cmd, chat_cmd, evaluate_cmd, few_shot_cmd, generate_cmd, serve_cmd


def add_evaluate_parsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register evaluation, few-shot prompting, and conversational chat subparsers.

    Args:
        subparsers: Subparser collection to attach commands to.
    """
    parser_evaluate = subparsers.add_parser("evaluate", help="Evaluate a trained model.")
    parser_evaluate.add_argument("--model", default="gemma-4", help="Model name.")
    parser_evaluate.add_argument("--dataset", default="test-data", help="Dataset to evaluate on.")
    parser_evaluate.add_argument("--backend", default="jax", help="Backend to use.")
    parser_evaluate.add_argument("--db-path", default=":memory:", help="Path to SQLite db for evaluation.")
    parser_evaluate.add_argument("--db-type", default="sqlite", help="Type of database backend.")
    parser_evaluate.add_argument("--db-kwargs", default="", help="JSON string of DB kwargs.")
    parser_evaluate.add_argument("--ddl", default="", help="DDL string to setup the evaluation schema.")
    parser_evaluate.add_argument("--image-path", default=None, help="Path to schema diagram, ERD image, or screenshot.")
    parser_evaluate.add_argument("--audio-path", default=None, help="Path to recorded query audio file.")
    parser_evaluate.add_argument("--modality", default="text", choices=["text", "vision", "audio", "multimodal"], help="Explicit modality selector.")
    parser_evaluate.set_defaults(func=evaluate_cmd)

    parser_few_shot = subparsers.add_parser("few-shot", help="Build a dynamic few-shot prompt.")
    parser_few_shot.add_argument("--model", default="gemma-4", help="Model name.")
    parser_few_shot.add_argument("--prompt", required=True, help="New user prompt.")
    parser_few_shot.add_argument("--examples", default="[]", help="JSON string representing few-shot examples.")
    parser_few_shot.add_argument("--backend", default="jax", help="Backend to use.")
    parser_few_shot.add_argument("--image-path", default=None, help="Path to schema diagram, ERD image, or screenshot.")
    parser_few_shot.add_argument("--audio-path", default=None, help="Path to recorded query audio file.")
    parser_few_shot.add_argument("--modality", default="text", choices=["text", "vision", "audio", "multimodal"], help="Explicit modality selector.")
    parser_few_shot.set_defaults(func=few_shot_cmd)

    parser_chat = subparsers.add_parser("chat", help="Execute a turn in a multi-turn conversational SQL chat.")
    parser_chat.add_argument("--model", default="gemma-4", help="Model name.")
    parser_chat.add_argument("--prompt", required=True, help="New user prompt.")
    parser_chat.add_argument("--history", default="[]", help="JSON string representing history.")
    parser_chat.add_argument("--backend", default="jax", help="Backend to use.")
    parser_chat.add_argument("--image-path", default=None, help="Path to schema diagram, ERD image, or screenshot.")
    parser_chat.add_argument("--audio-path", default=None, help="Path to recorded query audio file.")
    parser_chat.add_argument("--modality", default="text", choices=["text", "vision", "audio", "multimodal"], help="Explicit modality selector.")
    parser_chat.set_defaults(func=chat_cmd)


def add_serve_export_parsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register HTTP continuous batching serving and model export parsers.

    Args:
        subparsers: Subparser collection to attach commands to.
    """
    parser_serve = subparsers.add_parser("serve", help="Serve a model using continuous batching.")
    parser_serve.add_argument("--model", default="gemma-4", help="Model name.")
    parser_serve.add_argument("--port", type=int, default=8000, help="Port to bind to.")
    parser_serve.add_argument("--max-batch-size", type=int, default=256, help="Maximum batch size.")
    parser_serve.add_argument("--run-server", action="store_true", help="Start the uvicorn HTTP server.")
    parser_serve.add_argument("--backend", default="pytorch", help="Backend to use.")
    parser_serve.add_argument("--modality", default="text", choices=["text", "vision", "audio", "multimodal"], help="Explicit modality selector.")
    parser_serve.set_defaults(func=serve_cmd)

    parser_export = subparsers.add_parser("export", help="Export and save a trained model.")
    parser_export.add_argument("--model", default="gemma-4", help="Model name.")
    parser_export.add_argument("--path", default="./checkpoints", help="Export destination path.")
    parser_export.add_argument("--backend", default="jax", help="Backend to use.")
    parser_export.set_defaults(func=export_cmd)


def add_generate_agent_parsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register SQL generation and self-correction agent parsers.

    Args:
        subparsers: Subparser collection to attach commands to.
    """
    parser_generate = subparsers.add_parser("generate", help="Generate SQL from text using a trained model.")
    parser_generate.add_argument("--model", default="gemma-4", help="Model name.")
    parser_generate.add_argument("--prompt", required=True, help="Natural language prompt.")
    parser_generate.add_argument("--backend", default="jax", help="Backend to use.")
    parser_generate.add_argument("--beam-width", type=int, default=3, help="Number of beams for generation.")
    parser_generate.add_argument("--max-length", type=int, default=50, help="Maximum generation length.")
    parser_generate.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser_generate.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling threshold.")
    parser_generate.add_argument("--seed", type=int, default=42, help="Random seed for generation.")
    parser_generate.add_argument("--test-mode", action="store_true", help="Run generation in fast test mode.")
    parser_generate.add_argument("--show-confidence", action="store_true", help="Display the model's confidence score.")
    parser_generate.add_argument("--image-path", default=None, help="Path to schema diagram, ERD image, or screenshot.")
    parser_generate.add_argument("--audio-path", default=None, help="Path to recorded query audio file.")
    parser_generate.add_argument("--modality", default="text", choices=["text", "vision", "audio", "multimodal"], help="Explicit modality selector.")
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
    parser_agent.add_argument("--test-mode", action="store_true", help="Run agent loop in fast test mode.")
    parser_agent.add_argument("--image-path", default=None, help="Path to schema diagram, ERD image, or screenshot.")
    parser_agent.add_argument("--audio-path", default=None, help="Path to recorded query audio file.")
    parser_agent.add_argument("--modality", default="text", choices=["text", "vision", "audio", "multimodal"], help="Explicit modality selector.")
    parser_agent.set_defaults(func=agent_cmd)


def add_inference_parsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register serving, export, generation, and agentic loop subparsers.

    Args:
        subparsers: Subparser collection to attach commands to.
    """
    add_serve_export_parsers(subparsers)
    add_generate_agent_parsers(subparsers)
