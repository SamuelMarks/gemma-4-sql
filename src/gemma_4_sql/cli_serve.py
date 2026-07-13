"""Serving, Chat, and Evaluation CLI commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from gemma_4_sql.sdk import build_few_shot_prompt, chat_turn, evaluate, generate, run_agentic_loop, serve_model

if TYPE_CHECKING:
    import argparse


def evaluate_cmd(args: argparse.Namespace) -> None:
    """Evaluate an existing model.


    Args:
        args: Parsed command-line arguments containing command-specific options."""
    db_kwargs = {}
    if args.db_kwargs:  # pragma: no cover
        json = __import__("json")
        db_kwargs = json.loads(args.db_kwargs)
    evaluate(model_name=args.model, dataset_name=args.dataset, backend=args.backend, db_path=args.db_path, ddl=args.ddl, db_type=args.db_type, db_kwargs=db_kwargs)


def generate_cmd(args: argparse.Namespace) -> None:
    """Generate SQL from text.


    Args:
        args: Parsed command-line arguments containing command-specific options."""
    generate(model_name=args.model, prompt=args.prompt, backend=args.backend, beam_width=args.beam_width, max_length=args.max_length, show_confidence=getattr(args, "show_confidence", False))


def agent_cmd(args: argparse.Namespace) -> None:
    """Run agentic self-correction loop.

    Args:
        args: Parsed command-line arguments containing command-specific options."""
    db_kwargs = {}
    if args.db_kwargs:  # pragma: no cover
        json = __import__("json")
        db_kwargs = json.loads(args.db_kwargs)
    agent_context_cls = __import__("gemma_4_sql.sdk.agent", fromlist=["AgentContext"]).AgentContext
    context = agent_context_cls(db_path=args.db_path, ddl=args.ddl, db_type=args.db_type, max_retries=args.max_retries, min_confidence=getattr(args, "min_confidence", 0.0))
    run_agentic_loop(model_name=args.model, prompt=args.prompt, backend=args.backend, context=context, db_kwargs=db_kwargs)


def serve_cmd(args: argparse.Namespace) -> None:
    """Serve a model using continuous batching.


    Args:
        args: Parsed command-line arguments containing command-specific options."""
    serve_model(model_name=args.model, port=args.port, max_batch_size=args.max_batch_size, backend=args.backend)


def chat_cmd(args: argparse.Namespace) -> None:
    """Run a multi-turn conversational SQL chat turn.


    Args:
        args: Parsed command-line arguments containing command-specific options."""
    history = []
    if getattr(args, "history", ""):  # pragma: no cover
        try:
            history = json.loads(args.history)
        except json.JSONDecodeError:
            return
    chat_turn(model_name=args.model, history=history, new_prompt=args.prompt, backend=args.backend)


def few_shot_cmd(args: argparse.Namespace) -> None:
    """Run dynamic few-shot prompting.

    Args:
        args: Parsed command-line arguments containing command-specific options."""
    examples = []
    if getattr(args, "examples", ""):  # pragma: no cover  # pragma: no cover
        try:
            examples = json.loads(args.examples)
        except json.JSONDecodeError:
            return
    build_few_shot_prompt(model_name=args.model, prompt=args.prompt, examples=examples, backend=args.backend)
