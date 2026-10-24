"""Serving, Chat, and Evaluation CLI commands."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.exceptions import is_failed_status
from gemma_4_sql.sdk import build_few_shot_prompt, chat_turn, evaluate, generate, run_agentic_loop, serve_model
from gemma_4_sql.sdk.agent import AgentContext

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)


def evaluate_cmd(args: argparse.Namespace) -> int:
    """Evaluate an existing model.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if evaluation fails.

    """
    db_kwargs: dict[str, Any] = {}
    if args.db_kwargs:
        db_kwargs = json.loads(args.db_kwargs)
    res = evaluate(
        model_name=args.model,
        dataset_name=args.dataset,
        backend=args.backend,
        db_path=args.db_path,
        ddl=args.ddl,
        max_batches=getattr(args, "max_batches", 10),
        db_type=args.db_type,
        db_kwargs=db_kwargs,
        image_path=getattr(args, "image_path", None),
        audio_path=getattr(args, "audio_path", None),
        modality=getattr(args, "modality", "text"),
    )
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0


def generate_cmd(args: argparse.Namespace) -> int:
    """Generate SQL from text.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if generation fails.

    """
    res = generate(
        model_name=args.model,
        prompt=args.prompt,
        backend=args.backend,
        beam_width=args.beam_width,
        max_length=args.max_length,
        show_confidence=getattr(args, "show_confidence", False),
        temperature=getattr(args, "temperature", 1.0),
        top_p=getattr(args, "top_p", 1.0),
        seed=getattr(args, "seed", 42),
        image_path=getattr(args, "image_path", None),
        audio_path=getattr(args, "audio_path", None),
        modality=getattr(args, "modality", "text"),
    )
    sql_output = (res or {}).get("sql", "")
    if sql_output:
        print(str(sql_output))
    if getattr(args, "show_confidence", False) and res and "confidence_score" in res:
        print(f"Confidence: {float(str(res['confidence_score'])):.4f}")
    return 1 if is_failed_status(res) else 0


def agent_cmd(args: argparse.Namespace) -> int:
    """Run agentic self-correction loop.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if agent loop fails.

    """
    db_kwargs: dict[str, Any] = {}
    if args.db_kwargs:
        db_kwargs = json.loads(args.db_kwargs)
    context = AgentContext(
        db_path=args.db_path,
        ddl=args.ddl,
        db_type=args.db_type,
        max_retries=args.max_retries,
        min_confidence=getattr(args, "min_confidence", 0.0),
        image_path=getattr(args, "image_path", None),
        audio_path=getattr(args, "audio_path", None),
        modality=getattr(args, "modality", "text"),
    )
    res = run_agentic_loop(
        model_name=args.model,
        prompt=args.prompt,
        backend=args.backend,
        context=context,
        db_kwargs=db_kwargs,
    )
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0


def serve_cmd(args: argparse.Namespace) -> int:
    """Serve a model using continuous batching.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if server initialization fails.

    """
    res = serve_model(
        model_name=args.model,
        port=args.port,
        max_batch_size=args.max_batch_size,
        backend=args.backend,
        run_server=getattr(args, "run_server", False),
    )
    status = (res or {}).get("status", "unknown")
    print(f"Serving model {args.model} on port {args.port} (status: {status})")
    return 1 if is_failed_status(res) else 0


def chat_cmd(args: argparse.Namespace) -> int:
    """Run a multi-turn conversational SQL chat turn.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if chat turn fails.

    """
    history: list[dict[str, str]] = []
    if getattr(args, "history", ""):
        try:
            history = json.loads(args.history)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON history provided to chat_cmd: %s", exc)
            return 0
    res = chat_turn(
        model_name=args.model,
        history=history,
        new_prompt=args.prompt,
        backend=args.backend,
    )
    response_text = (res or {}).get("response", "")
    if response_text:
        print(str(response_text))
    return 1 if is_failed_status(res) else 0


def few_shot_cmd(args: argparse.Namespace) -> int:
    """Run dynamic few-shot prompting.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if few-shot prompt generation fails.

    """
    examples: list[dict[str, str]] = []
    if getattr(args, "examples", ""):
        try:
            examples = json.loads(args.examples)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON examples provided to few_shot_cmd: %s", exc)
            return 0
    res = build_few_shot_prompt(
        model_name=args.model,
        prompt=args.prompt,
        examples=examples,
        backend=args.backend,
    )
    prompt_text = (res or {}).get("few_shot_prompt", "")
    if prompt_text:
        print(str(prompt_text))
    return 1 if is_failed_status(res) else 0
