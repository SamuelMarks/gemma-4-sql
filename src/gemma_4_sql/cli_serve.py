"""Serving, Chat, and Evaluation CLI commands."""

from __future__ import annotations

import argparse
import json
import logging

from gemma_4_sql.cli_misc import parse_db_kwargs, parse_json_list, print_cli_error
from gemma_4_sql.exceptions import GemmaSQLError, is_failed_status
from gemma_4_sql.sdk import build_few_shot_prompt, chat_turn, evaluate, generate, run_agentic_loop, serve_model
from gemma_4_sql.sdk.agent import AgentContext
from gemma_4_sql.type_hints import JSONValue

logger = logging.getLogger(__name__)


def _extract_serve_multimodal_kwargs(args: argparse.Namespace) -> dict[str, JSONValue]:
    """Extract optional multimodal arguments (`image_path`, `audio_path`, `modality`) from CLI args.

    Args:
        args: Parsed command-line arguments namespace.

    Returns:
        Dictionary of non-default multimodal keyword arguments.

    """
    extra: dict[str, JSONValue] = {}
    args_dict = vars(args)
    image_path = args.image_path if "image_path" in args_dict else None
    audio_path = args.audio_path if "audio_path" in args_dict else None
    modality = args.modality if "modality" in args_dict else "text"
    if image_path is not None:
        extra["image_path"] = str(image_path)
    if audio_path is not None:
        extra["audio_path"] = str(audio_path)
    if modality and modality != "text":
        extra["modality"] = str(modality)
    return extra


def evaluate_cmd(args: argparse.Namespace) -> int:
    """Evaluate an existing model.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if evaluation fails.

    """
    db_kwargs = parse_db_kwargs(args.db_kwargs)
    if db_kwargs is None:
        return 1
    try:
        res = evaluate(
            model_name=args.model,
            dataset_name=args.dataset,
            backend=args.backend,
            db_path=args.db_path,
            ddl=args.ddl,
            max_batches=args.max_batches,
            db_type=args.db_type,
            db_kwargs=db_kwargs,
            image_path=args.image_path,
            audio_path=args.audio_path,
            modality=args.modality,
        )
    except GemmaSQLError as err:
        return print_cli_error(err, args.backend, model=args.model)
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0


def generate_cmd(args: argparse.Namespace) -> int:
    """Generate SQL from text.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if generation fails.

    """
    try:
        res = generate(
            model_name=args.model,
            prompt=args.prompt,
            backend=args.backend,
            beam_width=args.beam_width,
            max_length=args.max_length,
            show_confidence=args.show_confidence,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            image_path=args.image_path,
            audio_path=args.audio_path,
            modality=args.modality,
        )
    except GemmaSQLError as err:
        return print_cli_error(err, args.backend, model=args.model)
    sql_output = (res or {}).get("sql", "")
    if sql_output:
        print(str(sql_output))
    if args.show_confidence and "confidence_score" in res:
        print(f"Confidence: {float(str(res['confidence_score'])):.4f}")
    return 1 if is_failed_status(res) else 0


def agent_cmd(args: argparse.Namespace) -> int:
    """Run agentic self-correction loop.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if agent loop fails.

    """
    db_kwargs = parse_db_kwargs(args.db_kwargs)
    if db_kwargs is None:
        return 1
    context = AgentContext(
        db_path=args.db_path,
        ddl=args.ddl,
        db_type=args.db_type,
        max_retries=args.max_retries,
        min_confidence=args.min_confidence,
        image_path=args.image_path,
        audio_path=args.audio_path,
        modality=args.modality,
    )
    try:
        res = run_agentic_loop(
            model_name=args.model,
            prompt=args.prompt,
            backend=args.backend,
            context=context,
            db_kwargs=db_kwargs,
        )
    except GemmaSQLError as err:
        return print_cli_error(err, args.backend, model=args.model)
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0


def serve_cmd(args: argparse.Namespace) -> int:
    """Serve a model using continuous batching.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if server initialization fails.

    """
    mm_kwargs = _extract_serve_multimodal_kwargs(args)
    try:
        res = serve_model(
            model_name=args.model,
            port=args.port,
            max_batch_size=args.max_batch_size,
            backend=args.backend,
            run_server=args.run_server,
            **mm_kwargs,
        )
    except GemmaSQLError as err:
        return print_cli_error(err, args.backend, model=args.model)
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
    history = parse_json_list(args.history, "history provided to chat_cmd")
    if history is None:
        return 1
    mm_kwargs = _extract_serve_multimodal_kwargs(args)
    try:
        res = chat_turn(
            model_name=args.model,
            history=history,
            new_prompt=args.prompt,
            backend=args.backend,
            **mm_kwargs,
        )
    except GemmaSQLError as err:
        return print_cli_error(err, args.backend, model=args.model)
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
    examples = parse_json_list(args.examples, "examples provided to few_shot_cmd")
    if examples is None:
        return 1
    args_dict = vars(args)
    try:
        res = build_few_shot_prompt(
            model_name=args.model,
            prompt=args.prompt,
            examples=examples,
            backend=args.backend,
            image_path=args.image_path if "image_path" in args_dict else None,
            audio_path=args.audio_path if "audio_path" in args_dict else None,
            modality=str((args.modality if "modality" in args_dict else "text") or "text"),
        )
    except GemmaSQLError as err:
        return print_cli_error(err, args.backend, model=args.model)
    prompt_text = (res or {}).get("few_shot_prompt", "")
    if prompt_text:
        print(str(prompt_text))
    return 1 if is_failed_status(res) else 0
