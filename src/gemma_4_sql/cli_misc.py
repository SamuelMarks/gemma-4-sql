"""CLI commands for miscellaneous tasks."""

from __future__ import annotations

import argparse
import json
import logging

from gemma_4_sql.exceptions import ExecutionStatus, GemmaSQLError, is_failed_status
from gemma_4_sql.sdk.export import export_model
from gemma_4_sql.sdk.logging import log_metrics
from gemma_4_sql.sdk.quantize import quantize_model
from gemma_4_sql.sdk.rag import build_rag_prompt, extract_schema_entities, retrieve_relevant_schema
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import JSONValue

logger = logging.getLogger(__name__)


def print_cli_error(
    err: GemmaSQLError,
    backend: str,
    *,
    model: str | None = None,
    action: str | None = None,
    **extra_fields: JSONValue,
) -> int:
    """Print structured JSON error output for a failed CLI command and return exit code 1.

    Args:
        err: Caught `GemmaSQLError` instance.
        backend: Default target backend name if `err.backend` is unset.
        model: Optional default target model name if `err.model_name` is unset.
        action: Optional default action/operation name if `err.operation` is unset.
        **extra_fields: Additional command-specific context fields to include in the JSON payload.

    Returns:
        Exit code `1`.

    """
    payload: dict[str, JSONValue] = {"backend": err.backend or backend}
    if action is not None:
        payload["action"] = err.operation or action
    if model is not None:
        payload["model"] = err.model_name or model
    payload.update(extra_fields)
    payload["status"] = f"failed: {err!s}"
    payload["status_code"] = ExecutionStatus.FAILED.value
    payload["error_message"] = str(err)
    print(json.dumps(payload, indent=2))
    return 1


def parse_db_kwargs(raw_db_kwargs: str | None) -> dict[str, JSONValue] | None:
    """Parse `--db-kwargs` JSON string and validate that it represents a JSON object.

    Args:
        raw_db_kwargs: Optional raw JSON string supplied via `--db-kwargs`.

    Returns:
        Parsed dictionary of database keyword arguments, or `None` if parsing fails.

    """
    if not raw_db_kwargs:
        return {}
    try:
        parsed = json.loads(raw_db_kwargs)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON provided in --db-kwargs: %s", exc)
        return None
    if not isinstance(parsed, dict):
        logger.warning("Expected a JSON object for --db-kwargs, got %s", type(parsed).__name__)
        return None
    return parsed


def parse_json_list(raw_json: str | None, context_label: str) -> list[dict[str, str]] | None:
    """Parse an optional JSON string into a list of string dictionaries.

    Args:
        raw_json: Raw JSON string or None.
        context_label: Command label for warning logs on invalid JSON.

    Returns:
        Parsed list of dictionaries, or `None` if JSON decoding fails.

    """
    if not raw_json:
        return []
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON %s: %s", context_label, exc)
        return None
    return list(parsed) if isinstance(parsed, list) else []


def tokenize_cmd(args: argparse.Namespace) -> int:
    """Run tokenization.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if `--decode` receives invalid token IDs.

    """
    tokenizer = SQLTokenizer(model_name=args.hf_model, vocab_size=args.vocab_size)
    if args.decode:
        try:
            tokens = [int(t.strip()) for t in args.decode.split(",")]
            decoded = tokenizer.decode(tokens)
            print(str(decoded))
        except ValueError as e:
            logger.warning("Invalid decode tokens: %s", e)
            return 1
    elif args.encode:
        encoded = tokenizer.encode(args.encode)
        print(json.dumps(encoded))
    return 0


def quantize_cmd(args: argparse.Namespace) -> int:
    """Run quantization.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if quantization fails.

    """
    try:
        res = quantize_model(args.model, args.method, args.backend)
    except GemmaSQLError as err:
        return print_cli_error(err, args.backend, model=args.model, method=args.method)
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0


def export_cmd(args: argparse.Namespace) -> int:
    """Run model export.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if export fails.

    """
    try:
        res = export_model(args.model, args.path, args.backend)
    except GemmaSQLError as err:
        return print_cli_error(err, args.backend, model=args.model, export_path=args.path)
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0


def rag_cmd(args: argparse.Namespace) -> None:
    """Build a RAG-augmented prompt or extract schema context.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    """
    if args.action == "extract":
        schema = extract_schema_entities(args.ddl)
        print(json.dumps(schema, indent=2))
    elif args.action == "retrieve":
        schema = extract_schema_entities(args.ddl)
        retrieved = retrieve_relevant_schema(args.prompt, schema)
        print(str(retrieved))
    else:
        modality = str((args.modality if "modality" in vars(args) else "text") or "text")
        image_path = None if modality == "audio" else args.image_path
        audio_path = None if modality == "vision" else args.audio_path
        rag_prompt = build_rag_prompt(
            prompt=args.prompt,
            ddl=args.ddl,
            image_path=image_path,
            audio_path=audio_path,
        )
        print(str(rag_prompt))


def log_metrics_cmd(args: argparse.Namespace) -> int:
    """Log training metrics.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if metric parsing or logging fails.

    """
    metrics_dict: dict[str, float] = {}
    if args.metrics:
        for item in args.metrics.split(","):
            stripped = item.strip()
            if not stripped:
                continue
            if "=" not in stripped:
                logger.warning("Skipping malformed metric entry (expected key=value): %s", stripped)
                continue
            (k, v) = stripped.split("=", 1)
            try:
                metrics_dict[k.strip()] = float(v.strip())
            except ValueError as e:
                logger.warning("Invalid float value in metric entry '%s': %s", stripped, e)
                return 1
    try:
        res = log_metrics(metrics=metrics_dict, step=args.step, log_dir=args.log_dir, backend=args.backend)
    except GemmaSQLError as err:
        return print_cli_error(err, args.backend, step=args.step, log_dir=args.log_dir)
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0
