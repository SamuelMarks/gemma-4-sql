"""Training CLI commands."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

from gemma_4_sql.cli_misc import print_cli_error
from gemma_4_sql.exceptions import GemmaSQLError, is_failed_status
from gemma_4_sql.sdk import (
    TrainingConfig,
    apply_peft,
    posttrain_model,
    pretrain_model,
    run_dpo,
    sft_model,
    train_from_scratch,
)
from gemma_4_sql.type_hints import JSONValue, TrainResultDict


def _format_cli_error(err: GemmaSQLError, action: str, backend: str, model: str) -> int:
    """Print structured JSON error output for a failed CLI training command and return exit code 1.

    Args:
        err: Caught `GemmaSQLError` instance.
        action: Training action name.
        backend: Target backend name.
        model: Target model name.

    Returns:
        Exit code `1`.

    """
    return print_cli_error(err, backend, model=model, action=action)


def _extract_multimodal_kwargs(args: argparse.Namespace) -> dict[str, JSONValue]:
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


def _run_training_stage(
    args: argparse.Namespace,
    action: str,
    stage_fn: Callable[[TrainingConfig], TrainResultDict],
) -> int:
    """Construct a `TrainingConfig` from CLI args, invoke `stage_fn`, and print the JSON result.

    Args:
        args: Parsed command-line arguments containing command-specific options.
        action: Training action identifier string.
        stage_fn: SDK training function (`train_from_scratch`, `pretrain_model`, `sft_model`, or `posttrain_model`).

    Returns:
        Exit code `0` on success, or `1` if the training status indicates failure.

    """
    config = TrainingConfig(
        model_name=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        backend=args.backend,
        distributed_strategy=args.distributed_strategy,
        extra_kwargs=_extract_multimodal_kwargs(args),
    )
    try:
        res = stage_fn(config)
    except GemmaSQLError as err:
        return _format_cli_error(err, action, args.backend, args.model)
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0


def train_cmd(args: argparse.Namespace) -> int:
    """Train a new model from scratch.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if the training status indicates failure.

    """
    return _run_training_stage(args, "train_from_scratch", train_from_scratch)


def pretrain_cmd(args: argparse.Namespace) -> int:
    """Pretrain an existing model.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if the pretraining status indicates failure.

    """
    return _run_training_stage(args, "pretrain", pretrain_model)


def sft_cmd(args: argparse.Namespace) -> int:
    """Supervised fine-tune an existing model.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if the SFT status indicates failure.

    """
    return _run_training_stage(args, "sft", sft_model)


def posttrain_cmd(args: argparse.Namespace) -> int:
    """Post-train an existing model.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if the post-training status indicates failure.

    """
    return _run_training_stage(args, "posttrain", posttrain_model)


def dpo_cmd(args: argparse.Namespace) -> int:
    """Run Direct Preference Optimization (DPO).

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if the DPO status indicates failure.

    """
    mm_kwargs = _extract_multimodal_kwargs(args)
    try:
        res = run_dpo(
            model_name=args.model,
            dataset=args.dataset,
            backend=args.backend,
            beta=args.beta,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            **mm_kwargs,
        )
    except GemmaSQLError as err:
        return _format_cli_error(err, "dpo", args.backend, args.model)
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0


def peft_cmd(args: argparse.Namespace) -> int:
    """Apply PEFT / LoRA to an existing model.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if the PEFT status indicates failure.

    """
    target_modules = args.target_modules.split(",") if args.target_modules else None
    try:
        res = apply_peft(
            model_name=args.model,
            target_modules=target_modules,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            backend=args.backend,
        )
    except GemmaSQLError as err:
        return _format_cli_error(err, "apply_lora", args.backend, args.model)
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0
