"""Training CLI commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from gemma_4_sql.exceptions import is_failed_status
from gemma_4_sql.sdk import (
    TrainingConfig,
    apply_peft,
    posttrain_model,
    pretrain_model,
    run_dpo,
    sft_model,
    train_from_scratch,
)

if TYPE_CHECKING:
    import argparse


def train_cmd(args: argparse.Namespace) -> int:
    """Train a new model from scratch.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if the training status indicates failure.

    """
    batch_size = getattr(args, "batch_size", 2)
    distributed_strategy = getattr(args, "distributed_strategy", "none")
    config = TrainingConfig(
        model_name=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=batch_size,
        backend=args.backend,
        distributed_strategy=distributed_strategy,
    )
    res = train_from_scratch(config)
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0


def pretrain_cmd(args: argparse.Namespace) -> int:
    """Pretrain an existing model.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if the pretraining status indicates failure.

    """
    batch_size = getattr(args, "batch_size", 2)
    distributed_strategy = getattr(args, "distributed_strategy", "none")
    config = TrainingConfig(
        model_name=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=batch_size,
        backend=args.backend,
        distributed_strategy=distributed_strategy,
    )
    res = pretrain_model(config)
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0


def sft_cmd(args: argparse.Namespace) -> int:
    """Supervised fine-tune an existing model.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if the SFT status indicates failure.

    """
    batch_size = getattr(args, "batch_size", 2)
    distributed_strategy = getattr(args, "distributed_strategy", "none")
    config = TrainingConfig(
        model_name=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=batch_size,
        backend=args.backend,
        distributed_strategy=distributed_strategy,
    )
    res = sft_model(config)
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0


def posttrain_cmd(args: argparse.Namespace) -> int:
    """Post-train an existing model.

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if the post-training status indicates failure.

    """
    batch_size = getattr(args, "batch_size", 2)
    distributed_strategy = getattr(args, "distributed_strategy", "none")
    config = TrainingConfig(
        model_name=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=batch_size,
        backend=args.backend,
        distributed_strategy=distributed_strategy,
    )
    res = posttrain_model(config)
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0


def dpo_cmd(args: argparse.Namespace) -> int:
    """Run Direct Preference Optimization (DPO).

    Args:
        args: Parsed command-line arguments containing command-specific options.

    Returns:
        Exit code `0` on success, or `1` if the DPO status indicates failure.

    """
    batch_size = getattr(args, "batch_size", 2)
    epochs = getattr(args, "epochs", 1)
    learning_rate = getattr(args, "learning_rate", 1e-05)
    res = run_dpo(
        model_name=args.model,
        dataset=args.dataset,
        backend=args.backend,
        beta=args.beta,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
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
    res = apply_peft(
        model_name=args.model,
        target_modules=target_modules,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        backend=args.backend,
    )
    print(json.dumps(res, indent=2))
    return 1 if is_failed_status(res) else 0
