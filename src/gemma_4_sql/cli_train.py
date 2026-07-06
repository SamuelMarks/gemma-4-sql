"""Training CLI commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.sdk import TrainingConfig, apply_peft, posttrain_model, pretrain_model, run_dpo, sft_model, train_from_scratch

if TYPE_CHECKING:
    import argparse


def train_cmd(args: argparse.Namespace) -> None:
    """Train a new model from scratch.


    Args:
        args: Parsed command-line arguments containing command-specific options."""
    train_from_scratch(TrainingConfig(model_name=args.model, dataset=args.dataset, epochs=args.epochs, learning_rate=args.learning_rate, backend=args.backend, distributed_strategy=getattr(args, "distributed_strategy", "none")))


def pretrain_cmd(args: argparse.Namespace) -> None:
    """Pretrain an existing model.


    Args:
        args: Parsed command-line arguments containing command-specific options."""
    pretrain_model(TrainingConfig(model_name=args.model, dataset=args.dataset, epochs=args.epochs, learning_rate=args.learning_rate, backend=args.backend, distributed_strategy=getattr(args, "distributed_strategy", "none")))


def sft_cmd(args: argparse.Namespace) -> None:
    """Supervised fine-tune an existing model.


    Args:
        args: Parsed command-line arguments containing command-specific options."""
    sft_model(TrainingConfig(model_name=args.model, dataset=args.dataset, epochs=args.epochs, learning_rate=args.learning_rate, backend=args.backend, distributed_strategy=getattr(args, "distributed_strategy", "none")))


def posttrain_cmd(args: argparse.Namespace) -> None:
    """Post-train an existing model.


    Args:
        args: Parsed command-line arguments containing command-specific options."""
    posttrain_model(TrainingConfig(model_name=args.model, dataset=args.dataset, epochs=args.epochs, learning_rate=args.learning_rate, backend=args.backend, distributed_strategy=getattr(args, "distributed_strategy", "none")))


def dpo_cmd(args: argparse.Namespace) -> None:
    """Run Direct Preference Optimization (DPO).

    Args:
        args: Parsed command-line arguments containing command-specific options."""
    run_dpo(model_name=args.model, dataset=args.dataset, backend=args.backend, beta=args.beta)


def peft_cmd(args: argparse.Namespace) -> None:
    """Apply PEFT / LoRA to an existing model.

    Args:
        args: Parsed command-line arguments containing command-specific options."""
    target_modules = args.target_modules.split(",") if args.target_modules else None
    apply_peft(model_name=args.model, target_modules=target_modules, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, backend=args.backend)
