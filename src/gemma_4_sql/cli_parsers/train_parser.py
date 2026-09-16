"""CLI argument parsers for training, fine-tuning, PEFT, and quantization."""

from __future__ import annotations

import argparse
from typing import Callable

from gemma_4_sql.cli_misc import quantize_cmd
from gemma_4_sql.cli_train import dpo_cmd, peft_cmd, posttrain_cmd, pretrain_cmd, sft_cmd, train_cmd


def add_training_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help_text: str,
    backend: str,
    cmd_func: Callable[[argparse.Namespace], int | None],
) -> argparse.ArgumentParser:
    """Add a model training subcommand parser.

    Args:
        subparsers: Subparser collection to attach the parser to.
        name: Name of the training subparser command (e.g., 'train', 'sft').
        help_text: Descriptive help string for the command.
        backend: Default backend engine ('jax', 'pytorch', 'maxtext', 'keras').
        cmd_func: Callback function executed when this command is selected.

    Returns:
        The configured ArgumentParser instance for this training command.
    """
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("--model", default="gemma-4", help="Model name.")
    parser.add_argument("--dataset", required=True, help="Training dataset.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training dataloader.")
    parser.add_argument("--backend", default=backend, help="Backend to use.")
    parser.add_argument("--distributed-strategy", default="none", choices=["none", "ddp", "fsdp"], help="Distributed strategy.")
    parser.add_argument("--image-path", default=None, help="Path to schema diagram, ERD image, or table screenshot.")
    parser.add_argument("--audio-path", default=None, help="Path to recorded natural language query audio file.")
    parser.add_argument("--modality", default="text", choices=["text", "vision", "audio", "multimodal"], help="Explicit modality selector.")
    parser.set_defaults(func=cmd_func)
    return parser


def add_peft_quantize_parsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register DPO, PEFT/LoRA, and Quantize subcommand parsers.

    Args:
        subparsers: Subparser collection to attach the commands to.
    """
    parser_dpo = subparsers.add_parser("dpo", help="Run Direct Preference Optimization (DPO).")
    parser_dpo.add_argument("--model", default="gemma-4", help="Model name.")
    parser_dpo.add_argument("--dataset", required=True, help="Training dataset.")
    parser_dpo.add_argument("--beta", type=float, default=0.1, help="DPO temperature parameter.")
    parser_dpo.add_argument("--epochs", type=int, default=1, help="Number of DPO training epochs.")
    parser_dpo.add_argument("--learning-rate", type=float, default=1e-05, help="Learning rate for DPO.")
    parser_dpo.add_argument("--batch-size", type=int, default=2, help="Batch size for DPO dataloader.")
    parser_dpo.add_argument("--backend", default="jax", help="Backend to use.")
    parser_dpo.add_argument("--image-path", default=None, help="Path to schema diagram, ERD image, or table screenshot.")
    parser_dpo.add_argument("--audio-path", default=None, help="Path to recorded natural language query audio file.")
    parser_dpo.add_argument("--modality", default="text", choices=["text", "vision", "audio", "multimodal"], help="Explicit modality selector.")
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


def add_training_parsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register all model training, adaptation, and optimization subparsers.

    Args:
        subparsers: Subparser collection to attach training parsers to.
    """
    add_training_subparser(subparsers, "train", "Train a new model from scratch.", "jax", train_cmd)
    add_training_subparser(subparsers, "pretrain", "Pretrain an existing model.", "maxtext", pretrain_cmd)
    add_training_subparser(subparsers, "sft", "Supervised fine-tune an existing model.", "jax", sft_cmd)
    add_training_subparser(subparsers, "posttrain", "Post-train an existing model.", "keras", posttrain_cmd)
    add_peft_quantize_parsers(subparsers)
