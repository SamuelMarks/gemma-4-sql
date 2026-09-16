"""Tests for modular CLI parsers."""

from __future__ import annotations

import argparse

from gemma_4_sql.cli import build_root_parser
from gemma_4_sql.cli_parsers import (
    add_etl_parsers,
    add_evaluate_parsers,
    add_generate_agent_parsers,
    add_misc_parsers,
    add_serve_export_parsers,
    add_training_parsers,
)


def test_build_root_parser() -> None:
    """Test that the root parser compiles and recognizes core subcommands."""
    parser = build_root_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    args = parser.parse_args(["etl", "pretrain", "--dataset", "dummy_dataset"])
    assert args.command == "etl"
    assert args.etl_command == "pretrain"
    assert args.dataset == "dummy_dataset"


def test_add_etl_parsers() -> None:
    """Test registering ETL parsers."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    etl_parser = add_etl_parsers(subparsers)
    assert isinstance(etl_parser, argparse.ArgumentParser)

    args = parser.parse_args(["etl", "sft", "--split", "validation", "--batch-size", "16"])
    assert args.cmd == "etl"
    assert args.etl_command == "sft"
    assert args.split == "validation"
    assert args.batch_size == 16


def test_add_training_parsers() -> None:
    """Test registering training, DPO, PEFT, and Quantize parsers."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    add_training_parsers(subparsers)

    train_args = parser.parse_args(["train", "--dataset", "my_dataset", "--epochs", "3"])
    assert train_args.cmd == "train"
    assert train_args.epochs == 3

    dpo_args = parser.parse_args(["dpo", "--dataset", "my_dataset", "--beta", "0.2"])
    assert dpo_args.cmd == "dpo"
    assert dpo_args.beta == 0.2

    peft_args = parser.parse_args(["peft", "--model", "gemma", "--lora-r", "16"])
    assert peft_args.cmd == "peft"
    assert peft_args.lora_r == 16

    quant_args = parser.parse_args(["quantize", "--model", "gemma", "--method", "int8"])
    assert quant_args.cmd == "quantize"
    assert quant_args.method == "int8"


def test_add_serve_and_evaluate_parsers() -> None:
    """Test registering serve, generate, agent, and evaluate parsers."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    add_serve_export_parsers(subparsers)
    add_generate_agent_parsers(subparsers)
    add_evaluate_parsers(subparsers)

    serve_args = parser.parse_args(["serve", "--port", "9000", "--max-batch-size", "128"])
    assert serve_args.cmd == "serve"
    assert serve_args.port == 9000
    assert serve_args.max_batch_size == 128

    gen_args = parser.parse_args(["generate", "--prompt", "SELECT 1"])
    assert gen_args.cmd == "generate"
    assert gen_args.prompt == "SELECT 1"

    agent_args = parser.parse_args(["agent", "--prompt", "SELECT 1", "--max-retries", "5"])
    assert agent_args.cmd == "agent"
    assert agent_args.max_retries == 5

    eval_args = parser.parse_args(["evaluate", "--dataset", "eval_data"])
    assert eval_args.cmd == "evaluate"
    assert eval_args.dataset == "eval_data"


def test_add_misc_and_db_parsers() -> None:
    """Test registering DB, benchmark, tokenize, and RAG parsers."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    add_misc_parsers(subparsers)

    exec_args = parser.parse_args(["execute", "--query", "SELECT 1"])
    assert exec_args.cmd == "execute"
    assert exec_args.query == "SELECT 1"

    bench_args = parser.parse_args(["benchmark", "--model", "gemma-4", "--hardware", "cpu", "--image-path", "erd.png", "--audio-path", "q.wav", "--modality", "multimodal"])
    assert bench_args.cmd == "benchmark"
    assert bench_args.hardware == "cpu"
    assert bench_args.image_path == "erd.png"
    assert bench_args.audio_path == "q.wav"
    assert bench_args.modality == "multimodal"

    rag_args = parser.parse_args(["rag", "--ddl", "CREATE TABLE t (x INT);", "--image-path", "erd.png", "--modality", "vision"])
    assert rag_args.cmd == "rag"
    assert rag_args.ddl == "CREATE TABLE t (x INT);"
    assert rag_args.image_path == "erd.png"
    assert rag_args.modality == "vision"
