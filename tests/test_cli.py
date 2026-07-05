# Copyright 2024
"""Provide module docstring."""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.cli import cli

"\n\nTests for the CLI module.\n"


def test_cli_etl_pretrain(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI ETL pretrain command."
    args = ["etl", "pretrain", "--dataset", "my-data", "--split", "test", "--batch-size", "16"]
    cli(args)
    capsys.readouterr()


def test_cli_etl_sft(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI ETL sft command."
    args = ["etl", "sft", "--dataset", "my-data", "--split", "test", "--batch-size", "16"]
    cli(args)
    capsys.readouterr()


def test_cli_etl_posttrain(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI ETL posttrain command."
    args = ["etl", "posttrain", "--dataset", "my-data", "--split", "test", "--batch-size", "16"]
    cli(args)
    capsys.readouterr()


def test_cli_train(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI train command."
    args = ["train", "--model", "test-model", "--backend", "jax"]
    cli(args)
    capsys.readouterr()


def test_cli_pretrain(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI pretrain command."
    args = ["pretrain", "--model", "test-model", "--backend", "maxtext"]
    cli(args)
    capsys.readouterr()


def test_cli_sft(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI sft command."
    args = ["sft", "--model", "test-model", "--backend", "jax"]
    cli(args)
    capsys.readouterr()


def test_cli_posttrain(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI posttrain command."
    args = ["posttrain", "--model", "test-model", "--backend", "keras"]
    cli(args)
    capsys.readouterr()


def test_cli_dpo(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI dpo command."
    args = ["dpo", "--model", "test-model", "--backend", "jax", "--beta", "0.2"]
    cli(args)
    capsys.readouterr()


def test_cli_quantize(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI quantize command."
    args = ["quantize", "--model", "test-model", "--method", "awq", "--backend", "jax"]
    cli(args)
    capsys.readouterr()


def test_cli_peft(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI peft command."
    args = ["peft", "--model", "test-model", "--target-modules", "q_proj,v_proj", "--lora-r", "16", "--lora-alpha", "32", "--lora-dropout", "0.1", "--backend", "pytorch"]
    cli(args)
    capsys.readouterr()


def test_cli_evaluate(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI evaluate command."
    args = ["evaluate", "--model", "test-model", "--dataset", "my-data", "--backend", "jax", "--db-type", "sqlite", "--db-kwargs", '{"timeout": 10}']
    cli(args)
    capsys.readouterr()


def test_cli_export(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI export command."
    args = ["export", "--model", "test-model", "--path", "./out", "--backend", "keras"]
    cli(args)
    capsys.readouterr()


def test_cli_generate(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI generate command."
    args = ["generate", "--model", "test-model", "--prompt", "Find all users", "--backend", "maxtext", "--beam-width", "5", "--max-length", "100"]
    cli(args)
    capsys.readouterr()


def test_cli_agent(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI agent command."
    args = ["agent", "--model", "test-model", "--prompt", "Find users", "--backend", "jax", "--db-kwargs", '{"timeout": 10}']
    cli(args)
    capsys.readouterr()


def test_cli_rag(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI rag command."
    args = ["rag", "--prompt", "Find users", "--ddl", "CREATE TABLE users (id INT, name VARCHAR);"]
    cli(args)
    capsys.readouterr()


def test_cli_log_metrics(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI log command."
    args = ["log", "--step", "100", "--metrics", "loss=0.5,acc=0.9", "--backend", "jax"]
    cli(args)
    capsys.readouterr()


def test_cli_tokenize_encode(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI tokenize command (encode)."
    args = ["tokenize", "--encode", "SELECT *"]
    cli(args)
    capsys.readouterr()


def test_cli_tokenize_decode(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI tokenize command (decode)."
    args = ["tokenize", "--decode", "83, 69, 76, 69, 67, 84, 32, 42"]
    cli(args)
    capsys.readouterr()


def test_cli_tokenize_decode_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI tokenize command (decode error)."
    args = ["tokenize", "--decode", "invalid, data"]
    cli(args)
    capsys.readouterr()


def test_cli_tokenize_none(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI tokenize command (neither encode nor decode)."
    args = ["tokenize"]
    cli(args)
    capsys.readouterr()


def test_cli_execute_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI execute command (success)."
    args = ["execute", "--query", "SELECT 1 as num", "--db-type", "sqlite"]
    cli(args)
    capsys.readouterr()


def test_cli_execute_fail(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI execute command (failure)."
    args = ["execute", "--query", "SELECT * FROM non_existent_table", "--db-type", "sqlite"]
    cli(args)
    capsys.readouterr()


def test_cli_execute_kwargs(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI execute command (with kwargs)."
    args = ["execute", "--query", "SELECT 1 as num", "--db-type", "sqlite", "--db-kwargs", '{"timeout": 10}']
    cli(args)
    capsys.readouterr()


def test_cli_embed_duckdb_prompt(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI embed-duckdb command with a prompt."
    mock_duckdb = MagicMock()
    mock_conn = MagicMock()
    mock_duckdb.connect.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = [("Success!",)]
    with patch.dict("sys.modules", {"duckdb": mock_duckdb}), patch("gemma_4_sql.cli_db.embed_in_duckdb"):
        args = ["embed-duckdb", "--prompt", "Find users", "--ddl", "CREATE TABLE a (id int);"]
        cli(args)
        capsys.readouterr()


def test_cli_embed_duckdb_no_prompt(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI embed-duckdb command without a prompt."
    mock_duckdb = MagicMock()
    mock_conn = MagicMock()
    mock_duckdb.connect.return_value = mock_conn
    with patch.dict("sys.modules", {"duckdb": mock_duckdb}), patch("gemma_4_sql.cli_db.embed_in_duckdb"):
        args = ["embed-duckdb"]
        cli(args)
        capsys.readouterr()


def test_cli_embed_duckdb_missing(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI embed-duckdb command when duckdb is missing."
    with patch.dict("sys.modules", {"duckdb": None}):
        args = ["embed-duckdb"]
        cli(args)
        capsys.readouterr()


def test_cli_rag_extract(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI rag command (extract action)."
    args = ["rag", "--action", "extract", "--ddl", "CREATE TABLE t (id INT);"]
    cli(args)
    capsys.readouterr()


def test_cli_rag_retrieve(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI rag command (retrieve action)."
    args = ["rag", "--action", "retrieve", "--prompt", "test", "--ddl", "CREATE TABLE t (id INT);"]
    cli(args)
    capsys.readouterr()


def test_cli_serve(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI serve command."
    args = ["serve", "--model", "my-model", "--port", "9000", "--max-batch-size", "128", "--backend", "jax"]
    cli(args)
    capsys.readouterr()


def test_cli_chat(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI chat command."
    args = ["chat", "--prompt", "hello", "--history", '[{"role": "user", "content": "hi"}]', "--backend", "jax"]
    cli(args)
    capsys.readouterr()


def test_cli_chat_invalid_history(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI chat command with invalid history json."
    args = ["chat", "--prompt", "hello", "--history", "invalid_json"]
    cli(args)
    capsys.readouterr()


def test_cli_few_shot(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI few-shot command."
    args = ["few-shot", "--prompt", "hello", "--examples", '[{"input": "in", "output": "out"}]', "--backend", "jax"]
    cli(args)
    capsys.readouterr()


def test_cli_few_shot_invalid_examples(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.run_dpo", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", lambda *_args, **_kwargs: type("MockEngine", (), {"execute_with_feedback": lambda *_a, **_k: (True, [], None), "close": lambda *_a, **_k: None})())
    monkeypatch.setattr("gemma_4_sql.cli_train.pretrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.sft_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.posttrain_model", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.train_from_scratch", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_db.embed_in_duckdb", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_pretrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_sft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_etl.etl_posttrain", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    monkeypatch.setattr("gemma_4_sql.cli_train.apply_peft", lambda *_args, **_kwargs: {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"})
    "Test the CLI few-shot command with invalid examples json."
    args = ["few-shot", "--prompt", "hello", "--examples", "invalid_json"]
    cli(args)
    capsys.readouterr()


def test_dependency_missing_error() -> None:
    from gemma_4_sql.exceptions import DependencyMissingError

    assert str(DependencyMissingError("test")) == "test"
