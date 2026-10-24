"""Behavioral unit tests for CLI entry points, subcommands, and output formatting."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.cli import cli
from gemma_4_sql.cli_misc import export_cmd, log_metrics_cmd, quantize_cmd, rag_cmd, tokenize_cmd
from gemma_4_sql.cli_serve import agent_cmd, chat_cmd, evaluate_cmd, few_shot_cmd, generate_cmd, serve_cmd


@pytest.fixture(autouse=True)
def mock_cli_environment(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
    """Mock CLI backend dispatch targets and record all invocations for verification."""
    recorded_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    base_res: dict[str, Any] = {
        "status": "completed",
        "metrics": {},
        "sql": "SELECT 1",
        "history": [{"role": "assistant", "content": "hi"}],
        "response": "hi",
    }

    def make_recorder(func_name: str) -> Any:
        """Create a recording mock function for `func_name`."""

        def _mock_fn(*args: Any, **kwargs: Any) -> dict[str, Any]:
            """Record call arguments and return simulated CLI command result."""
            recorded_calls.append((func_name, args, dict(kwargs)))
            return base_res

        return _mock_fn

    modules = {
        "cli_serve": ["evaluate", "generate", "run_agentic_loop", "serve_model", "chat_turn", "build_few_shot_prompt"],
        "cli_train": ["run_dpo", "pretrain_model", "sft_model", "posttrain_model", "train_from_scratch", "apply_peft"],
        "cli_benchmark": ["benchmark"],
        "cli_misc": [
            "quantize_model",
            "export_model",
            "log_metrics",
            "extract_schema_entities",
            "retrieve_relevant_schema",
            "build_rag_prompt",
        ],
        "cli_etl": ["etl_pretrain", "etl_sft", "etl_posttrain"],
        "cli_db": ["embed_in_duckdb"],
    }

    for mod, funcs in modules.items():
        for func in funcs:
            monkeypatch.setattr(f"gemma_4_sql.{mod}.{func}", make_recorder(func))

    return recorded_calls


def test_cli_etl_commands(
    mock_cli_environment: list[tuple[str, tuple[Any, ...], dict[str, Any]]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI ETL subcommands construct and forward ETLConfig with parsed CLI arguments."""
    for subcmd in ("pretrain", "sft", "posttrain"):
        res = cli(["etl", subcmd, "--dataset", "my-data", "--split", "test", "--batch-size", "16", "--backend", "jax"])
        assert res == 0
        capsys.readouterr()

    assert [c[0] for c in mock_cli_environment] == ["etl_pretrain", "etl_sft", "etl_posttrain"]
    for _, args, kwargs in mock_cli_environment:
        cfg = args[0]
        assert cfg.dataset_name == "my-data"
        assert cfg.split == "test"
        assert cfg.batch_size == 16
        assert kwargs["backend"] == "jax"


def test_cli_training_commands(
    mock_cli_environment: list[tuple[str, tuple[Any, ...], dict[str, Any]]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI training subcommands construct and forward TrainingConfig, DPO parameters, and PEFT arguments."""
    assert cli(["train", "--model", "test-model", "--backend", "jax", "--dataset", "dummy"]) == 0
    assert cli(["pretrain", "--model", "test-model", "--backend", "maxtext", "--dataset", "dummy"]) == 0
    assert cli(["sft", "--model", "test-model", "--backend", "jax", "--dataset", "dummy"]) == 0
    assert cli(["posttrain", "--model", "test-model", "--backend", "keras", "--dataset", "dummy"]) == 0
    assert cli(["dpo", "--model", "test-model", "--backend", "jax", "--beta", "0.2", "--dataset", "dummy"]) == 0
    assert (
        cli([
            "peft",
            "--model",
            "test-model",
            "--target-modules",
            "q_proj,v_proj",
            "--lora-r",
            "16",
            "--lora-alpha",
            "32",
            "--lora-dropout",
            "0.1",
            "--backend",
            "pytorch",
        ])
        == 0
    )
    capsys.readouterr()

    names = [c[0] for c in mock_cli_environment]
    assert names == ["train_from_scratch", "pretrain_model", "sft_model", "posttrain_model", "run_dpo", "apply_peft"]
    assert mock_cli_environment[0][1][0].model_name == "test-model"
    assert mock_cli_environment[0][1][0].backend == "jax"
    assert mock_cli_environment[1][1][0].backend == "maxtext"
    assert mock_cli_environment[3][1][0].backend == "keras"
    assert mock_cli_environment[4][2]["beta"] == 0.2
    assert mock_cli_environment[5][2]["target_modules"] == ["q_proj", "v_proj"]
    assert mock_cli_environment[5][2]["lora_r"] == 16
    assert mock_cli_environment[5][2]["lora_alpha"] == 32
    assert mock_cli_environment[5][2]["lora_dropout"] == 0.1
    assert mock_cli_environment[5][2]["backend"] == "pytorch"


def test_cli_quantize_and_export(
    mock_cli_environment: list[tuple[str, tuple[Any, ...], dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI quantize and export commands forward arguments and print JSON output."""
    assert cli(["quantize", "--model", "test-model", "--method", "awq", "--backend", "jax"]) == 0
    assert cli(["export", "--model", "test-model", "--path", "./out", "--backend", "keras"]) == 0
    capsys.readouterr()

    assert mock_cli_environment[0][0] == "quantize_model"
    assert mock_cli_environment[0][1] == ("test-model", "awq", "jax")
    assert mock_cli_environment[1][0] == "export_model"
    assert mock_cli_environment[1][1] == ("test-model", "./out", "keras")

    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *_a, **_k: {"status": "q"})
    quantize_cmd(argparse.Namespace(model="dummy", method="dummy", backend="dummy"))
    assert '"status": "q"' in capsys.readouterr().out

    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *_a, **_k: {"status": "e"})
    export_cmd(argparse.Namespace(model="dummy", path="dummy", backend="dummy"))
    assert '"status": "e"' in capsys.readouterr().out


def test_cli_evaluate_and_agent(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify evaluate_cmd and agent_cmd handle db_kwargs and empty responses."""
    assert (
        cli([
            "evaluate",
            "--model",
            "test-model",
            "--dataset",
            "my-data",
            "--backend",
            "jax",
            "--db-type",
            "sqlite",
            "--db-kwargs",
            '{"timeout": 10}',
        ])
        == 0
    )
    assert cli(["agent", "--model", "test-model", "--prompt", "Find users", "--backend", "jax", "--db-kwargs", '{"timeout": 10}']) == 0
    capsys.readouterr()

    eval_args = argparse.Namespace(model="d", dataset="d", backend="d", db_path="d", ddl="d", db_type="d", db_kwargs=json.dumps({"key": "value"}))
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda **kw: {"status": "evaluated"} if kw.get("db_kwargs") == {"key": "value"} else {})
    evaluate_cmd(eval_args)
    assert "evaluated" in capsys.readouterr().out

    eval_args_empty = argparse.Namespace(model="d", dataset="d", backend="d", db_path="d", ddl="d", db_type="d", db_kwargs="")
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda **_k: {})
    evaluate_cmd(eval_args_empty)
    assert capsys.readouterr().out.strip() == "{}"

    agent_args = argparse.Namespace(
        model="d",
        prompt="d",
        backend="d",
        db_path="d",
        ddl="d",
        db_type="d",
        max_retries=1,
        db_kwargs=json.dumps({"key": "value"}),
    )
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda **kw: {"status": "agent_run"} if kw.get("db_kwargs") == {"key": "value"} else {})
    agent_cmd(agent_args)
    assert "agent_run" in capsys.readouterr().out

    agent_args_empty = argparse.Namespace(model="d", prompt="d", backend="d", db_path="d", ddl="d", db_type="d", max_retries=1, db_kwargs="")
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda **_k: {})
    agent_cmd(agent_args_empty)
    assert capsys.readouterr().out.strip() == "{}"


def test_cli_generate_variants(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify generate_cmd handles confidence display, non-confidence display, and empty SQL."""
    assert (
        cli([
            "generate",
            "--model",
            "test-model",
            "--prompt",
            "Find all users",
            "--backend",
            "maxtext",
            "--beam-width",
            "5",
            "--max-length",
            "100",
            "--show-confidence",
        ])
        == 0
    )
    capsys.readouterr()

    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda **_k: {"sql": "SELECT 1;", "confidence_score": 0.95})
    generate_cmd(argparse.Namespace(model="d", prompt="d", backend="d", beam_width=3, max_length=50, show_confidence=True))
    out_conf = capsys.readouterr().out
    assert "SELECT 1;" in out_conf
    assert "Confidence: 0.9500" in out_conf

    generate_cmd(argparse.Namespace(model="d", prompt="d", backend="d", beam_width=3, max_length=50, show_confidence=False))
    out_no_conf = capsys.readouterr().out
    assert "SELECT 1;" in out_no_conf
    assert "Confidence" not in out_no_conf

    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda **_k: {"sql": ""})
    generate_cmd(argparse.Namespace(model="d", prompt="d", backend="d", beam_width=3, max_length=50, show_confidence=False))
    assert capsys.readouterr().out == ""


def test_cli_rag_variants(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify rag_cmd handles build, extract, retrieve, and multimodal arguments."""
    assert cli(["rag", "--prompt", "Find users", "--ddl", "CREATE TABLE users (id INT, name VARCHAR);"]) == 0
    assert cli(["rag", "--action", "extract", "--ddl", "CREATE TABLE t (id INT);"]) == 0
    assert cli(["rag", "--action", "retrieve", "--prompt", "test", "--ddl", "CREATE TABLE t (id INT);"]) == 0
    capsys.readouterr()

    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda prompt, ddl, **_k: f"{prompt} with {ddl}")
    rag_cmd(argparse.Namespace(action="build", prompt="Test prompt", ddl="Test DDL", image_path="test.jpg", audio_path="test.mp3"))
    assert "Test prompt with Test DDL" in capsys.readouterr().out

    rag_cmd(argparse.Namespace(action="build", prompt="d", ddl="d", image_path=None, audio_path=None))
    assert "d with d" in capsys.readouterr().out


def test_cli_log_metrics_variants(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify log_metrics_cmd parses key=value pairs and empty metric strings."""
    assert cli(["log", "--step", "100", "--metrics", "loss=0.5,acc=0.9", "--backend", "jax"]) == 0
    capsys.readouterr()

    def mock_log_metrics(**kwargs: Any) -> dict[str, str]:
        """Return status based on whether metrics dictionary is populated."""
        return {"status": "logged" if kwargs["metrics"] == {"loss": 1.0, "acc": 0.9} else "logged_empty"}

    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", mock_log_metrics)
    log_metrics_cmd(argparse.Namespace(metrics="loss=1.0, ,malformed,acc=0.9", step=1, log_dir="logs", backend="dummy"))
    assert "logged" in capsys.readouterr().out

    log_metrics_cmd(argparse.Namespace(metrics="", step=1, log_dir="logs", backend="dummy"))
    assert "logged_empty" in capsys.readouterr().out


def test_cli_tokenize_variants(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify tokenize subcommand handles encode, decode, invalid decode, and no-op."""
    assert cli(["tokenize", "--encode", "SELECT *"]) == 0
    assert cli(["tokenize", "--decode", "83, 69, 76, 69, 67, 84, 32, 42"]) == 0
    assert cli(["tokenize", "--decode", "invalid, data"]) == 0
    assert cli(["tokenize"]) == 0
    capsys.readouterr()

    monkeypatch.setattr("gemma_4_sql.cli_misc.SQLTokenizer.encode", lambda _self, _text: [1, 2, 3])
    tokenize_cmd(argparse.Namespace(hf_model=None, vocab_size=256, encode="test", decode=None))
    assert "[1, 2, 3]" in capsys.readouterr().out

    monkeypatch.setattr("gemma_4_sql.cli_misc.SQLTokenizer.decode", lambda _self, _tokens: "test")
    tokenize_cmd(argparse.Namespace(hf_model=None, vocab_size=256, encode=None, decode="1, 2, 3"))
    assert "test" in capsys.readouterr().out


def test_cli_execute_and_embed_duckdb(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify execute and embed-duckdb commands handle success, failure, and missing duckdb."""
    assert cli(["execute", "--query", "SELECT 1 as num", "--db-type", "sqlite"]) == 0
    assert cli(["execute", "--query", "SELECT * FROM non_existent_table", "--db-type", "sqlite"]) != 0
    assert cli(["execute", "--query", "SELECT 1 as num", "--db-type", "sqlite", "--db-kwargs", '{"timeout": 10}']) == 0
    capsys.readouterr()

    mock_duckdb = MagicMock()
    mock_conn = MagicMock()
    mock_duckdb.connect.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = [("Success!",)]
    with patch.dict("sys.modules", {"duckdb": mock_duckdb}), patch("gemma_4_sql.cli_db.embed_in_duckdb"):
        assert cli(["embed-duckdb", "--prompt", "Find users", "--ddl", "CREATE TABLE a (id int);"]) == 0
        assert cli(["embed-duckdb"]) == 0
        capsys.readouterr()

    with patch.dict("sys.modules", {"duckdb": None}):
        cli(["embed-duckdb"])
        capsys.readouterr()


def test_cli_serve_chat_and_few_shot(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify serve, chat, and few-shot commands handle valid, empty, and invalid JSON inputs."""
    assert cli(["serve", "--model", "my-model", "--port", "9000", "--max-batch-size", "128", "--backend", "jax"]) == 0
    assert cli(["chat", "--prompt", "hello", "--history", '[{"role": "user", "content": "hi"}]', "--backend", "jax"]) == 0
    assert cli(["chat", "--prompt", "hello", "--history", "invalid_json"]) == 0
    assert cli(["few-shot", "--prompt", "hello", "--examples", '[{"input": "in", "output": "out"}]', "--backend", "jax"]) == 0
    assert cli(["few-shot", "--prompt", "hello", "--examples", "invalid_json"]) == 0
    capsys.readouterr()

    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda **_k: {"status": "serving"})
    serve_cmd(argparse.Namespace(model="dummy", port=8000, max_batch_size=32, backend="dummy", run_server=False))
    assert "Serving model dummy on port 8000 (status: serving)" in capsys.readouterr().out

    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda **_k: None)
    serve_cmd(argparse.Namespace(model="dummy", port=8000, max_batch_size=32, backend="dummy", run_server=False))
    assert "Serving model dummy on port 8000 (status: unknown)" in capsys.readouterr().out

    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda **_k: {})
    chat_cmd(argparse.Namespace(model="d", prompt="d", backend="d", history=""))
    assert capsys.readouterr().out == ""

    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda **kw: {"few_shot_prompt": "prompt with examples" if kw.get("examples") else ""})
    few_shot_cmd(argparse.Namespace(model="dummy", prompt="dummy", backend="dummy", examples=json.dumps([{"prompt": "q", "sql": "s"}])))
    assert "prompt with examples" in capsys.readouterr().out

    few_shot_cmd(argparse.Namespace(model="dummy", prompt="dummy", backend="dummy", examples=""))
    assert capsys.readouterr().out == ""


def test_cli_if_main(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify running src/gemma_4_sql/cli.py as __main__ with real CLI arguments exits with code 0."""
    monkeypatch.setattr(sys, "argv", ["gemma-4-sql", "tokenize", "--encode", "SELECT 1"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path("src/gemma_4_sql/cli.py", run_name="__main__")
    assert excinfo.value.code == 0
    assert "[" in capsys.readouterr().out
