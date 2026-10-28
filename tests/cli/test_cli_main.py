"""Behavioral unit tests for CLI entry points, subcommands, and output formatting."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.cli import cli
from gemma_4_sql.cli_serve import chat_cmd, few_shot_cmd, serve_cmd
from gemma_4_sql.type_hints import ETLConfig, JSONDict, JSONValue, TrainingConfig

type CLICallArg = str | int | float | bool | ETLConfig | TrainingConfig | list[str] | dict[str, float] | None
type CLIRecordedCall = tuple[str, tuple[CLICallArg, ...], dict[str, CLICallArg | JSONValue]]


@pytest.fixture(autouse=True)
def mock_cli_environment(monkeypatch: pytest.MonkeyPatch) -> list[CLIRecordedCall]:
    """Mock CLI backend dispatch targets and record all invocations for verification."""
    recorded_calls: list[CLIRecordedCall] = []
    base_res: JSONDict = {
        "status": "completed",
        "metrics": {},
        "sql": "SELECT 1",
        "history": [{"role": "assistant", "content": "hi"}],
        "response": "hi",
    }

    def make_recorder(func_name: str) -> Callable[..., JSONDict]:
        """Create a recording mock function for `func_name`."""

        def _mock_fn(*args: CLICallArg, **kwargs: CLICallArg | JSONValue) -> JSONDict:
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
    mock_cli_environment: list[CLIRecordedCall],
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
        assert isinstance(cfg, ETLConfig)
        assert cfg.dataset_name == "my-data"
        assert cfg.split == "test"
        assert cfg.batch_size == 16
        assert kwargs["backend"] == "jax"


def test_cli_training_commands(
    mock_cli_environment: list[CLIRecordedCall],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI training subcommands construct and forward TrainingConfig, DPO parameters, and PEFT arguments."""
    assert cli(["train", "--model", "test-model", "--backend", "jax", "--dataset", "dummy", "--image-path", "img.png", "--audio-path", "aud.wav", "--modality", "multimodal"]) == 0
    assert cli(["pretrain", "--model", "test-model", "--backend", "maxtext", "--dataset", "dummy"]) == 0
    assert cli(["sft", "--model", "test-model", "--backend", "jax", "--dataset", "dummy"]) == 0
    assert cli(["posttrain", "--model", "test-model", "--backend", "keras", "--dataset", "dummy"]) == 0
    assert cli(["dpo", "--model", "test-model", "--backend", "jax", "--beta", "0.2", "--dataset", "dummy", "--image-path", "img.png"]) == 0
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
    cfg0 = mock_cli_environment[0][1][0]
    cfg1 = mock_cli_environment[1][1][0]
    cfg3 = mock_cli_environment[3][1][0]
    assert isinstance(cfg0, TrainingConfig)
    assert isinstance(cfg1, TrainingConfig)
    assert isinstance(cfg3, TrainingConfig)
    assert cfg0.model_name == "test-model"
    assert cfg0.backend == "jax"
    assert cfg0.extra_kwargs == {"image_path": "img.png", "audio_path": "aud.wav", "modality": "multimodal"}
    assert cfg1.backend == "maxtext"
    assert cfg3.backend == "keras"
    assert mock_cli_environment[4][2]["beta"] == 0.2
    assert mock_cli_environment[4][2]["image_path"] == "img.png"
    assert mock_cli_environment[5][2]["target_modules"] == ["q_proj", "v_proj"]
    assert mock_cli_environment[5][2]["lora_r"] == 16
    assert mock_cli_environment[5][2]["lora_alpha"] == 32
    assert mock_cli_environment[5][2]["lora_dropout"] == 0.1
    assert mock_cli_environment[5][2]["backend"] == "pytorch"


def test_cli_quantize_and_export(
    mock_cli_environment: list[CLIRecordedCall],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI quantize and export commands forward arguments and print JSON output."""
    assert cli(["quantize", "--model", "test-model", "--method", "awq", "--backend", "jax"]) == 0
    assert '"status": "completed"' in capsys.readouterr().out
    assert cli(["export", "--model", "test-model", "--path", "./out", "--backend", "keras"]) == 0
    assert '"status": "completed"' in capsys.readouterr().out

    assert mock_cli_environment[0][0] == "quantize_model"
    assert mock_cli_environment[0][1] == ("test-model", "awq", "jax")
    assert mock_cli_environment[1][0] == "export_model"
    assert mock_cli_environment[1][1] == ("test-model", "./out", "keras")


def test_cli_evaluate_and_agent(
    mock_cli_environment: list[CLIRecordedCall],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify evaluate_cmd and agent_cmd handle valid, empty, and invalid db_kwargs."""
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
    assert mock_cli_environment[-1][2]["db_kwargs"] == {"timeout": 10}
    assert cli(["evaluate", "--model", "test-model", "--dataset", "my-data"]) == 0
    assert mock_cli_environment[-1][2]["db_kwargs"] == {}
    assert cli(["evaluate", "--model", "test-model", "--db-kwargs", "not_json"]) == 1
    assert cli(["evaluate", "--model", "test-model", "--db-kwargs", "[1, 2]"]) == 1

    assert cli(["agent", "--model", "test-model", "--prompt", "Find users", "--backend", "jax", "--db-kwargs", '{"timeout": 10}']) == 0
    assert mock_cli_environment[-1][2]["db_kwargs"] == {"timeout": 10}
    assert cli(["agent", "--model", "test-model", "--prompt", "Find users"]) == 0
    assert mock_cli_environment[-1][2]["db_kwargs"] == {}
    assert cli(["agent", "--model", "test-model", "--prompt", "Find users", "--db-kwargs", "bad_json"]) == 1
    capsys.readouterr()


def test_cli_generate_variants(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify generate_cmd handles confidence display, non-confidence display, and empty SQL."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda **_k: {"sql": "SELECT 1;", "confidence_score": 0.95, "status": "success"})  # pyright: ignore[reportUnknownLambdaType]
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
    out_conf = capsys.readouterr().out
    assert "SELECT 1;" in out_conf
    assert "Confidence: 0.9500" in out_conf

    assert cli(["generate", "--model", "test-model", "--prompt", "Find all users"]) == 0
    out_no_conf = capsys.readouterr().out
    assert "SELECT 1;" in out_no_conf
    assert "Confidence" not in out_no_conf

    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda **_k: {"sql": "", "status": "success"})  # pyright: ignore[reportUnknownLambdaType]
    assert cli(["generate", "--model", "test-model", "--prompt", "Find all users"]) == 0
    assert capsys.readouterr().out == ""


def test_cli_rag_variants(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify rag_cmd handles build, extract, retrieve, and multimodal arguments."""
    assert cli(["rag", "--prompt", "Find users", "--ddl", "CREATE TABLE users (id INT, name VARCHAR);"]) == 0
    assert cli(["rag", "--action", "extract", "--ddl", "CREATE TABLE t (id INT);"]) == 0
    assert cli(["rag", "--action", "retrieve", "--prompt", "test", "--ddl", "CREATE TABLE t (id INT);"]) == 0
    capsys.readouterr()

    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", lambda prompt, ddl, **_k: f"{prompt} with {ddl}")  # pyright: ignore[reportUnknownLambdaType]
    assert cli(["rag", "--prompt", "Test prompt", "--ddl", "Test DDL", "--image-path", "test.jpg", "--audio-path", "test.mp3"]) == 0
    assert "Test prompt with Test DDL" in capsys.readouterr().out


def test_cli_log_metrics_variants(
    mock_cli_environment: list[CLIRecordedCall],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify log_metrics_cmd parses key=value pairs, skips malformed entries, and returns 1 on invalid float."""
    assert cli(["log", "--step", "100", "--metrics", "loss=0.5, ,malformed,acc=0.9", "--backend", "jax"]) == 0
    assert mock_cli_environment[-1][2]["metrics"] == {"loss": 0.5, "acc": 0.9}
    assert cli(["log", "--step", "100", "--metrics", "", "--backend", "jax"]) == 0
    assert mock_cli_environment[-1][2]["metrics"] == {}
    assert cli(["log", "--step", "100", "--metrics", "loss=not_a_float", "--backend", "jax"]) == 1
    capsys.readouterr()


def test_cli_tokenize_variants(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify tokenize subcommand handles encode, decode, invalid decode, and no-op."""
    assert cli(["tokenize", "--encode", "SELECT *"]) == 0
    assert "[83, 69, 76, 69, 67, 84, 32, 42]" in capsys.readouterr().out
    assert cli(["tokenize", "--decode", "83, 69, 76, 69, 67, 84, 32, 42"]) == 0
    assert "SELECT *" in capsys.readouterr().out
    assert cli(["tokenize", "--decode", "invalid, data"]) == 1
    assert cli(["tokenize"]) == 0
    capsys.readouterr()


def test_cli_execute_and_embed_duckdb(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify execute and embed-duckdb commands handle success, failure, invalid JSON, and missing duckdb."""
    assert cli(["execute", "--query", "SELECT 1 as num", "--db-type", "sqlite"]) == 0
    assert cli(["execute", "--query", "SELECT * FROM non_existent_table", "--db-type", "sqlite"]) != 0
    assert cli(["execute", "--query", "SELECT 1 as num", "--db-type", "sqlite", "--db-kwargs", '{"timeout": 10}']) == 0
    assert cli(["execute", "--query", "SELECT 1 as num", "--db-type", "sqlite", "--db-kwargs", "invalid_json"]) == 1
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
        assert cli(["embed-duckdb"]) == 1
        capsys.readouterr()


def test_cli_serve_chat_and_few_shot(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify serve, chat, and few-shot commands handle valid, empty, and invalid JSON inputs."""
    assert cli(["serve", "--model", "my-model", "--port", "9000", "--max-batch-size", "128", "--backend", "jax", "--modality", "vision"]) == 0
    assert cli(["chat", "--prompt", "hello", "--history", '[{"role": "user", "content": "hi"}]', "--backend", "jax", "--image-path", "a.png", "--audio-path", "b.wav", "--modality", "multimodal"]) == 0
    assert cli(["chat", "--prompt", "hello", "--history", "invalid_json"]) == 1
    assert cli(["few-shot", "--prompt", "hello", "--examples", '[{"input": "in", "output": "out"}]', "--backend", "jax"]) == 0
    assert cli(["few-shot", "--prompt", "hello", "--examples", "invalid_json"]) == 1
    capsys.readouterr()

    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", lambda **_k: None)  # pyright: ignore[reportUnknownLambdaType]
    serve_cmd(argparse.Namespace(model="dummy", port=8000, max_batch_size=32, backend="dummy", run_server=False, modality="text"))
    assert "Serving model dummy on port 8000 (status: unknown)" in capsys.readouterr().out

    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda **_k: {"response": "SELECT 1;"})  # pyright: ignore[reportUnknownLambdaType]
    assert cli(["chat", "--prompt", "hello"]) == 0
    assert "SELECT 1;" in capsys.readouterr().out

    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda **_k: {})  # pyright: ignore[reportUnknownLambdaType]
    chat_cmd(argparse.Namespace(model="d", prompt="d", backend="d", history="", image_path=None, audio_path=None, modality="text"))
    assert capsys.readouterr().out == ""

    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda **kw: {"few_shot_prompt": "prompt with examples" if kw.get("examples") else ""})  # pyright: ignore[reportUnknownLambdaType]
    assert cli(["few-shot", "--prompt", "dummy", "--examples", '[{"prompt": "q", "sql": "s"}]']) == 0
    assert "prompt with examples" in capsys.readouterr().out

    few_shot_cmd(argparse.Namespace(model="dummy", prompt="dummy", backend="dummy", examples=""))
    assert capsys.readouterr().out == ""


def test_cli_real_end_to_end_unmocked(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify end-to-end CLI execution for rag, tokenize, and execute with real SDK functions."""
    from gemma_4_sql.sdk.rag import build_rag_prompt, extract_schema_entities, retrieve_relevant_schema

    monkeypatch.setattr("gemma_4_sql.cli_misc.extract_schema_entities", extract_schema_entities)
    monkeypatch.setattr("gemma_4_sql.cli_misc.retrieve_relevant_schema", retrieve_relevant_schema)
    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", build_rag_prompt)

    ddl = "CREATE TABLE users (id INT, email TEXT); CREATE TABLE orders (id INT, total REAL);"
    assert cli(["rag", "--action", "extract", "--ddl", ddl]) == 0
    extracted = json.loads(capsys.readouterr().out)
    assert extracted == {"users": ["id", "email"], "orders": ["id", "total"]}

    assert cli(["rag", "--action", "build", "--prompt", "Find user email", "--ddl", ddl]) == 0
    built_prompt = capsys.readouterr().out
    assert "Table: users" in built_prompt
    assert "SELECT" in built_prompt

    assert cli(["execute", "--ddl", ddl, "--query", "SELECT COUNT(*) FROM users", "--db-type", "sqlite"]) == 0
    exec_out = json.loads(capsys.readouterr().out)
    assert exec_out["success"] is True
    assert exec_out["results"] == [[0]]


def test_cli_if_main(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify running src/gemma_4_sql/cli.py as __main__ with real CLI arguments exits with code 0."""
    monkeypatch.setattr(sys, "argv", ["gemma-4-sql", "tokenize", "--encode", "SELECT 1"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path("src/gemma_4_sql/cli.py", run_name="__main__")
    assert excinfo.value.code == 0
    assert "[" in capsys.readouterr().out


def test_cli_gemma_sql_error_handling(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI subcommands catch GemmaSQLError, print structured JSON error payloads, and return exit code 1."""
    from gemma_4_sql.exceptions import ConfigurationError, ETLError, ExportError, InferenceError, QuantizationError, TrainingError

    def _raise_train(*_a: CLICallArg, **_k: CLICallArg | JSONValue) -> JSONDict:
        """Raise TrainingError."""
        raise TrainingError("train err", backend="jax", model_name="m")

    for fn in ("train_from_scratch", "pretrain_model", "sft_model", "posttrain_model", "run_dpo", "apply_peft"):
        monkeypatch.setattr(f"gemma_4_sql.cli_train.{fn}", _raise_train)

    assert cli(["train", "--model", "m", "--dataset", "d"]) == 1
    assert cli(["pretrain", "--model", "m", "--dataset", "d"]) == 1
    assert cli(["sft", "--model", "m", "--dataset", "d"]) == 1
    assert cli(["posttrain", "--model", "m", "--dataset", "d"]) == 1
    assert cli(["dpo", "--model", "m", "--dataset", "d"]) == 1
    assert cli(["peft", "--model", "m"]) == 1

    def _raise_etl(*_a: CLICallArg, **_k: CLICallArg | JSONValue) -> JSONDict:
        """Raise ETLError."""
        raise ETLError("etl err", backend="jax")

    for fn in ("etl_pretrain", "etl_sft", "etl_posttrain"):
        monkeypatch.setattr(f"gemma_4_sql.cli_etl.{fn}", _raise_etl)

    assert cli(["etl", "pretrain"]) == 1
    assert cli(["etl", "sft"]) == 1
    assert cli(["etl", "posttrain"]) == 1

    def _raise_quant(*_a: CLICallArg, **_k: CLICallArg | JSONValue) -> JSONDict:
        """Raise QuantizationError."""
        raise QuantizationError("quant err", backend="jax", model_name="m")

    def _raise_export(*_a: CLICallArg, **_k: CLICallArg | JSONValue) -> JSONDict:
        """Raise ExportError."""
        raise ExportError("export err", backend="jax", model_name="m")

    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", _raise_quant)
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", _raise_export)
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", _raise_train)

    assert cli(["quantize", "--model", "m"]) == 1
    assert cli(["export", "--model", "m", "--path", "./out"]) == 1
    assert cli(["log", "--step", "1", "--metrics", "loss=1.0"]) == 1

    def _raise_inf(*_a: CLICallArg, **_k: CLICallArg | JSONValue) -> JSONDict:
        """Raise InferenceError."""
        raise InferenceError("inf err", backend="jax", model_name="m")

    for fn in ("evaluate", "generate", "run_agentic_loop", "serve_model", "chat_turn", "build_few_shot_prompt"):
        monkeypatch.setattr(f"gemma_4_sql.cli_serve.{fn}", _raise_inf)

    assert cli(["evaluate", "--model", "m"]) == 1
    assert cli(["generate", "--model", "m", "--prompt", "p"]) == 1
    assert cli(["agent", "--model", "m", "--prompt", "p"]) == 1
    assert cli(["serve", "--model", "m"]) == 1
    assert cli(["chat", "--model", "m", "--prompt", "p"]) == 1
    assert cli(["few-shot", "--model", "m", "--prompt", "p"]) == 1

    def _raise_cfg(*_a: CLICallArg, **_k: CLICallArg | JSONValue) -> None:
        """Raise ConfigurationError."""
        raise ConfigurationError("bad db type")

    monkeypatch.setattr("gemma_4_sql.cli_db.LiveDatabaseEngine", _raise_cfg)
    assert cli(["execute", "--query", "SELECT 1", "--db-type", "unknown"]) == 1
    capsys.readouterr()
