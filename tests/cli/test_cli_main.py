"""Provide module docstring."""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.cli import cli


@pytest.fixture(autouse=True)
def mock_cli_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock all CLI backend dependencies to prevent actual execution during parsing tests."""
    base_res = {"status": "completed", "metrics": {}, "sql": "SELECT 1", "history": [{"role": "assistant", "content": "hi"}], "response": "hi"}

    def mock_fn(*args, **kwargs):
        return base_res

    modules = {
        "cli_serve": ["evaluate", "generate", "run_agentic_loop", "serve_model", "chat_turn", "build_few_shot_prompt"],
        "cli_train": ["run_dpo", "pretrain_model", "sft_model", "posttrain_model", "train_from_scratch", "apply_peft"],
        "cli_benchmark": ["benchmark"],
        "cli_misc": ["quantize_model", "export_model", "log_metrics", "extract_schema_entities", "retrieve_relevant_schema", "build_rag_prompt"],
        "cli_etl": ["etl_pretrain", "etl_sft", "etl_posttrain"],
        "cli_db": ["embed_in_duckdb"],
    }

    for mod, funcs in modules.items():
        for func in funcs:
            monkeypatch.setattr(f"gemma_4_sql.{mod}.{func}", mock_fn)


"\n\nTests for the CLI module.\n"


def test_cli_etl_pretrain(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI ETL pretrain command."
    args = ["etl", "pretrain", "--dataset", "my-data", "--split", "test", "--batch-size", "16"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_etl_sft(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI ETL sft command."
    args = ["etl", "sft", "--dataset", "my-data", "--split", "test", "--batch-size", "16"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_etl_posttrain(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI ETL posttrain command."
    args = ["etl", "posttrain", "--dataset", "my-data", "--split", "test", "--batch-size", "16"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_train(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI train command."
    args = ["train", "--model", "test-model", "--backend", "jax", "--dataset", "dummy"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_pretrain(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI pretrain command."
    args = ["pretrain", "--model", "test-model", "--backend", "maxtext", "--dataset", "dummy"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_sft(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI sft command."
    args = ["sft", "--model", "test-model", "--backend", "jax", "--dataset", "dummy"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_posttrain(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI posttrain command."
    args = ["posttrain", "--model", "test-model", "--backend", "keras", "--dataset", "dummy"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_dpo(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI dpo command."
    args = ["dpo", "--model", "test-model", "--backend", "jax", "--beta", "0.2", "--dataset", "dummy"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_quantize(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI quantize command."
    args = ["quantize", "--model", "test-model", "--method", "awq", "--backend", "jax"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_peft(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI peft command."
    args = ["peft", "--model", "test-model", "--target-modules", "q_proj,v_proj", "--lora-r", "16", "--lora-alpha", "32", "--lora-dropout", "0.1", "--backend", "pytorch"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_evaluate(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI evaluate command."
    args = ["evaluate", "--model", "test-model", "--dataset", "my-data", "--backend", "jax", "--db-type", "sqlite", "--db-kwargs", '{"timeout": 10}']
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_export(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI export command."
    args = ["export", "--model", "test-model", "--path", "./out", "--backend", "keras"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_generate(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["generate", "--model", "test-model", "--prompt", "Find all users", "--backend", "maxtext", "--beam-width", "5", "--max-length", "100", "--show-confidence"]
    res = cli(args)
    assert res == 0


def test_cli_agent(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI agent command."
    args = ["agent", "--model", "test-model", "--prompt", "Find users", "--backend", "jax", "--db-kwargs", '{"timeout": 10}']
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_rag(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["rag", "--prompt", "Find users", "--ddl", "CREATE TABLE users (id INT, name VARCHAR);"]
    res = cli(args)
    assert res == 0
    capsys.readouterr()


def test_cli_log_metrics(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI log command."
    args = ["log", "--step", "100", "--metrics", "loss=0.5,acc=0.9", "--backend", "jax"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_tokenize_encode(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI tokenize command (encode)."
    args = ["tokenize", "--encode", "SELECT *"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_tokenize_decode(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI tokenize command (decode)."
    args = ["tokenize", "--decode", "83, 69, 76, 69, 67, 84, 32, 42"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_tokenize_decode_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI tokenize command (decode error)."
    args = ["tokenize", "--decode", "invalid, data"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_tokenize_none(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI tokenize command (neither encode nor decode)."
    args = ["tokenize"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_execute_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI execute command (success)."
    args = ["execute", "--query", "SELECT 1 as num", "--db-type", "sqlite"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_execute_fail(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI execute command (failure)."
    args = ["execute", "--query", "SELECT * FROM non_existent_table", "--db-type", "sqlite"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res != 0


def test_cli_execute_kwargs(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI execute command (with kwargs)."
    args = ["execute", "--query", "SELECT 1 as num", "--db-type", "sqlite", "--db-kwargs", '{"timeout": 10}']
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_embed_duckdb_prompt(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
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
    "Test the CLI embed-duckdb command when duckdb is missing."
    with patch.dict("sys.modules", {"duckdb": None}):
        args = ["embed-duckdb"]
        cli(args)
        capsys.readouterr()


def test_cli_rag_extract(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI rag command (extract action)."
    args = ["rag", "--action", "extract", "--ddl", "CREATE TABLE t (id INT);"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_rag_retrieve(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI rag command (retrieve action)."
    args = ["rag", "--action", "retrieve", "--prompt", "test", "--ddl", "CREATE TABLE t (id INT);"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_serve(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI serve command."
    args = ["serve", "--model", "my-model", "--port", "9000", "--max-batch-size", "128", "--backend", "jax"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_chat(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI chat command."
    args = ["chat", "--prompt", "hello", "--history", '[{"role": "user", "content": "hi"}]', "--backend", "jax"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_chat_invalid_history(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI chat command with invalid history json."
    args = ["chat", "--prompt", "hello", "--history", "invalid_json"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_few_shot(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI few-shot command."
    args = ["few-shot", "--prompt", "hello", "--examples", '[{"input": "in", "output": "out"}]', "--backend", "jax"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_cli_few_shot_invalid_examples(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute logic."""
    "Test the CLI few-shot command with invalid examples json."
    args = ["few-shot", "--prompt", "hello", "--examples", "invalid_json"]
    res = cli(args)
    out, err = capsys.readouterr()
    assert res == 0


def test_dependency_missing_error() -> None:
    """Test function."""
    from gemma_4_sql.exceptions import DependencyMissingError

    assert str(DependencyMissingError("test")) == "test"


def test_cli_serve_branches(monkeypatch, mock_cli_environment, capsys: pytest.CaptureFixture[str]) -> None:
    """Test cli serve branches functionality."""
    import argparse

    import gemma_4_sql.cli_serve as cli_s

    # evaluate
    args = argparse.Namespace(model="m", dataset="d", backend="b", db_path="p", ddl="d", db_type="t", db_kwargs='{"a": 1}')
    monkeypatch.setattr(cli_s, "evaluate", lambda **k: None)
    cli_s.evaluate_cmd(args)

    # agent
    args = argparse.Namespace(model="m", prompt="p", backend="b", db_path="p", ddl="d", db_type="t", max_retries=1, db_kwargs='{"a": 1}', min_confidence=0.0)
    monkeypatch.setattr(cli_s, "run_agentic_loop", lambda **k: None)
    cli_s.agent_cmd(args)

    # chat
    args = argparse.Namespace(model="m", prompt="p", backend="b", history='[{"a": 1}]')
    monkeypatch.setattr(cli_s, "chat_turn", lambda **k: None)
    cli_s.chat_cmd(args)

    # few shot
    args = argparse.Namespace(model="m", prompt="p", backend="b", examples='[{"a": 1}]')
    monkeypatch.setattr(cli_s, "build_few_shot_prompt", lambda **k: None)
    cli_s.few_shot_cmd(args)

    # test bad json
    args = argparse.Namespace(model="m", prompt="p", backend="b", history="bad")
    cli_s.chat_cmd(args)
    args = argparse.Namespace(model="m", prompt="p", backend="b", examples="bad")
    cli_s.few_shot_cmd(args)


def test_cli_if_main(monkeypatch, mock_cli_environment, capsys: pytest.CaptureFixture[str]) -> None:
    """Docstring for test_cli_if_main."""
    import runpy
    import sys

    import gemma_4_sql.cli

    # We patch cli so it returns 0 when called via sys.exit(cli())
    monkeypatch.setattr(gemma_4_sql.cli, "cli", lambda args=None: 0)

    monkeypatch.setattr(sys, "argv", ["gemma_4_sql", "--help"])
    import sys as _sys

    monkeypatch.setattr(_sys, "exit", lambda x: None)

    import argparse

    def mock_parse_args(*args, **kwargs):
        """Docstring for mock_parse_args."""
        ns = argparse.Namespace()
        ns.func = lambda x: 0
        return ns

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", mock_parse_args)

    # Do not catch SystemExit, let it return
    runpy.run_path("src/gemma_4_sql/cli.py", run_name="__main__")
