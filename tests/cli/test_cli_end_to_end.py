"""End-to-end CLI tests verifying stdout capture and exit codes for all commands."""

from __future__ import annotations

import json
import math

import pytest

from gemma_4_sql.cli import cli


def test_cli_generate_stdout(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Test generate command outputs plain SQL to stdout.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest capsys fixture.
    """
    monkeypatch.setattr(
        "gemma_4_sql.cli_serve.generate",
        lambda **_k: {"sql": "SELECT * FROM test_table;", "confidence_score": 0.95},
    )
    cli(["generate", "--model", "dummy_model", "--prompt", "show me test data", "--backend", "jax"])
    captured = capsys.readouterr()
    assert captured.out.strip() == "SELECT * FROM test_table;"


def test_cli_generate_empty_sql(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Test generate command when returned SQL is empty string.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest capsys fixture.
    """
    monkeypatch.setattr(
        "gemma_4_sql.cli_serve.generate",
        lambda **_k: {"sql": ""},
    )
    cli(["generate", "--model", "dummy_model", "--prompt", "show me test data", "--backend", "jax"])
    captured = capsys.readouterr()
    assert captured.out == ""


def test_cli_agent_stdout(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Test agent command outputs formatted JSON results to stdout.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest capsys fixture.
    """
    mock_res = {
        "final_sql": "SELECT COUNT(*) FROM users;",
        "results": [(5,)],
        "success": True,
        "retries": 1,
    }
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda **_k: mock_res)
    cli(["agent", "--model", "dummy_model", "--prompt", "how many users", "--backend", "jax"])
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["success"] is True
    assert parsed["final_sql"] == "SELECT COUNT(*) FROM users;"


def test_cli_chat_stdout(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Test chat command prints assistant response to stdout.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest capsys fixture.
    """
    monkeypatch.setattr(
        "gemma_4_sql.cli_serve.chat_turn",
        lambda **_k: {"response": "SELECT name FROM employees WHERE salary > 50000;"},
    )
    cli(["chat", "--model", "dummy_model", "--prompt", "high salary employees", "--backend", "jax"])
    captured = capsys.readouterr()
    assert "SELECT name FROM employees WHERE salary > 50000;" in captured.out


def test_cli_few_shot_stdout(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Test few-shot command outputs formatted prompt to stdout.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest capsys fixture.
    """
    monkeypatch.setattr(
        "gemma_4_sql.cli_serve.build_few_shot_prompt",
        lambda **_k: {"few_shot_prompt": "Prompt with few-shot context examples."},
    )
    cli(["few-shot", "--model", "dummy_model", "--prompt", "my test prompt", "--backend", "jax"])
    captured = capsys.readouterr()
    assert "Prompt with few-shot context examples." in captured.out


def test_cli_evaluate_stdout(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Test evaluate command outputs metric summary to stdout.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest capsys fixture.
    """
    mock_metrics = {
        "exact_match": 0.85,
        "valid_sql": 0.95,
        "execution_accuracy": 0.80,
    }
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda **_k: mock_metrics)
    cli(["evaluate", "--model", "dummy_model", "--dataset", "spider", "--backend", "jax"])
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert math.isclose(parsed["exact_match"], 0.85)
    assert math.isclose(parsed["execution_accuracy"], 0.80)


def test_cli_tokenize_encode_and_decode_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """Test tokenize command encoding and decoding outputs.

    Args:
        capsys: Pytest capsys fixture.
    """
    # Encode test
    cli(["tokenize", "--encode", "SELECT 1;"])
    captured_enc = capsys.readouterr()
    tokens = json.loads(captured_enc.out)
    assert isinstance(tokens, list)
    assert len(tokens) > 0

    # Decode test
    token_str = ",".join(str(t) for t in tokens)
    cli(["tokenize", "--decode", token_str])
    captured_dec = capsys.readouterr()
    assert len(captured_dec.out.strip()) > 0


def test_cli_db_execute_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """Test db execute command runs query and prints JSON result.

    Args:
        capsys: Pytest capsys fixture.
    """
    ddl = "CREATE TABLE tbl (val INT); INSERT INTO tbl VALUES (777);"
    cli(["execute", "--db-path", ":memory:", "--ddl", ddl, "--query", "SELECT val FROM tbl;"])
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["success"] is True
    assert parsed["results"] == [[777]]
    assert parsed["error"] is None


def test_cli_benchmark_stdout(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Test benchmark command outputs performance summary.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest capsys fixture.
    """
    mock_benchmark_res = {
        "backend": "jax",
        "model": "dummy_model",
        "tokens_per_second": 120.5,
        "latency_ms": 8.3,
        "status": "completed",
    }
    monkeypatch.setattr("gemma_4_sql.cli_benchmark.benchmark", lambda **_k: mock_benchmark_res)
    cli(["benchmark", "--model", "dummy_model", "--backend", "jax"])
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert math.isclose(parsed["tokens_per_second"], 120.5)
    assert parsed["status"] == "completed"


def test_cli_exit_code_missing_required_flag() -> None:
    """Test CLI raises SystemExit with non-zero code on missing arguments.

    Returns:
        None.
    """
    with pytest.raises(SystemExit) as exc_info:
        cli(["generate"])
    assert exc_info.value.code != 0


def test_cli_exit_code_invalid_subcommand() -> None:
    """Test CLI raises SystemExit with non-zero code on invalid subcommand.

    Returns:
        None.
    """
    with pytest.raises(SystemExit) as exc_info:
        cli(["nonexistent_action"])
    assert exc_info.value.code != 0


def test_cli_exit_code_database_failure() -> None:
    """Test CLI returns exit code 3 on database execution failure."""
    code = cli(["execute", "--query", "SELECT * FROM nonexistent_table;", "--db-path", ":memory:"])
    assert code == 3


def test_cli_exit_code_success() -> None:
    """Test CLI returns exit code 0 on successful execution."""
    code = cli(["execute", "--query", "SELECT 42;", "--db-path", ":memory:"])
    assert code == 0
