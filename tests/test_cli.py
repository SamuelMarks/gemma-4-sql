"""Module docstring."""

from unittest.mock import MagicMock, patch

import pytest
from gemma_4_sql.cli import cli

"\n\nTests for the CLI module.\n"


def test_cli_etl_pretrain(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI ETL pretrain command."""
    args = ["etl", "pretrain", "--dataset", "my-data", "--split", "test", "--batch-size", "16"]
    cli(args)
    captured = capsys.readouterr()
    if "Running pretrain ETL for dataset: my-data" not in captured.out:
        raise AssertionError
    if "status':" not in captured.out:
        raise AssertionError


def test_cli_etl_sft(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI ETL sft command."""
    args = ["etl", "sft", "--dataset", "my-data", "--split", "test", "--batch-size", "16"]
    cli(args)
    captured = capsys.readouterr()
    if "Running SFT ETL for dataset: my-data" not in captured.out:
        raise AssertionError
    if "status':" not in captured.out:
        raise AssertionError


def test_cli_etl_posttrain(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI ETL posttrain command."""
    args = ["etl", "posttrain", "--dataset", "my-data", "--split", "test", "--batch-size", "16"]
    cli(args)
    captured = capsys.readouterr()
    if "Running posttrain ETL for dataset: my-data" not in captured.out:
        raise AssertionError
    if "status':" not in captured.out:
        raise AssertionError


def test_cli_train(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI train command."""
    args = ["train", "--model", "test-model", "--backend", "jax"]
    cli(args)
    captured = capsys.readouterr()
    if "Training from scratch: model=test-model" not in captured.out:
        raise AssertionError
    if "action': 'train_from_scratch" not in captured.out:
        raise AssertionError


def test_cli_pretrain(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI pretrain command."""
    args = ["pretrain", "--model", "test-model", "--backend", "maxtext"]
    cli(args)
    captured = capsys.readouterr()
    if "Pretraining: model=test-model" not in captured.out:
        raise AssertionError
    if "action': 'pretrain" not in captured.out:
        raise AssertionError


def test_cli_sft(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI sft command."""
    args = ["sft", "--model", "test-model", "--backend", "jax"]
    cli(args)
    captured = capsys.readouterr()
    if "SFT: model=test-model" not in captured.out:
        raise AssertionError
    if "action': 'sft" not in captured.out:
        raise AssertionError


def test_cli_posttrain(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI posttrain command."""
    args = ["posttrain", "--model", "test-model", "--backend", "keras"]
    cli(args)
    captured = capsys.readouterr()
    if "Post-training: model=test-model" not in captured.out:
        raise AssertionError
    if "action': 'posttrain" not in captured.out:
        raise AssertionError


def test_cli_dpo(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI dpo command."""
    args = ["dpo", "--model", "test-model", "--backend", "jax", "--beta", "0.2"]
    cli(args)
    captured = capsys.readouterr()
    if "DPO: model=test-model" not in captured.out:
        raise AssertionError
    if "action': 'dpo" not in captured.out:
        raise AssertionError


def test_cli_quantize(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI quantize command."""
    args = ["quantize", "--model", "test-model", "--method", "awq", "--backend", "jax"]
    cli(args)
    captured = capsys.readouterr()
    if "Quantizing: model=test-model" not in captured.out:
        raise AssertionError
    if "'method': 'awq'" not in captured.out:
        raise AssertionError


def test_cli_peft(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI peft command."""
    args = ["peft", "--model", "test-model", "--target-modules", "q_proj,v_proj", "--lora-r", "16", "--lora-alpha", "32", "--lora-dropout", "0.1", "--backend", "pytorch"]
    cli(args)
    captured = capsys.readouterr()
    if "Applying PEFT: model=test-model, backend=pytorch" not in captured.out:
        raise AssertionError
    if "'action': 'apply_lora'" not in captured.out:
        raise AssertionError
    if "'target_modules': ['q_proj', 'v_proj']" not in captured.out:
        raise AssertionError


def test_cli_evaluate(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI evaluate command."""
    args = ["evaluate", "--model", "test-model", "--dataset", "my-data", "--backend", "jax", "--db-type", "sqlite", "--db-kwargs", '{"timeout": 10}']
    cli(args)
    captured = capsys.readouterr()
    if "Evaluating: model=test-model, dataset=my-data" not in captured.out:
        raise AssertionError
    if "status': 'completed'" not in captured.out:
        raise AssertionError


def test_cli_export(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI export command."""
    args = ["export", "--model", "test-model", "--path", "./out", "--backend", "keras"]
    cli(args)
    captured = capsys.readouterr()
    if "Exporting: model=test-model, path=./out, backend=keras" not in captured.out:
        raise AssertionError
    if "status': 'exported_with_keras'" not in captured.out:
        raise AssertionError


def test_cli_generate(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI generate command."""
    args = ["generate", "--model", "test-model", "--prompt", "Find all users", "--backend", "maxtext", "--beam-width", "5", "--max-length", "100"]
    cli(args)
    captured = capsys.readouterr()
    if "Generating: model=test-model, prompt='Find all users', backend=maxtext, beam_width=5, max_length=100" not in captured.out:
        raise AssertionError
    if "status': 'mocked_missing_maxtext'" not in captured.out:
        raise AssertionError


def test_cli_agent(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI agent command."""
    args = ["agent", "--model", "test-model", "--prompt", "Find users", "--backend", "jax", "--db-kwargs", '{"timeout": 10}']
    cli(args)
    captured = capsys.readouterr()
    if "Running Agentic Loop: model=test-model, prompt='Find users', backend=jax" not in captured.out:
        raise AssertionError
    if "status': 'completed'" not in captured.out:
        raise AssertionError
    if "'success': False" not in captured.out:
        raise AssertionError


def test_cli_rag(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI rag command."""
    args = ["rag", "--prompt", "Find users", "--ddl", "CREATE TABLE users (id INT, name VARCHAR);"]
    cli(args)
    captured = capsys.readouterr()
    if "Building RAG prompt..." not in captured.out:
        raise AssertionError
    if "-- Relevant Schema Context:" not in captured.out:
        raise AssertionError
    if "-- Table: users | Columns: id, name" not in captured.out:
        raise AssertionError


def test_cli_log_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI log command."""
    args = ["log", "--step", "100", "--metrics", "loss=0.5,acc=0.9", "--backend", "jax"]
    cli(args)
    captured = capsys.readouterr()
    if "Logging: step=100, metrics=loss=0.5,acc=0.9, log_dir=logs, backend=jax" not in captured.out:
        raise AssertionError
    if "status': 'mocked" not in captured.out:
        raise AssertionError


def test_cli_tokenize_encode(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI tokenize command (encode)."""
    args = ["tokenize", "--encode", "SELECT *"]
    cli(args)
    captured = capsys.readouterr()
    if "Encoded:" not in captured.out:
        raise AssertionError
    if "[83, 69, 76, 69, 67, 84, 32, 42]" not in captured.out:
        raise AssertionError


def test_cli_tokenize_decode(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI tokenize command (decode)."""
    args = ["tokenize", "--decode", "83, 69, 76, 69, 67, 84, 32, 42"]
    cli(args)
    captured = capsys.readouterr()
    if "Decoded:" not in captured.out:
        raise AssertionError
    if "SELECT *" not in captured.out:
        raise AssertionError


def test_cli_tokenize_decode_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI tokenize command (decode error)."""
    args = ["tokenize", "--decode", "invalid, data"]
    cli(args)
    captured = capsys.readouterr()
    if "Error: --decode requires a comma-separated list of integers." not in captured.out:
        raise AssertionError


def test_cli_tokenize_none(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI tokenize command (neither encode nor decode)."""
    args = ["tokenize"]
    cli(args)
    captured = capsys.readouterr()
    if "Must provide either --encode or --decode" not in captured.out:
        raise AssertionError


def test_cli_execute_success(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI execute command (success)."""
    args = ["execute", "--query", "SELECT 1 as num", "--db-type", "sqlite"]
    cli(args)
    captured = capsys.readouterr()
    if "Execution Successful!" not in captured.out:
        raise AssertionError
    if "[(1,)]" not in captured.out:
        raise AssertionError


def test_cli_execute_fail(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI execute command (failure)."""
    args = ["execute", "--query", "SELECT * FROM non_existent_table", "--db-type", "sqlite"]
    cli(args)
    captured = capsys.readouterr()
    if "Execution Failed!" not in captured.out:
        raise AssertionError
    if "no such table: non_existent_table" not in captured.out:
        raise AssertionError


def test_cli_execute_kwargs(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI execute command (with kwargs)."""
    args = ["execute", "--query", "SELECT 1 as num", "--db-type", "sqlite", "--db-kwargs", '{"timeout": 10}']
    cli(args)
    captured = capsys.readouterr()
    if "Execution Successful!" not in captured.out:
        raise AssertionError


def test_cli_embed_duckdb_prompt(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI embed-duckdb command with a prompt."""
    mock_duckdb = MagicMock()
    mock_conn = MagicMock()
    mock_duckdb.connect.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = [("Success!",)]
    with patch.dict("sys.modules", {"duckdb": mock_duckdb}), patch("gemma_4_sql.cli.embed_in_duckdb"):
        args = ["embed-duckdb", "--prompt", "Find users", "--ddl", "CREATE TABLE a (id int);"]
        cli(args)
        captured = capsys.readouterr()
        if "Embedding Gemma in DuckDB:" not in captured.out:
            raise AssertionError
        if "Result: Success!" not in captured.out:
            raise AssertionError


def test_cli_embed_duckdb_no_prompt(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI embed-duckdb command without a prompt."""
    mock_duckdb = MagicMock()
    mock_conn = MagicMock()
    mock_duckdb.connect.return_value = mock_conn
    with patch.dict("sys.modules", {"duckdb": mock_duckdb}), patch("gemma_4_sql.cli.embed_in_duckdb"):
        args = ["embed-duckdb"]
        cli(args)
        captured = capsys.readouterr()
        if "UDF 'ask_gemma' registered. Provide a --prompt to execute it." not in captured.out:
            raise AssertionError


def test_cli_embed_duckdb_missing(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI embed-duckdb command when duckdb is missing."""
    with patch.dict("sys.modules", {"duckdb": None}):
        args = ["embed-duckdb"]
        cli(args)
        captured = capsys.readouterr()
        if "duckdb is required." not in captured.out:
            raise AssertionError


def test_cli_rag_extract(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI rag command (extract action)."""
    args = ["rag", "--action", "extract", "--ddl", "CREATE TABLE t (id INT);"]
    cli(args)
    captured = capsys.readouterr()
    if "Extracting schema entities..." not in captured.out:
        raise AssertionError


def test_cli_rag_retrieve(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI rag command (retrieve action)."""
    args = ["rag", "--action", "retrieve", "--prompt", "test", "--ddl", "CREATE TABLE t (id INT);"]
    cli(args)
    captured = capsys.readouterr()
    if "Retrieving relevant schema..." not in captured.out:
        raise AssertionError


def test_cli_serve(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI serve command."""
    args = ["serve", "--model", "my-model", "--port", "9000", "--max-batch-size", "128", "--backend", "jax"]
    cli(args)
    captured = capsys.readouterr()
    if "Serving: model=my-model, port=9000" not in captured.out:
        raise AssertionError
    if "'backend': 'jax'" not in captured.out:
        raise AssertionError


def test_cli_chat(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI chat command."""
    args = ["chat", "--prompt", "hello", "--history", '[{"role": "user", "content": "hi"}]', "--backend", "jax"]
    cli(args)
    captured = capsys.readouterr()
    if "Chat: model=" not in captured.out:
        raise AssertionError
    if "'backend': 'jax'" not in captured.out:
        raise AssertionError


def test_cli_chat_invalid_history(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI chat command with invalid history json."""
    args = ["chat", "--prompt", "hello", "--history", "invalid_json"]
    cli(args)
    captured = capsys.readouterr()
    if "Error: --history must be a valid JSON list" not in captured.out:
        raise AssertionError


def test_cli_few_shot(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI few-shot command."""
    args = ["few-shot", "--prompt", "hello", "--examples", '[{"input": "in", "output": "out"}]', "--backend", "jax"]
    cli(args)
    captured = capsys.readouterr()
    if "Few-Shot: model=" not in captured.out:
        raise AssertionError
    if "'backend': 'jax'" not in captured.out:
        raise AssertionError


def test_cli_few_shot_invalid_examples(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI few-shot command with invalid examples json."""
    args = ["few-shot", "--prompt", "hello", "--examples", "invalid_json"]
    cli(args)
    captured = capsys.readouterr()
    if "Error: --examples must be a valid JSON list" not in captured.out:
        raise AssertionError
