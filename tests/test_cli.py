from unittest.mock import MagicMock, patch

"""
Tests for the CLI module.
"""

from pytest import CaptureFixture

from gemma_4_sql.cli import cli


def test_cli_etl_pretrain(capsys: CaptureFixture[str]) -> None:
    """Test the CLI ETL pretrain command."""
    args = [
        "etl",
        "pretrain",
        "--dataset",
        "my-data",
        "--split",
        "test",
        "--batch-size",
        "16",
    ]
    cli(args)
    captured = capsys.readouterr()
    assert "Running pretrain ETL for dataset: my-data" in captured.out
    assert "status':" in captured.out


def test_cli_etl_sft(capsys: CaptureFixture[str]) -> None:
    """Test the CLI ETL sft command."""
    args = [
        "etl",
        "sft",
        "--dataset",
        "my-data",
        "--split",
        "test",
        "--batch-size",
        "16",
    ]
    cli(args)
    captured = capsys.readouterr()
    assert "Running SFT ETL for dataset: my-data" in captured.out
    assert "status':" in captured.out


def test_cli_etl_posttrain(capsys: CaptureFixture[str]) -> None:
    """Test the CLI ETL posttrain command."""
    args = [
        "etl",
        "posttrain",
        "--dataset",
        "my-data",
        "--split",
        "test",
        "--batch-size",
        "16",
    ]
    cli(args)
    captured = capsys.readouterr()
    assert "Running posttrain ETL for dataset: my-data" in captured.out
    assert "status':" in captured.out


def test_cli_train(capsys: CaptureFixture[str]) -> None:
    """Test the CLI train command."""
    args = ["train", "--model", "test-model", "--backend", "jax"]
    cli(args)
    captured = capsys.readouterr()
    assert "Training from scratch: model=test-model" in captured.out
    assert "action': 'train_from_scratch" in captured.out


def test_cli_pretrain(capsys: CaptureFixture[str]) -> None:
    """Test the CLI pretrain command."""
    args = ["pretrain", "--model", "test-model", "--backend", "maxtext"]
    cli(args)
    captured = capsys.readouterr()
    assert "Pretraining: model=test-model" in captured.out
    assert "action': 'pretrain" in captured.out


def test_cli_sft(capsys: CaptureFixture[str]) -> None:
    """Test the CLI sft command."""
    args = ["sft", "--model", "test-model", "--backend", "jax"]
    cli(args)
    captured = capsys.readouterr()
    assert "SFT: model=test-model" in captured.out
    assert "action': 'sft" in captured.out


def test_cli_posttrain(capsys: CaptureFixture[str]) -> None:
    """Test the CLI posttrain command."""
    args = ["posttrain", "--model", "test-model", "--backend", "keras"]
    cli(args)
    captured = capsys.readouterr()
    assert "Post-training: model=test-model" in captured.out
    assert "action': 'posttrain" in captured.out


def test_cli_dpo(capsys: CaptureFixture[str]) -> None:
    """Test the CLI dpo command."""
    args = ["dpo", "--model", "test-model", "--backend", "jax", "--beta", "0.2"]
    cli(args)
    captured = capsys.readouterr()
    assert "DPO: model=test-model" in captured.out
    assert "action': 'dpo" in captured.out


def test_cli_quantize(capsys: CaptureFixture[str]) -> None:
    """Test the CLI quantize command."""
    args = ["quantize", "--model", "test-model", "--method", "awq", "--backend", "jax"]
    cli(args)
    captured = capsys.readouterr()
    assert "Quantizing: model=test-model" in captured.out
    assert "'method': 'awq'" in captured.out


def test_cli_peft(capsys: CaptureFixture[str]) -> None:
    """Test the CLI peft command."""
    args = [
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
    ]
    cli(args)
    captured = capsys.readouterr()
    assert "Applying PEFT: model=test-model, backend=pytorch" in captured.out
    assert "'action': 'apply_lora'" in captured.out
    assert "'target_modules': ['q_proj', 'v_proj']" in captured.out


def test_cli_evaluate(capsys: CaptureFixture[str]) -> None:
    """Test the CLI evaluate command."""
    args = [
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
    ]
    cli(args)
    captured = capsys.readouterr()
    assert "Evaluating: model=test-model, dataset=my-data" in captured.out
    assert "status': 'completed'" in captured.out


def test_cli_export(capsys: CaptureFixture[str]) -> None:
    """Test the CLI export command."""
    args = ["export", "--model", "test-model", "--path", "./out", "--backend", "keras"]
    cli(args)
    captured = capsys.readouterr()
    assert "Exporting: model=test-model, path=./out, backend=keras" in captured.out
    assert "status': 'mock_exported'" in captured.out


def test_cli_generate(capsys: CaptureFixture[str]) -> None:
    """Test the CLI generate command."""
    args = [
        "generate",
        "--model",
        "test-model",
        "--prompt",
        "Find all users",
        "--backend",
        "maxtext",
    ]
    cli(args)
    captured = capsys.readouterr()
    assert (
        "Generating: model=test-model, prompt='Find all users', backend=maxtext"
        in captured.out
    )
    assert "status': 'mocked_missing_maxtext'" in captured.out


def test_cli_agent(capsys: CaptureFixture[str]) -> None:
    """Test the CLI agent command."""
    args = [
        "agent",
        "--model",
        "test-model",
        "--prompt",
        "Find users",
        "--backend",
        "jax",
        "--db-kwargs",
        '{"timeout": 10}',
    ]
    cli(args)
    captured = capsys.readouterr()
    assert (
        "Running Agentic Loop: model=test-model, prompt='Find users', backend=jax"
        in captured.out
    )
    assert "status': 'completed'" in captured.out
    assert "'success': False" in captured.out


def test_cli_rag(capsys: CaptureFixture[str]) -> None:
    """Test the CLI rag command."""
    args = [
        "rag",
        "--prompt",
        "Find users",
        "--ddl",
        "CREATE TABLE users (id INT, name VARCHAR);",
    ]
    cli(args)
    captured = capsys.readouterr()
    assert "Building RAG prompt..." in captured.out
    assert "-- Relevant Schema Context:" in captured.out
    assert "-- Table: users | Columns: id, name" in captured.out


def test_cli_log_metrics(capsys: CaptureFixture[str]) -> None:
    """Test the CLI log command."""
    args = ["log", "--step", "100", "--metrics", "loss=0.5,acc=0.9", "--backend", "jax"]
    cli(args)
    captured = capsys.readouterr()
    assert "Logging: step=100, metrics=loss=0.5,acc=0.9, log_dir=logs, backend=jax" in captured.out
    assert "status': 'mocked" in captured.out


def test_cli_tokenize_encode(capsys: CaptureFixture[str]) -> None:
    """Test the CLI tokenize command (encode)."""
    args = ["tokenize", "--encode", "SELECT *"]
    cli(args)
    captured = capsys.readouterr()
    assert "Encoded:" in captured.out
    assert "[83, 69, 76, 69, 67, 84, 32, 42]" in captured.out


def test_cli_tokenize_decode(capsys: CaptureFixture[str]) -> None:
    """Test the CLI tokenize command (decode)."""
    args = ["tokenize", "--decode", "83, 69, 76, 69, 67, 84, 32, 42"]
    cli(args)
    captured = capsys.readouterr()
    assert "Decoded:" in captured.out
    assert "SELECT *" in captured.out


def test_cli_tokenize_decode_error(capsys: CaptureFixture[str]) -> None:
    """Test the CLI tokenize command (decode error)."""
    args = ["tokenize", "--decode", "invalid, data"]
    cli(args)
    captured = capsys.readouterr()
    assert (
        "Error: --decode requires a comma-separated list of integers." in captured.out
    )


def test_cli_tokenize_none(capsys: CaptureFixture[str]) -> None:
    """Test the CLI tokenize command (neither encode nor decode)."""
    args = ["tokenize"]
    cli(args)
    captured = capsys.readouterr()
    assert "Must provide either --encode or --decode" in captured.out


def test_cli_execute_success(capsys: CaptureFixture[str]) -> None:
    """Test the CLI execute command (success)."""
    args = ["execute", "--query", "SELECT 1 as num", "--db-type", "sqlite"]
    cli(args)
    captured = capsys.readouterr()
    assert "Execution Successful!" in captured.out
    assert "[(1,)]" in captured.out


def test_cli_execute_fail(capsys: CaptureFixture[str]) -> None:
    """Test the CLI execute command (failure)."""
    args = [
        "execute",
        "--query",
        "SELECT * FROM non_existent_table",
        "--db-type",
        "sqlite",
    ]
    cli(args)
    captured = capsys.readouterr()
    assert "Execution Failed!" in captured.out
    assert "no such table: non_existent_table" in captured.out


def test_cli_execute_kwargs(capsys: CaptureFixture[str]) -> None:
    """Test the CLI execute command (with kwargs)."""
    args = [
        "execute",
        "--query",
        "SELECT 1 as num",
        "--db-type",
        "sqlite",
        "--db-kwargs",
        '{"timeout": 10}',
    ]
    cli(args)
    captured = capsys.readouterr()
    assert "Execution Successful!" in captured.out


def test_cli_embed_duckdb_prompt(capsys: CaptureFixture[str]) -> None:
    """Test the CLI embed-duckdb command with a prompt."""

    mock_duckdb = MagicMock()
    mock_conn = MagicMock()
    mock_duckdb.connect.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = [("Success!",)]

    with patch.dict("sys.modules", {"duckdb": mock_duckdb}):
        with patch("gemma_4_sql.cli.embed_in_duckdb"):
            args = [
                "embed-duckdb",
                "--prompt",
                "Find users",
                "--ddl",
                "CREATE TABLE a (id int);",
            ]
            cli(args)
            captured = capsys.readouterr()
            assert "Embedding Gemma in DuckDB:" in captured.out
            assert "Result: Success!" in captured.out


def test_cli_embed_duckdb_no_prompt(capsys: CaptureFixture[str]) -> None:
    """Test the CLI embed-duckdb command without a prompt."""

    mock_duckdb = MagicMock()
    mock_conn = MagicMock()
    mock_duckdb.connect.return_value = mock_conn

    with patch.dict("sys.modules", {"duckdb": mock_duckdb}):
        with patch("gemma_4_sql.cli.embed_in_duckdb"):
            args = ["embed-duckdb"]
            cli(args)
            captured = capsys.readouterr()
            assert (
                "UDF 'ask_gemma' registered. Provide a --prompt to execute it."
                in captured.out
            )


def test_cli_embed_duckdb_missing(capsys: CaptureFixture[str]) -> None:
    """Test the CLI embed-duckdb command when duckdb is missing."""

    with patch.dict("sys.modules", {"duckdb": None}):
        args = ["embed-duckdb"]
        cli(args)
        captured = capsys.readouterr()
        assert "duckdb is required." in captured.out


def test_cli_rag_extract(capsys: CaptureFixture[str]) -> None:
    """Test the CLI rag command (extract action)."""
    args = ["rag", "--action", "extract", "--ddl", "CREATE TABLE t (id INT);"]
    cli(args)
    captured = capsys.readouterr()
    assert "Extracting schema entities..." in captured.out


def test_cli_rag_retrieve(capsys: CaptureFixture[str]) -> None:
    """Test the CLI rag command (retrieve action)."""
    args = [
        "rag",
        "--action",
        "retrieve",
        "--prompt",
        "test",
        "--ddl",
        "CREATE TABLE t (id INT);",
    ]
    cli(args)
    captured = capsys.readouterr()
    assert "Retrieving relevant schema..." in captured.out


def test_cli_serve(capsys: CaptureFixture[str]) -> None:
    """Test the CLI serve command."""
    args = ["serve", "--model", "my-model", "--port", "9000", "--max-batch-size", "128", "--backend", "jax"]
    cli(args)
    captured = capsys.readouterr()
    assert "Serving: model=my-model, port=9000" in captured.out
    assert "'backend': 'jax'" in captured.out


def test_cli_chat(capsys: CaptureFixture[str]) -> None:
    """Test the CLI chat command."""
    args = ["chat", "--prompt", "hello", "--history", '[{"role": "user", "content": "hi"}]', "--backend", "jax"]
    cli(args)
    captured = capsys.readouterr()
    assert "Chat: model=" in captured.out
    assert "'backend': 'jax'" in captured.out

def test_cli_chat_invalid_history(capsys: CaptureFixture[str]) -> None:
    """Test the CLI chat command with invalid history json."""
    args = ["chat", "--prompt", "hello", "--history", "invalid_json"]
    cli(args)
    captured = capsys.readouterr()
    assert "Error: --history must be a valid JSON list" in captured.out


def test_cli_few_shot(capsys: CaptureFixture[str]) -> None:
    """Test the CLI few-shot command."""
    args = ["few-shot", "--prompt", "hello", "--examples", '[{"input": "in", "output": "out"}]', "--backend", "jax"]
    cli(args)
    captured = capsys.readouterr()
    assert "Few-Shot: model=" in captured.out
    assert "'backend': 'jax'" in captured.out

def test_cli_few_shot_invalid_examples(capsys: CaptureFixture[str]) -> None:
    """Test the CLI few-shot command with invalid examples json."""
    args = ["few-shot", "--prompt", "hello", "--examples", "invalid_json"]
    cli(args)
    captured = capsys.readouterr()
    assert "Error: --examples must be a valid JSON list" in captured.out
