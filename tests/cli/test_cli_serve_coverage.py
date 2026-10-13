import argparse
import json

from gemma_4_sql.cli_serve import agent_cmd, chat_cmd, evaluate_cmd, few_shot_cmd, generate_cmd


def test_evaluate_cmd_with_db_kwargs(monkeypatch, capsys):
    args = argparse.Namespace(
        model="dummy",
        dataset="dummy",
        backend="dummy",
        db_path="dummy",
        ddl="dummy",
        db_type="dummy",
        db_kwargs=json.dumps({"key": "value"}),
    )

    def mock_evaluate(**kwargs):
        assert kwargs["db_kwargs"] == {"key": "value"}
        return {"status": "evaluated"}

    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", mock_evaluate)
    evaluate_cmd(args)
    captured = capsys.readouterr()
    assert "evaluated" in captured.out


def test_generate_cmd_empty_sql(monkeypatch, capsys):
    args = argparse.Namespace(
        model="dummy",
        prompt="dummy",
        backend="dummy",
        beam_width=3,
        max_length=50,
        show_confidence=False,
    )

    def mock_generate(**kwargs):
        return {"sql": ""}

    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", mock_generate)
    generate_cmd(args)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_agent_cmd_with_db_kwargs(monkeypatch, capsys):
    args = argparse.Namespace(
        model="dummy",
        prompt="dummy",
        backend="dummy",
        db_path="dummy",
        ddl="dummy",
        db_type="dummy",
        max_retries=1,
        db_kwargs=json.dumps({"key": "value"}),
    )

    def mock_run_agentic_loop(**kwargs):
        assert kwargs["db_kwargs"] == {"key": "value"}
        return {"status": "agent_run"}

    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", mock_run_agentic_loop)
    agent_cmd(args)
    captured = capsys.readouterr()
    assert "agent_run" in captured.out


def test_chat_cmd_with_history(monkeypatch, capsys):
    args = argparse.Namespace(model="dummy", prompt="dummy", backend="dummy", history=json.dumps([{"role": "user", "content": "hi"}]))

    def mock_chat_turn(**kwargs):
        assert kwargs["history"] == [{"role": "user", "content": "hi"}]
        return {"response": "hello"}

    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", mock_chat_turn)
    chat_cmd(args)
    captured = capsys.readouterr()
    assert "hello" in captured.out


def test_chat_cmd_with_invalid_history(capsys):
    args = argparse.Namespace(model="dummy", prompt="dummy", backend="dummy", history="invalid json")
    chat_cmd(args)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_few_shot_cmd_with_examples(monkeypatch, capsys):
    args = argparse.Namespace(model="dummy", prompt="dummy", backend="dummy", examples=json.dumps([{"prompt": "q", "sql": "s"}]))

    def mock_build_few_shot_prompt(**kwargs):
        assert kwargs["examples"] == [{"prompt": "q", "sql": "s"}]
        return {"few_shot_prompt": "prompt with examples"}

    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", mock_build_few_shot_prompt)
    few_shot_cmd(args)
    captured = capsys.readouterr()
    assert "prompt with examples" in captured.out


def test_few_shot_cmd_with_invalid_examples(capsys):
    args = argparse.Namespace(model="dummy", prompt="dummy", backend="dummy", examples="invalid json")
    few_shot_cmd(args)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_few_shot_cmd_empty_prompt(monkeypatch, capsys):
    args = argparse.Namespace(model="dummy", prompt="dummy", backend="dummy", examples="")

    def mock_build(**kwargs):
        return {"few_shot_prompt": ""}

    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", mock_build)
    few_shot_cmd(args)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_serve_cmd(monkeypatch, capsys):
    args = argparse.Namespace(
        model="dummy",
        port=8000,
        max_batch_size=32,
        backend="dummy",
        run_server=False,
    )

    def mock_serve_model(**kwargs):
        return {"status": "serving"}

    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", mock_serve_model)
    from gemma_4_sql.cli_serve import serve_cmd

    serve_cmd(args)
    captured = capsys.readouterr()
    assert "Serving model dummy on port 8000 (status: serving)" in captured.out


def test_serve_cmd_none(monkeypatch, capsys):
    args = argparse.Namespace(
        model="dummy",
        port=8000,
        max_batch_size=32,
        backend="dummy",
        run_server=False,
    )

    def mock_serve_model(**kwargs):
        return None

    monkeypatch.setattr("gemma_4_sql.cli_serve.serve_model", mock_serve_model)
    from gemma_4_sql.cli_serve import serve_cmd

    serve_cmd(args)
    captured = capsys.readouterr()
    assert "Serving model dummy on port 8000 (status: unknown)" in captured.out


def test_generate_cmd_with_confidence(monkeypatch, capsys):
    import argparse

    from gemma_4_sql.cli_serve import generate_cmd

    args = argparse.Namespace(
        model="dummy",
        prompt="Test prompt",
        backend="dummy",
        beam_width=3,
        max_length=50,
        show_confidence=True,
    )

    def mock_generate(**kwargs):
        return {"sql": "SELECT 1;", "confidence_score": 0.95}

    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", mock_generate)
    generate_cmd(args)
    captured = capsys.readouterr()
    assert "SELECT 1;" in captured.out
    assert "Confidence: 0.9500" in captured.out
