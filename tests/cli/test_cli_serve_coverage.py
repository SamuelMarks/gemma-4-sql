"""Module docstring."""

import pytest

from gemma_4_sql.cli import cli


def test_cli_serve_coverage(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Docstring for test_cli_serve_coverage."""
    monkeypatch.setattr("gemma_4_sql.cli_serve.evaluate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("gemma_4_sql.cli_serve.generate", lambda *_args, **_kwargs: {"sql": "", "confidence_score": 0.99})
    monkeypatch.setattr("gemma_4_sql.cli_serve.run_agentic_loop", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("gemma_4_sql.cli_serve.chat_turn", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("gemma_4_sql.cli_serve.build_few_shot_prompt", lambda *_args, **_kwargs: {"few_shot_prompt": "hello"})

    # evaluate without db-kwargs
    cli(["evaluate", "--model", "test-model", "--dataset", "my-data"])

    # generate without show-confidence
    cli(["generate", "--model", "test-model", "--prompt", "test"])

    # agent without db-kwargs
    cli(["agent", "--model", "test-model", "--prompt", "test"])

    # chat with empty history
    cli(["chat", "--model", "test-model", "--prompt", "hi", "--history", ""])

    # few-shot with empty examples
    cli(["few-shot", "--model", "test-model", "--prompt", "hi", "--examples", ""])

    capsys.readouterr()
