"""Module docstring."""

import pytest

from gemma_4_sql.cli import cli


def test_cli_log_no_metrics(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Docstring for test_cli_log_no_metrics."""
    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", lambda *_args, **_kwargs: {"status": "completed"})
    args = ["log", "--step", "100", "--backend", "jax"]
    cli(args)
    capsys.readouterr()
