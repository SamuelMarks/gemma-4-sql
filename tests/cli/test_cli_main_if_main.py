"""Module docstring."""


def test_cli_if_main_execution(monkeypatch):
    """Docstring for test_cli_if_main_execution."""
    import runpy

    import gemma_4_sql.cli

    # We patch cli so it just returns 0 when called inside the module execution
    monkeypatch.setattr(gemma_4_sql.cli, "cli", lambda args=None: 0)

    # We can't rely on mock sys.exit if runpy.run_path is involved,
    # instead we catch the actual SystemExit
    import sys as _sys

    monkeypatch.setattr(_sys, "argv", ["gemma_4_sql", "--help"])

    # This must actually be in the `tests/cli` dir or run from root
    try:
        runpy.run_path("src/gemma_4_sql/cli.py", run_name="__main__")
    except SystemExit as excinfo:
        assert excinfo.code == 0
