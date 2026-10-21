"""Unit and behavioral tests for backend maxtext."""

from gemma_4_sql.backends.maxtext import __all__, get_trainer


def test_get_trainer():
    """Execute and verify `test_get_trainer` behavior and invariants."""
    assert get_trainer() == "maxtext_trainer"


def test_all_exports():
    """Execute and verify `test_all_exports` behavior and invariants."""
    assert "MaxTextHyperparameters" in __all__
    assert "benchmark_model" in __all__
