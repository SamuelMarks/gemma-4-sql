"""Unit and behavioral tests for backend maxtext."""

from gemma_4_sql.backends.maxtext import __all__, get_trainer


def test_get_trainer():
    """Verify get trainer behavior."""
    assert get_trainer() == "maxtext_trainer"


def test_all_exports():
    """Verify all exports behavior."""
    assert "MaxTextHyperparameters" in __all__
    assert "benchmark_model" in __all__
