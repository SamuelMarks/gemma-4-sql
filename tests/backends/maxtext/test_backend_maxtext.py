"""Module docstring."""

from gemma_4_sql.backends.maxtext import __all__, get_trainer


def test_get_trainer():
    """Docstring for test_get_trainer."""
    assert get_trainer() == "maxtext_trainer"


def test_all_exports():
    """Docstring for test_all_exports."""
    assert "MaxTextHyperparameters" in __all__
    assert "benchmark_model" in __all__
