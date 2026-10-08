"""Tests for Keras backend initialization."""

from gemma_4_sql.backends.keras import get_trainer


def test_get_trainer():
    """Docstring for test_get_trainer."""
    assert get_trainer() == "keras_trainer"
