"""Tests for Keras backend initialization."""

from gemma_4_sql.backends.keras import get_trainer
from tests.conftest import requires_keras

pytestmark = requires_keras


def test_get_trainer():
    """Execute and verify `test_get_trainer` behavior and invariants."""
    assert get_trainer() == "keras_trainer"
