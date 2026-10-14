"""Unit and behavioral tests for keras api."""

from gemma_4_sql.backends.keras import apply_lora, benchmark_model, export_model, get_trainer, quantize_model, run_dpo, serve_model, train_model
from tests.conftest import requires_keras

pytestmark = requires_keras


def test_keras_apis() -> None:
    """Verify behavior of test_keras_apis."""
    assert callable(train_model)
    assert callable(serve_model)
    assert callable(benchmark_model)
    assert callable(run_dpo)
    assert callable(export_model)
    assert callable(quantize_model)
    assert callable(apply_lora)
    assert callable(get_trainer)
    assert get_trainer() == "keras_trainer"
