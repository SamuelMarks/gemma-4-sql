"""
Tests for SDK Agent module.
"""

import pytest

from gemma_4_sql.sdk.agent import run_agentic_loop


def test_agentic_loop_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run_agentic_loop with jax backend."""
    from gemma_4_sql.sdk.registry import get_backend

    jax_agent = get_backend("jax")

    def mock_run_agentic_loop(*args, **kwargs):
        return {"backend": "jax", "status": "mocked"}

    monkeypatch.setattr(jax_agent, "run_agentic_loop", mock_run_agentic_loop)

    res = run_agentic_loop(model_name="model", prompt="prompt", backend="jax")
    assert res["backend"] == "jax"
    assert res["status"] == "mocked"


def test_agentic_loop_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run_agentic_loop with keras backend."""
    from gemma_4_sql.sdk.registry import get_backend

    keras_agent = get_backend("keras")

    def mock_run_agentic_loop(*args, **kwargs):
        return {"backend": "keras", "status": "mocked"}

    monkeypatch.setattr(keras_agent, "run_agentic_loop", mock_run_agentic_loop)

    res = run_agentic_loop(model_name="model", prompt="prompt", backend="keras")
    assert res["backend"] == "keras"
    assert res["status"] == "mocked"


def test_agentic_loop_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run_agentic_loop with maxtext backend."""
    from gemma_4_sql.sdk.registry import get_backend

    maxtext_agent = get_backend("maxtext")

    def mock_run_agentic_loop(*args, **kwargs):
        return {"backend": "maxtext", "status": "mocked"}

    monkeypatch.setattr(maxtext_agent, "run_agentic_loop", mock_run_agentic_loop)

    res = run_agentic_loop(model_name="model", prompt="prompt", backend="maxtext")
    assert res["backend"] == "maxtext"
    assert res["status"] == "mocked"


def test_agentic_loop_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run_agentic_loop with pytorch backend."""
    from gemma_4_sql.sdk.registry import get_backend

    pytorch_agent = get_backend("pytorch")

    def mock_run_agentic_loop(*args, **kwargs):
        return {"backend": "pytorch", "status": "mocked"}

    monkeypatch.setattr(pytorch_agent, "run_agentic_loop", mock_run_agentic_loop)

    res = run_agentic_loop(model_name="model", prompt="prompt", backend="pytorch")
    assert res["backend"] == "pytorch"
    assert res["status"] == "mocked"


def test_agentic_loop_invalid_backend() -> None:
    """Test run_agentic_loop with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        run_agentic_loop(model_name="model", prompt="prompt", backend="invalid")
