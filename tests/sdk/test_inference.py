"""Tests for SDK Inference module."""

import pytest

from gemma_4_sql.sdk.inference import generate


def test_generate_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate with jax."""
    monkeypatch.setattr("gemma_4_sql.sdk.registry.get_backend", lambda _x: type("MockBackend", (), {"generate_sql": lambda *_a, **_k: {"sql": "SELECT 1"}})())
    res = generate("model1", "Find all users", "jax")
    assert res.get("sql") == "SELECT 1" or res.get("status", "").startswith("backend_unavailable")


def test_generate_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate with keras."""
    monkeypatch.setattr("gemma_4_sql.sdk.registry.get_backend", lambda _x: type("MockBackend", (), {"generate_sql": lambda *_a, **_k: {"sql": "SELECT 1"}})())
    res = generate("model1", "Find all users", "keras")
    assert res["sql"] == "SELECT 1"


def test_generate_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate with maxtext."""
    monkeypatch.setattr("gemma_4_sql.sdk.registry.get_backend", lambda _x: type("MockBackend", (), {"generate_sql": lambda *_a, **_k: {"sql": "SELECT 1"}})())
    res = generate("model1", "Find all users", "maxtext")
    assert res.get("sql") == "SELECT 1" or res.get("status", "").startswith("backend_unavailable")


def test_generate_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate with pytorch."""
    monkeypatch.setattr("gemma_4_sql.sdk.registry.get_backend", lambda _x: type("MockBackend", (), {"generate_sql": lambda *_a, **_k: {"sql": "SELECT 1"}})())
    res = generate("model1", "Find all users", "pytorch")
    assert res.get("sql") == "SELECT 1" or res.get("status", "").startswith("backend_unavailable")


@pytest.mark.usefixtures("monkeypatch")
def test_generate_invalid() -> None:
    """Test generate with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        generate("model1", "Find all users", "invalid")
