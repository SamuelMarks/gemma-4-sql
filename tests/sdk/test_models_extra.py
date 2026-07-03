"""Tests for missing models.py coverage."""

import pytest

from gemma_4_sql.sdk.models import TrainingConfig, train_from_scratch


def test_train_from_scratch_pytorch(monkeypatch: object) -> object:
    """Initialize function test_train_from_scratch_pytorch.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    pt_train = get_backend("pytorch")
    monkeypatch.setattr(pt_train, "train_model", lambda **_kw: {"status": "mock"})
    res = train_from_scratch(TrainingConfig(model_name="mock", dataset="mock", backend="pytorch"))
    if not res == {"status": "mock"}:
        raise AssertionError


def test_train_from_scratch_unknown() -> object:
    """Initialize function test_train_from_scratch_unknown."""
    with pytest.raises(ValueError, match=r".*"):
        train_from_scratch(TrainingConfig(model_name="mock", dataset="mock", backend="unknown"))
