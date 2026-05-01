"""Tests for missing models.py coverage."""

import pytest

from gemma_4_sql.sdk.models import train_from_scratch


def test_train_from_scratch_pytorch(monkeypatch):
    from gemma_4_sql.sdk.registry import get_backend
    pt_train = get_backend("pytorch")

    monkeypatch.setattr(pt_train, "train_model", lambda **kw: {"status": "mock"})
    res = train_from_scratch(model_name="mock", dataset="mock", backend="pytorch")
    assert res == {"status": "mock"}


def test_train_from_scratch_unknown():
    with pytest.raises(ValueError):
        train_from_scratch(model_name="mock", dataset="mock", backend="unknown")
