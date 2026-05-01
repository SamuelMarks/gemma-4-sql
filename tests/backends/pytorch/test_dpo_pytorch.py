"""
Tests for PyTorch DPO logic.
"""

from __future__ import annotations

from typing import Any

import pytest

from gemma_4_sql.backends.pytorch.dpo import dpo_loss, run_dpo


class MockTensor:
    def __sub__(self, other: Any) -> MockTensor:
        return MockTensor()

    def __mul__(self, other: Any) -> MockTensor:
        return MockTensor()

    def __rmul__(self, other: Any) -> MockTensor:
        return MockTensor()

    def __neg__(self) -> MockTensor:
        return MockTensor()

    def item(self) -> float:
        return 0.42

    def mean(self) -> MockTensor:
        return MockTensor()

    def detach(self) -> MockTensor:
        return MockTensor()


class MockTorch:
    def tensor(self, x: Any) -> MockTensor:
        return MockTensor()


class MockF:
    def logsigmoid(self, x: Any) -> MockTensor:
        return MockTensor()


def test_run_dpo_pytorch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch DPO when missing."""
    import gemma_4_sql.backends.pytorch.dpo as torch_dpo

    monkeypatch.setattr(torch_dpo, "torch", None)
    monkeypatch.setattr(torch_dpo, "F", None)

    res = run_dpo("model", "data")
    assert res["status"] == "mocked_missing_torch"
    assert res["final_loss"] == 0.0

    loss, ch_r, re_r = dpo_loss(None, None, None, None)
    assert loss == 0.0
    assert ch_r == 0.0
    assert re_r == 0.0


def test_run_dpo_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch DPO."""
    import gemma_4_sql.backends.pytorch.dpo as torch_dpo

    monkeypatch.setattr(torch_dpo, "torch", MockTorch())
    monkeypatch.setattr(torch_dpo, "F", MockF())

    res = run_dpo("model", "data")
    assert res["backend"] == "pytorch"
    assert res["status"] == "completed"
    assert res["final_loss"] == 0.42
