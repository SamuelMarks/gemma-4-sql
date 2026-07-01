"""Tests for PyTorch Benchmark."""

from __future__ import annotations

import typing

import gemma_4_sql.backends.pytorch.benchmark as pt_bm
from gemma_4_sql.backends.pytorch.benchmark import benchmark_model

if typing.TYPE_CHECKING:
    import pytest


class MockTorch:
    long = "long"

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    def zeros(self, *args: object, **kwargs: object) -> object:
        return [0]


class MockAutoModelForCausalLM:
    pass


def test_benchmark_pytorch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt_bm, "torch", None)
    monkeypatch.setattr(pt_bm, "AutoModelForCausalLM", None)

    res = benchmark_model("model", "gpu", 1)
    if not res["status"] == "mocked_missing_torch":
        raise AssertionError


def test_benchmark_pytorch_real(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt_bm, "torch", MockTorch())
    monkeypatch.setattr(pt_bm, "AutoModelForCausalLM", MockAutoModelForCausalLM)

    res = benchmark_model("model", "gpu", 1, test_mode=True, num_runs=2)
    if not res["status"] == "success":
        raise AssertionError
    if not res["tokens_per_sec"] > 0:
        raise AssertionError


def test_benchmark_pytorch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt_bm, "torch", MockTorch())
    monkeypatch.setattr(pt_bm, "AutoModelForCausalLM", MockAutoModelForCausalLM)

    def raise_err(*args: object, **kwargs: object) -> object:
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(MockTorch, "zeros", raise_err)

    res = benchmark_model("model", "gpu", 1, test_mode=True)
    if "failed" not in str(res["status"]):
        raise AssertionError


def test_benchmark_test_mode(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.pytorch.benchmark as m_benchmark

    class MockTorch:
        zeros = lambda *args, **kwargs: type("T", (), {"to": lambda self, d: self})()
        long = 1
        no_grad = lambda: type("CM", (), {"__enter__": lambda self: None, "__exit__": lambda self, *a: None})()

        class cuda:
            is_available = lambda: True
            synchronize = lambda: None
            max_memory_allocated = lambda: 1024 * 1024 * 10
            reset_peak_memory_stats = lambda: None

    monkeypatch.setattr(m_benchmark, "torch", MockTorch)
    monkeypatch.setattr(m_benchmark, "AutoModelForCausalLM", object())
    res = m_benchmark.benchmark_model("m", "cuda", 1, test_mode=True)
    assert res["status"] == "success"


def test_chat_test_mode(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.pytorch.chat as m_chat

    monkeypatch.setattr(m_chat, "AutoTokenizer", object())
    monkeypatch.setattr(m_chat, "torch", object())
    res = m_chat.chat_turn("m", [], "prompt", test_mode=True)
    assert res["status"] == "success_pytorch_chat"


def test_chat_error(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.pytorch.chat as m_chat

    monkeypatch.setattr(m_chat, "AutoTokenizer", object())
    monkeypatch.setattr(m_chat, "torch", object())

    def raise_err(*args, **kwargs):
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(m_chat, "generate_sql", raise_err)

    class MockTokenizerObj:
        def apply_chat_template(self, *args, **kwargs):
            return "tmpl"

    class MockTokenizer:
        from_pretrained = lambda *args, **kwargs: MockTokenizerObj()

    monkeypatch.setattr(m_chat, "AutoTokenizer", MockTokenizer)

    res = m_chat.chat_turn("m", [], "prompt", test_mode=False)
    assert "failed" in res["status"]


def test_evaluate_real_full(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.pytorch.evaluate as m_eval

    monkeypatch.setattr(m_eval, "generate_sql", lambda *args, **kwargs: {"sql": "SELECT 1"})

    class MockEngine:
        def execute_with_feedback(self, p):
            return (True, None, None)

        def compare_queries(self, p, t):
            return True

        def close(self):
            pass

    monkeypatch.setattr(m_eval, "LiveDatabaseEngine", lambda *args, **kwargs: MockEngine())

    res = m_eval.evaluate_model("m", "ds", test_mode=False)
    assert res["status"] == "completed"


def test_evaluate_mock_preds(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.pytorch.evaluate as m_eval

    monkeypatch.setattr(m_eval, "generate_sql", lambda *args, **kwargs: {"sql": "SELECT 1"})

    class MockEngine:
        def execute_with_feedback(self, p):
            return (True, None, None)

        def compare_queries(self, p, t):
            return True

        def close(self):
            pass

    monkeypatch.setattr(m_eval, "LiveDatabaseEngine", lambda *args, **kwargs: MockEngine())
    res = m_eval.evaluate_model("m", "ds", test_mode=False, mock_predictions=["SELECT 1"], mock_truths=["SELECT 1"])
    assert res["status"] == "completed"


def test_evaluate_dataloader(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.pytorch.evaluate as m_eval

    monkeypatch.setattr(m_eval, "generate_sql", lambda *args, **kwargs: {"sql": "SELECT 1"})

    class MockEngine:
        def execute_with_feedback(self, p):
            return (True, None, None)

        def compare_queries(self, p, t):
            return True

        def close(self):
            pass

    monkeypatch.setattr(m_eval, "LiveDatabaseEngine", lambda *args, **kwargs: MockEngine())

    class MockLoader:
        def __iter__(self):
            for _ in range(12):
                yield {"inputs": [[1]], "targets": [[2]]}

    monkeypatch.setattr(m_eval, "build_dataloader", lambda *args, **kwargs: {"loader": MockLoader()})

    res = m_eval.evaluate_model("m", "ds", test_mode=False)
    assert res["status"] == "completed"
