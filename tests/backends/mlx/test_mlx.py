"""Tests for MLX backend."""

from typing import Never

import pytest

import gemma_4_sql.backends.mlx.benchmark as bm
import gemma_4_sql.backends.mlx.inference as inf
import gemma_4_sql.backends.mlx.train as tr
from gemma_4_sql.backends.mlx import dpo, etl, export, logging, peft, quantize


def test_train_mocked() -> None:
    """Execute function."""
    res = tr.train_model("sft", "mod", "dat", 1, 0.1)
    if res["status"] != "mocked_missing_mlx":
        raise AssertionError


def test_etl_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(etl, "datasets", None)
    res = etl.build_dataloader("ds", "train")
    if res["status"] != "mocked":
        raise AssertionError


def test_inference_mocked() -> None:
    """Execute function."""
    res = inf.generate_sql("mod", "prompt")
    if res["status"] != "mocked_missing_mlx":
        raise AssertionError


def test_peft_mocked() -> None:
    """Execute function."""
    res = peft.apply_lora("mod", ["q_proj"], 8, 16, 0.1)
    if res["status"] != "mocked_missing_mlx":
        raise AssertionError
    if dpo.run_dpo("m", "d")["backend"] != "mlx":
        raise AssertionError
    if export.export_model("m", "p")["backend"] != "mlx":
        raise AssertionError
    if logging.log_metrics({"l": 1.0}, 1, "d")["backend"] != "mlx":
        raise AssertionError
    if quantize.quantize_model("m", "awq")["backend"] != "mlx":
        raise AssertionError


class MockMlxCuda:
    """Provide class docstring."""

    @staticmethod
    def is_available() -> bool:
        """Execute function."""
        return True

    @staticmethod
    def synchronize() -> None:
        """Execute function."""

    @staticmethod
    def max_memory_allocated() -> float:
        """Execute function."""
        return 1024 * 1024 * 100


class MockMlxTensor:
    """Provide class docstring."""

    def to(self, _device: object) -> object:
        """Execute function."""
        return self


class MockMlx:
    """Provide class docstring."""

    cuda = MockMlxCuda()
    long = "long"

    @staticmethod
    def zeros(_shape: object, _dtype: object) -> object:
        """Execute function."""
        return MockMlxTensor()

    class MockNoGrad:
        """Provide class docstring."""

        def __enter__(self) -> object:
            """Execute function."""

        def __exit__(self, *args: object) -> object:
            """Execute function."""

    no_grad = MockNoGrad


class MockModel:
    """Provide class docstring."""

    def to(self, device: object) -> None:
        """Execute function."""

    def eval(self) -> None:
        """Execute function."""

    def __call__(self, x: object) -> object:
        """Execute function."""
        return x


class MockAutoModel:
    """Provide class docstring."""

    @staticmethod
    def from_pretrained(_name: object) -> object:
        """Execute function."""
        return MockModel()


def test_benchmark_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(bm, "mlx", MockMlx())
    monkeypatch.setattr(bm, "AutoModelForCausalLM", MockAutoModel())
    res = bm.benchmark_model("mod", "gpu", 2, test_mode=False)
    if res["status"] != "success":
        raise AssertionError
    res = bm.benchmark_model("mod", "cpu", 2, test_mode=False)
    if res["status"] != "success":
        raise AssertionError

    def mock_fail(*_args: object, **_kwargs: object) -> Never:
        """Execute function."""
        msg = "failed!"
        raise RuntimeError(msg)

    monkeypatch.setattr(MockAutoModel, "from_pretrained", mock_fail)
    res = bm.benchmark_model("mod", "gpu", 2, test_mode=False)
    if "failed" not in res["status"]:
        raise AssertionError
