"""Tests for MLX backend."""

from typing import NoReturn as Never

import pytest

import gemma_4_sql.backends.mlx.benchmark as bm
import gemma_4_sql.backends.mlx.inference as inf
import gemma_4_sql.backends.mlx.train as tr
from gemma_4_sql.backends.mlx import etl, export, peft
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ETLConfig, TrainingConfig


def test_train_mocked() -> None:
    """Execute function."""
    with pytest.raises(DependencyMissingError):
        tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=1, learning_rate=0.1))


def test_etl_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(etl, "datasets", None)
    with pytest.raises(DependencyMissingError):
        etl.build_dataloader(ETLConfig(dataset_name="ds", split="train"))


def test_inference_mocked() -> None:
    """Execute function."""
    with pytest.raises(DependencyMissingError):
        inf.generate_sql("mod", "prompt")


def test_peft_mocked() -> None:
    """Execute function."""
    with pytest.raises(DependencyMissingError):
        peft.apply_lora("mod", ["q_proj"], 8, 16, 0.1)


def test_export_mlx_none(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> None:
    """Test MLX export when mx is None."""
    monkeypatch.setattr(export, "mx", None)
    with pytest.raises(RuntimeError, match="MLX is not installed, cannot export model."):
        export.export_model("m", str(tmp_path))


class MockMlxCuda:
    """Provide class docstring."""

    @staticmethod
    def is_available() -> bool:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return True

    @staticmethod
    def synchronize() -> None:
        """Execute function."""

    @staticmethod
    def max_memory_allocated() -> float:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return 1024 * 1024 * 100


class MockMlxTensor:
    """Provide class docstring."""

    def to(self, _device: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return self


class MockMlx:
    """Provide class docstring."""

    cuda = MockMlxCuda()
    long = "long"

    @staticmethod
    def zeros(_shape: object, dtype: object = None) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
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
        """Execute function.

        Returns:
            object: Description of return.

        """
        return x


class MockAutoModel:
    """Provide class docstring."""

    @staticmethod
    def from_pretrained(_name: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockModel()


def test_benchmark_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(bm, "mlx", MockMlx())
    monkeypatch.setattr(bm, "AutoModelForCausalLM", MockAutoModel())
    res = bm.benchmark_model("mod", "gpu", 2, test_mode=False)
    if res["status"] != "success":
        raise AssertionError
    res = bm.benchmark_model("mod", "cpu", 2, test_mode=False)
    if res["status"] != "success":
        raise AssertionError

    def mock_fail(*_args: object, **_kwargs: object) -> Never:
        """Execute function.

        Raises:
            RuntimeError: Description.

        """
        msg = "failed!"
        raise RuntimeError(msg)

    monkeypatch.setattr(MockAutoModel, "from_pretrained", mock_fail)
    res = bm.benchmark_model("mod", "gpu", 2, test_mode=False)
    if "failed" not in res["status"]:
        raise AssertionError
