# Copyright 2024
"""Provide module docstring."""

from typing import NoReturn as Never

import pytest

import gemma_4_sql.backends.mlx.benchmark as bm


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
