"""Tests for MLX benchmark without MagicMock."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import gemma_4_sql.backends.mlx.benchmark as benchmark_module
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import DTypeProtocol, MlxArrayProtocol
from tests.conftest import requires_mlx
from tests.utils.tensor_helpers import (
    TypedMlxLinearProtocol,
    TypedMlxSequentialProtocol,
    get_typed_mlx_nn,
    get_typed_mx,
    mx_eval_model,
)

pytestmark = requires_mlx


def test_mlx_benchmark_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test mlx benchmark missing deps."""
    monkeypatch.setattr(benchmark_module, "mx", None)
    monkeypatch.setattr(benchmark_module, "load", None)

    with pytest.raises(DependencyMissingError, match="MLX dependencies"):
        benchmark_module._load_mlx_model_and_device("model", "cpu")

    with pytest.raises(DependencyMissingError, match="MLX dependencies"):
        benchmark_module._run_benchmark_pass(None, 1, 1)


def test_load_mlx_model_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _load_mlx_model_and_device across cpu, gpu, single return, and device setter errors."""
    mlx_nn = get_typed_mlx_nn()

    lin = mlx_nn.Linear(4, 4)

    class _FakeMxWithSetDevice:
        """Typed `mlx.core` double supporting `Device`, `cpu`, `gpu`, and `set_default_device`."""

        cpu = "cpu"
        gpu = "gpu"

        def __init__(self, *, fail_set: bool = False) -> None:
            """Initialize device tracking."""
            self.devices_set: list[str] = []
            self.fail_set = fail_set

        @staticmethod
        def Device(dev: str) -> str:
            """Return `dev`."""
            return dev

        def set_default_device(self, dev: str) -> None:
            """Record `dev` or raise `ValueError`.

            Raises:
                ValueError: If `self.fail_set` is True.
            """
            if self.fail_set:
                raise ValueError("error")
            self.devices_set.append(dev)

    class _FakeMxWithoutSetDevice:
        """Minimal `mlx.core` double without `set_default_device`."""

    fake_mx = _FakeMxWithSetDevice()
    return_tuple = True

    def _fake_load(_name: str) -> tuple[TypedMlxLinearProtocol, str] | TypedMlxLinearProtocol:
        """Return `(lin, 'tok')` or `lin`."""
        return (lin, "tok") if return_tuple else lin

    monkeypatch.setattr(benchmark_module, "mx", fake_mx)
    monkeypatch.setattr(benchmark_module, "load", _fake_load)

    model, device = benchmark_module._load_mlx_model_and_device("model", "cpu")
    assert model is lin
    assert device == "cpu"
    assert fake_mx.devices_set == ["cpu"]

    _, device2 = benchmark_module._load_mlx_model_and_device("model", "gpu")
    assert device2 == "gpu"

    return_tuple = False
    model3, _ = benchmark_module._load_mlx_model_and_device("model", "gpu")
    assert model3 is lin

    fake_mx.fail_set = True
    benchmark_module._load_mlx_model_and_device("model", "gpu")

    monkeypatch.setattr(benchmark_module, "mx", _FakeMxWithoutSetDevice())
    model4, dev4 = benchmark_module._load_mlx_model_and_device("model", "gpu")
    assert dev4 == "gpu" and model4 is lin


def test_sync_and_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _sync_and_eval on list and scalar inputs."""
    evaluated: list[tuple[str, ...]] = []

    class _EvalMxStub:
        """Typed `mlx.core` double recording `eval` calls."""

        @staticmethod
        def eval(*args: str) -> None:
            """Append `args`."""
            evaluated.append(tuple(args))

    monkeypatch.setattr(benchmark_module, "mx", _EvalMxStub())

    benchmark_module._sync_and_eval(["t1", "t2"])
    assert evaluated[-1] == ("t1", "t2")

    benchmark_module._sync_and_eval("t1")
    assert evaluated[-1] == ("t1",)

    monkeypatch.setattr(benchmark_module, "mx", None)
    benchmark_module._sync_and_eval("t1")


def test_get_peak_memory_mb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _get_peak_memory_mb across Metal peak, active, top-level peak, and unavailable branches."""

    class _FullMetalStub:
        """Metal namespace supporting `is_available`, `get_peak_memory`, and `get_active_memory`."""

        def __init__(self, *, avail: bool = True, peak: int = 10 * 1048576, active: int = 5 * 1048576) -> None:
            """Store memory counters."""
            self.avail = avail
            self.peak = peak
            self.active = active

        def is_available(self) -> bool:
            """Return `self.avail`."""
            return self.avail

        def get_peak_memory(self) -> int:
            """Return `self.peak`."""
            return self.peak

        def get_active_memory(self) -> int:
            """Return `self.active`."""
            return self.active

    class _ActiveOnlyMetalStub:
        """Metal namespace without `get_peak_memory`."""

        @staticmethod
        def is_available() -> bool:
            """Return `True`."""
            return True

        @staticmethod
        def get_active_memory() -> int:
            """Return 3 MiB."""
            return 3 * 1048576

    class _AvailOnlyMetalStub:
        """Metal namespace with neither peak nor active getters."""

        @staticmethod
        def is_available() -> bool:
            """Return `True`."""
            return True

    class _MxWithMetalAndTopPeak:
        """MLX stub with `.metal` and top-level `get_peak_memory`."""

        def __init__(self, metal_obj: _FullMetalStub | _ActiveOnlyMetalStub | _AvailOnlyMetalStub) -> None:
            """Store `metal_obj`."""
            self.metal = metal_obj

        @staticmethod
        def get_peak_memory() -> int:
            """Return 2 MiB."""
            return 2 * 1048576

    class _EmptyMxStub:
        """MLX stub with neither `.metal` nor `get_peak_memory`."""

    monkeypatch.setattr(benchmark_module, "mx", _MxWithMetalAndTopPeak(_FullMetalStub(peak=10 * 1048576)))
    assert benchmark_module._get_peak_memory_mb() == 10.0

    monkeypatch.setattr(benchmark_module, "mx", _MxWithMetalAndTopPeak(_FullMetalStub(peak=0, active=5 * 1048576)))
    assert benchmark_module._get_peak_memory_mb() == 5.0

    monkeypatch.setattr(benchmark_module, "mx", _MxWithMetalAndTopPeak(_AvailOnlyMetalStub()))
    assert benchmark_module._get_peak_memory_mb() == 2.0

    monkeypatch.setattr(benchmark_module, "mx", _MxWithMetalAndTopPeak(_FullMetalStub(avail=False)))
    assert benchmark_module._get_peak_memory_mb() == 2.0

    monkeypatch.setattr(benchmark_module, "mx", _MxWithMetalAndTopPeak(_ActiveOnlyMetalStub()))
    assert benchmark_module._get_peak_memory_mb() == 3.0

    monkeypatch.setattr(benchmark_module, "mx", _EmptyMxStub())
    assert benchmark_module._get_peak_memory_mb() == 0.0

    monkeypatch.setattr(benchmark_module, "mx", None)
    assert benchmark_module._get_peak_memory_mb() == 0.0


def test_run_benchmark_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _run_benchmark_pass across e2e, prefill, decode, reset errors, and None model."""
    real_mx = get_typed_mx()

    class _ResetMetalStub:
        """Metal stub supporting `is_available` and `reset_peak_memory`."""

        def __init__(self, *, fail_reset: bool = False) -> None:
            """Store `fail_reset`."""
            self.fail_reset = fail_reset

        @staticmethod
        def is_available() -> bool:
            """Return `False`."""
            return False

        def reset_peak_memory(self) -> None:
            """Raise `RuntimeError` if `fail_reset` is set.

            Raises:
                RuntimeError: If `self.fail_reset` is True.
            """
            if self.fail_reset:
                raise RuntimeError("error")

    class _BenchMxStub:
        """MLX stub delegating `zeros` to real `mlx.core`."""

        int32 = real_mx.int32

        def __init__(self, metal_obj: _ResetMetalStub | None) -> None:
            """Store `metal_obj`."""
            self.metal = metal_obj

        @staticmethod
        def zeros(shape: tuple[int, int], *, dtype: DTypeProtocol) -> MlxArrayProtocol:
            """Delegate to `real_mx.zeros`.

            Returns:
                Zero-filled MLX array.
            """
            return real_mx.zeros(shape, dtype=dtype)

        @staticmethod
        def eval(*_args: MlxArrayProtocol | str) -> None:
            """No-op eval."""

    def _model_fn(_x: benchmark_module.MlxArrayProtocol) -> str:
        """Return `'out'`."""
        return "out"

    times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] * 10
    monkeypatch.setattr(benchmark_module, "mx", _BenchMxStub(_ResetMetalStub()))
    with patch("time.perf_counter", side_effect=list(times)):
        tokens_sec, lat_ms, _ = benchmark_module._run_benchmark_pass(
            model=_model_fn,
            batch_size=1,
            num_runs=1,
            prompt_len=2,
            decode_tokens=2,
        )
        assert tokens_sec > 0
        assert lat_ms > 0

    monkeypatch.setattr(benchmark_module, "mx", _BenchMxStub(_ResetMetalStub(fail_reset=True)))
    with patch("time.perf_counter", side_effect=list(times)):
        benchmark_module._run_benchmark_pass(_model_fn, 1, 1, mode="prefill")
        benchmark_module._run_benchmark_pass(_model_fn, 1, 1, mode="decode")

    monkeypatch.setattr(benchmark_module, "mx", _BenchMxStub(None))
    with patch("time.perf_counter", side_effect=list(times)):
        benchmark_module._run_benchmark_pass(None, 1, 1)


def test_benchmark_model_behavioral(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark_model executes real MLX forward passes on a tiny mlx.nn.Module through run_benchmark_wrapper."""
    mlx_nn = get_typed_mlx_nn()

    tiny_model = mlx_nn.Sequential(mlx_nn.Embedding(32, 8), mlx_nn.Linear(8, 32))
    mx_eval_model(tiny_model)

    def _load_tiny(_m: str) -> tuple[TypedMlxSequentialProtocol, str]:
        """Return `(tiny_model, 'tok')`."""
        return (tiny_model, "tok")

    monkeypatch.setattr(benchmark_module, "load", _load_tiny)

    res = benchmark_module.benchmark_model("model", "cpu", 1, num_runs=1, prompt_len=4, decode_tokens=2)
    assert res["backend"] == "mlx"
    assert res["status"] == "success"
    assert res["tokens_per_sec"] > 0.0
    assert res["latency_ms"] > 0.0
