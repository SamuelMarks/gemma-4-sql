"""Tests for PyTorch benchmark and TorchRuntimeAdapter using typed protocol doubles without MagicMock."""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import Self
from unittest.mock import patch

import pytest
import torch

import gemma_4_sql.backends.pytorch.benchmark as benchmark_module
from gemma_4_sql.backends.pytorch.runtime_adapter import TorchRuntimeAdapter
from gemma_4_sql.exceptions import DependencyMissingError
from tests.conftest import requires_torch

pytestmark = requires_torch


class _FakeCudaNamespace:
    """Typed `torch.cuda` double tracking synchronization, memory stats, and availability."""

    def __init__(self, *, available: bool = False, max_bytes: int = 10 * 1048576) -> None:
        """Initialize CUDA stub state."""
        self.available = available
        self.max_bytes = max_bytes
        self.sync_calls = 0
        self.reset_calls = 0

    def is_available(self) -> bool:
        """Return `self.available`."""
        return self.available

    def synchronize(self) -> None:
        """Increment `sync_calls`."""
        self.sync_calls += 1

    def max_memory_allocated(self) -> int:
        """Return `self.max_bytes`."""
        return self.max_bytes

    def reset_peak_memory_stats(self) -> None:
        """Increment `reset_calls`."""
        self.reset_calls += 1


class _FakeMpsBackend:
    """Typed `torch.backends.mps` availability double."""

    def __init__(self, *, available: bool = False) -> None:
        """Store availability flag."""
        self.available = available

    def is_available(self) -> bool:
        """Return `self.available`."""
        return self.available


class _FakeBackendsNamespace:
    """Typed `torch.backends` container double."""

    def __init__(self, mps_available: bool = False) -> None:
        """Initialize `.mps` availability double."""
        self.mps = _FakeMpsBackend(available=mps_available)


class _FakeMpsRuntimeNamespace:
    """Typed `torch.mps` runtime double."""

    def __init__(self, *, driver_bytes: int = 5 * 1048576) -> None:
        """Initialize MPS runtime state."""
        self.driver_bytes = driver_bytes
        self.sync_calls = 0

    def synchronize(self) -> None:
        """Increment `sync_calls`."""
        self.sync_calls += 1

    def driver_allocated_memory(self) -> int:
        """Return `self.driver_bytes`."""
        return self.driver_bytes


class _FakeTorchRuntimeModule:
    """Typed `torch` module double satisfying `TorchRuntimeAdapter` structural protocols."""

    def __init__(
        self,
        *,
        cuda_available: bool = False,
        mps_available: bool = False,
        compile_raises: bool = False,
    ) -> None:
        """Initialize fake torch namespaces and dtypes."""
        self.cuda = _FakeCudaNamespace(available=cuda_available)
        self.backends = _FakeBackendsNamespace(mps_available=mps_available)
        self.mps = _FakeMpsRuntimeNamespace()
        self.bfloat16: str | None = "bf16"
        self.float16: str | None = "fp16"
        self.float32: str | None = "fp32"
        self.long: str | None = "int64"
        self.compile_raises = compile_raises
        self.seeds: list[int] = []

    def compile[ModelT](self, model: ModelT) -> ModelT:
        """Return `model` or raise `RuntimeError` if `compile_raises` is set.

        Raises:
            RuntimeError: If `self.compile_raises` is True.
        """
        if self.compile_raises:
            raise RuntimeError("compile error")
        return model

    def manual_seed(self, seed: int) -> None:
        """Record `seed`."""
        self.seeds.append(seed)

    @contextlib.contextmanager
    def no_grad(self) -> Generator[None, None, None]:
        """Yield inside a no-grad stub context."""
        yield

    def randint(self, low: int, high: int, size: tuple[int, ...], *, dtype: str | None = None) -> torch.Tensor:
        """Delegate synthetic integer input creation to real `torch.randint`.

        Returns:
            Random integer tensor.
        """
        _ = dtype
        return torch.randint(low, min(high, 64), size, dtype=torch.long)


class _MinimalTorchModuleWithoutHelpers:
    """Minimal `torch` double omitting `manual_seed`, `no_grad`, `cuda`, `backends`, and `mps`."""

    long: None = None

    def randint(
        self, _low: int, _high: int, _size: tuple[int, ...], *, dtype: str | None = None
    ) -> _NonMovableTensorStub:
        """Return a tensor stub whose `.to()` raises `TypeError`."""
        _ = dtype
        return _NonMovableTensorStub()


class _NonMovableTensorStub:
    """Minimal `TorchTensorProtocol`-compatible input stub whose `.to(...)` raises `TypeError`."""

    shape = (1, 32)
    ndim = 2
    dtype = None

    def to(self, _dev: str) -> Self:
        """Raise `TypeError` to exercise the fallback branch in `TorchRuntimeAdapter.randint`.

        Raises:
            TypeError: Always raised to test device fallback.
        """
        raise TypeError("unsupported device")


class _MovableBenchModel:
    """Typed benchmark model double supporting `.to()`, `.eval()`, `.generate()`, and `.config`."""

    class _Cfg:
        """Model config with custom `vocab_size`."""

        vocab_size = 128

    config = _Cfg()

    def __init__(self) -> None:
        """Initialize call counters."""
        self.moved_to: list[str] = []
        self.eval_called = False

    def __call__(self, *args: benchmark_module.ForwardArg, **kwargs: benchmark_module.ForwardArg) -> torch.Tensor:
        """Return dummy output tensor."""
        _ = (args, kwargs)
        return torch.zeros((1, 4))

    def to(self, target: str) -> Self:
        """Record `target` device/dtype.

        Returns:
            Self after recording `target`.
        """
        self.moved_to.append(str(target))
        return self

    def eval(self) -> Self:
        """Mark `eval_called`.

        Returns:
            Self after setting `eval_called`.
        """
        self.eval_called = True
        return self

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 20,
        min_new_tokens: int = 0,
    ) -> torch.Tensor:
        """Return `input_ids`."""
        _ = (max_new_tokens, min_new_tokens)
        return input_ids


class _MinimalBenchModel:
    """Minimal callable benchmark model without `.to()`, `.generate()`, or `.config`."""

    def eval(self) -> Self:
        """Perform a no-op eval call.

        Returns:
            Self unchanged.
        """
        return self

    def __call__(self, *args: benchmark_module.ForwardArg, **kwargs: benchmark_module.ForwardArg) -> torch.Tensor:
        """Return dummy output tensor."""
        _ = (args, kwargs)
        return torch.zeros((1, 4))


def test_pytorch_benchmark_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test pytorch benchmark missing deps."""
    monkeypatch.setattr(benchmark_module, "torch", None)
    monkeypatch.setattr(benchmark_module, "AutoModelForCausalLM", None)

    with pytest.raises(DependencyMissingError, match="PyTorch dependencies"):
        benchmark_module.benchmark_model("model")


def test_get_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _get_device across cpu, cuda, and mps."""
    fake_torch = _FakeTorchRuntimeModule()
    monkeypatch.setattr(benchmark_module, "torch", fake_torch)

    assert benchmark_module._get_device("cpu") == "cpu"

    fake_torch.cuda.available = True
    assert benchmark_module._get_device("gpu") == "cuda"

    fake_torch.cuda.available = False
    fake_torch.backends.mps.available = True
    assert benchmark_module._get_device("gpu") == "mps"

    fake_torch.backends.mps.available = False
    assert benchmark_module._get_device("gpu") == "cpu"


def test_load_pytorch_model_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _load_pytorch_model_and_device across native and HF backends."""
    fake_torch = _FakeTorchRuntimeModule()

    class _FakeHFAutoLoader:
        """Typed `AutoModelForCausalLM` loader double."""

        current_model: _MovableBenchModel | _MinimalBenchModel = _MovableBenchModel()

        @classmethod
        def from_pretrained(
            cls, _name: str, *, torch_dtype: str | None = None
        ) -> _MovableBenchModel | _MinimalBenchModel:
            """Return `cls.current_model`."""
            _ = torch_dtype
            return cls.current_model

    monkeypatch.setattr(benchmark_module, "torch", fake_torch)
    monkeypatch.setattr(benchmark_module, "AutoModelForCausalLM", _FakeHFAutoLoader)

    native_stub: _MovableBenchModel | _MinimalBenchModel = _MovableBenchModel()

    from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config

    def _build_native_stub(_cfg: Gemma4Config | None = None) -> _MovableBenchModel | _MinimalBenchModel:
        """Return `native_stub`."""
        _ = _cfg
        return native_stub

    with (
        patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM", side_effect=_build_native_stub),
        patch.object(Gemma4Config, "from_preset", return_value=Gemma4Config.gemma4_tiny()),
    ):
        _res, dev = benchmark_module._load_pytorch_model_and_device(
            "test_model",
            "cpu",
            dtype="bfloat16",
            backend_alias="pytorch_native",
        )
        assert dev == "cpu"

        fake_torch.bfloat16 = None
        fake_torch.float32 = None
        benchmark_module._load_pytorch_model_and_device(
            "test_model", "cpu", dtype="unknown", backend_alias="pytorch_native"
        )
        benchmark_module._load_pytorch_model_and_device(
            "test_model", "cpu", dtype="bfloat16", backend_alias="pytorch_native"
        )

        native_stub = _MinimalBenchModel()
        benchmark_module._load_pytorch_model_and_device(
            "test_model", "cpu", dtype="unknown", backend_alias="pytorch_native"
        )

    benchmark_module._load_pytorch_model_and_device("model", "cpu", dtype="bfloat16", backend_alias="pytorch")

    fake_torch.compile_raises = True
    benchmark_module._load_pytorch_model_and_device("model", "cpu", dtype="bfloat16", backend_alias="pytorch")

    _FakeHFAutoLoader.current_model = _MinimalBenchModel()
    monkeypatch.setattr(benchmark_module, "torch", _MinimalTorchModuleWithoutHelpers())
    benchmark_module._load_pytorch_model_and_device("model", "cpu", dtype="bfloat16", backend_alias="pytorch")


def test_sync_and_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _sync_cuda and _get_memory_mb."""
    fake_torch = _FakeTorchRuntimeModule()
    monkeypatch.setattr(benchmark_module, "torch", fake_torch)

    benchmark_module._sync_cuda("cuda")
    assert fake_torch.cuda.sync_calls == 1

    benchmark_module._sync_cuda("mps")
    assert fake_torch.mps.sync_calls == 1

    benchmark_module._sync_cuda("cpu")
    assert benchmark_module._get_memory_mb(None, "cuda") == 10.0
    assert benchmark_module._get_memory_mb(None, "mps") == 5.0
    assert benchmark_module._get_memory_mb(None, "cpu") > 0.0


def test_run_benchmark_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _run_benchmark_pass across prefill, decode, and minimal runtime adapter fallbacks."""
    fake_torch = _FakeTorchRuntimeModule()
    monkeypatch.setattr(benchmark_module, "torch", fake_torch)

    model = _MovableBenchModel()
    times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] * 5

    with patch("time.time", side_effect=list(times)):
        tps, _lat, _mem = benchmark_module._run_benchmark_pass(model, "cuda", 1, 1, 1, "prefill", 10)
        assert tps > 0
        assert fake_torch.cuda.reset_calls == 1

    with patch("time.time", side_effect=list(times)):
        tps2, _lat2, _mem2 = benchmark_module._run_benchmark_pass(model, "cpu", 1, 1, 1, "decode", 10)
        assert tps2 > 0

    monkeypatch.setattr(benchmark_module, "torch", _MinimalTorchModuleWithoutHelpers())
    with patch("time.time", side_effect=list(times)):
        benchmark_module._run_benchmark_pass(_MinimalBenchModel(), "cpu", 1, 1, 1, "prefill", 10)


def test_benchmark_model_behavioral(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark_model executes through run_benchmark_wrapper with real stats."""
    fake_torch = _FakeTorchRuntimeModule()
    monkeypatch.setattr(benchmark_module, "torch", fake_torch)

    class _DummyLoader:
        """Minimal HF loader stub."""

        @staticmethod
        def from_pretrained(_n: str, *, torch_dtype: str | None = None) -> _MinimalBenchModel:
            """Return `_MinimalBenchModel`."""
            _ = torch_dtype
            return _MinimalBenchModel()

    assert _DummyLoader.from_pretrained("x") is not None
    monkeypatch.setattr(benchmark_module, "AutoModelForCausalLM", _DummyLoader)

    with (
        patch.object(benchmark_module, "_load_pytorch_model_and_device", return_value=(_MinimalBenchModel(), "cpu")),
        patch.object(benchmark_module, "_run_benchmark_pass", return_value=(250.0, 4.0, 128.0)),
    ):
        res = benchmark_module.benchmark_model("model", "cpu", 1)
        assert res["backend"] == "pytorch"
        assert res["status"] == "success"
        assert res["tokens_per_sec"] == 250.0
        assert res["latency_ms"] == 4.0

    with patch("time.time", side_effect=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
        benchmark_module._run_benchmark_pass(_MinimalBenchModel(), "cpu", 1, 1, 1, "decode", 10)


def test_real_pytorch_tiny_gemma4_benchmark_cpu() -> None:
    """Verify _run_benchmark_pass executes on a real tiny Gemma4ForCausalLM in prefill and decode modes."""
    from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
    from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM

    cfg = Gemma4Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=1,
    )
    model = Gemma4ForCausalLM(cfg)
    tps, lat, mem = benchmark_module._run_benchmark_pass(model, "cpu", 1, 1, 1, "prefill", 2)
    assert tps > 0.0
    assert lat > 0.0
    assert mem > 0.0

    tps_dec, lat_dec, mem_dec = benchmark_module._run_benchmark_pass(model, "cpu", 1, 1, 1, "decode", 2)
    assert tps_dec > 0.0
    assert lat_dec > 0.0
    assert mem_dec > 0.0


def test_torch_runtime_adapter_edge_cases() -> None:
    """Verify TorchRuntimeAdapter handles None torch, invalid vocab_size, and float16/float32 dtypes."""
    empty_adapter = TorchRuntimeAdapter(None)
    assert empty_adapter.get_device("gpu") == "cpu"
    assert empty_adapter.resolve_dtype("bfloat16") is None
    assert empty_adapter.long_dtype() is None
    assert empty_adapter.compile_model("m") == "m"
    empty_adapter.synchronize("cuda")
    assert empty_adapter.max_memory_mb("cuda", 32.0) == 32.0
    empty_adapter.reset_peak_memory_stats("cuda")
    empty_adapter.manual_seed(0)
    with empty_adapter.no_grad():
        pass

    class _DtypeAndNonCallableCompileStub:
        """Typed torch stub with float16/float32 dtypes and non-callable compile attribute."""

        float16 = "fp16"
        float32 = "fp32"
        compile = None

    adapter = TorchRuntimeAdapter(_DtypeAndNonCallableCompileStub())
    assert adapter.resolve_dtype("float16") == "fp16"
    assert adapter.resolve_dtype("float32") == "fp32"
    assert adapter.compile_model("m") == "m"

    class _BadVocabModel:
        """Model with non-positive vocab_size to exercise fallback 256000."""

        class _Cfg:
            """Config with vocab_size = 0."""

            vocab_size = 0

        config = _Cfg()

        def __call__(self, *args: benchmark_module.ForwardArg, **kwargs: benchmark_module.ForwardArg) -> float:
            """Return 0.0."""
            _ = (args, kwargs)
            return 0.0

    assert _BadVocabModel()() == 0.0
    assert benchmark_module._resolve_vocab_bound(_BadVocabModel()) == 256000

    class _NoneCfgModel:
        """Model with `config = None` and `eval = None`."""

        config = None
        eval = None

        def __call__(self, *args: benchmark_module.ForwardArg, **kwargs: benchmark_module.ForwardArg) -> float:
            """Return 0.0."""
            _ = (args, kwargs)
            return 0.0

    none_cfg_m = _NoneCfgModel()
    assert none_cfg_m() == 0.0
    assert benchmark_module._resolve_vocab_bound(none_cfg_m) == 256000
    benchmark_module._move_and_eval(none_cfg_m, "cpu")

    class _NullNamespacesTorchStub:
        """Torch stub with `cuda = None`, `backends = None`, and `mps = None`."""

        cuda = None
        backends = None
        mps = None

    null_ns_adapter = TorchRuntimeAdapter(_NullNamespacesTorchStub())
    assert null_ns_adapter.get_device("gpu") == "cpu"
    null_ns_adapter.synchronize("cuda")
    null_ns_adapter.synchronize("mps")
    assert null_ns_adapter.max_memory_mb("cuda", 16.0) == 16.0
    assert null_ns_adapter.max_memory_mb("mps", 16.0) == 16.0
    null_ns_adapter.reset_peak_memory_stats("cuda")
    null_ns_adapter.reset_peak_memory_stats("cpu")

    class _NullInnerMpsBackends:
        """Backends stub with `mps = None`."""

        mps = None

    class _NullInnerMpsTorch:
        """Torch stub with `backends.mps = None`."""

        backends = _NullInnerMpsBackends()

    assert TorchRuntimeAdapter(_NullInnerMpsTorch()).get_device("gpu") == "cpu"
    assert TorchRuntimeAdapter(_MinimalTorchModuleWithoutHelpers()).get_device("gpu") == "cpu"
