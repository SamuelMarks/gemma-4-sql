"""Unit and behavioral tests for PyTorch/XLA (`torch_xla` PJRT) TPU parity across all submodules."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self
from unittest.mock import patch

import pytest
import torch
from torch import nn

import gemma_4_sql.backends.pytorch as pt_pkg
import gemma_4_sql.backends.pytorch.benchmark as pt_bench
import gemma_4_sql.backends.pytorch.distributed as pt_dist
import gemma_4_sql.backends.pytorch.dpo as pt_dpo
import gemma_4_sql.backends.pytorch.export as pt_export
import gemma_4_sql.backends.pytorch.inference as pt_infer
import gemma_4_sql.backends.pytorch.peft as pt_peft
import gemma_4_sql.backends.pytorch.quantize as pt_quant
import gemma_4_sql.backends.pytorch.serve as pt_serve
import gemma_4_sql.backends.pytorch.train as pt_train
from gemma_4_sql.backends.common_serve import DirectRunnableAppProtocol
from gemma_4_sql.backends.pytorch.adapter import PyTorchBackend
from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM
from gemma_4_sql.backends.pytorch.runtime_adapter import TorchRuntimeAdapter
from gemma_4_sql.exceptions import HardwareUnavailableError
from gemma_4_sql.type_hints import (
    DPOConfig,
    ETLConfig,
    ForwardArg,
    ForwardOutput,
    TorchTensorProtocol,
    TrainingConfig,
)
from tests.conftest import requires_torch
from tests.utils.tpu_doubles import (
    FakeTorchXLACoreModel,
    FakeTorchXLAModule,
    FakeTorchXLARuntime,
    FakeXLADeviceDescriptor,
)

pytestmark = requires_torch


class _TrackingOptimizer:
    """Typed `TorchOptimizerProtocol` double counting `step` and `zero_grad` calls."""

    def __init__(self) -> None:
        """Initialize counters."""
        self.steps = 0
        self.zero_grads = 0

    def step(self) -> None:
        """Increment `steps`."""
        self.steps += 1

    def zero_grad(self) -> None:
        """Increment `zero_grads`."""
        self.zero_grads += 1


class _FakeCudaNamespace:
    """Typed `torch.cuda` namespace double."""

    def __init__(self, *, available: bool, count: int = 4) -> None:
        """Configure availability and GPU count.

        Args:
            available: Whether CUDA is available.
            count: Reported GPU device count.

        """
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        """Return `self._available`.

        Returns:
            Availability boolean.

        """
        return self._available

    def device_count(self) -> int:
        """Return `self._count`.

        Returns:
            Simulated GPU count.

        """
        return self._count


class _FakeTorchForAdapter:
    """Typed `torch` module double for `TorchRuntimeAdapter` compile and CUDA tests."""

    def __init__(self, *, cuda_available: bool = False, reject_backend_kwarg: bool = False) -> None:
        """Initialize CUDA namespace and `compile` tracking.

        Args:
            cuda_available: Whether CUDA is available.
            reject_backend_kwarg: If `True`, `compile(..., backend=...)` raises `TypeError`.

        """
        self.cuda = _FakeCudaNamespace(available=cuda_available, count=8)
        self.reject_backend_kwarg = reject_backend_kwarg
        self.compiled_backends: list[str | None] = []

    def compile[ModelT](self, model: ModelT, *, backend: str | None = None) -> ModelT:
        """Record `backend` or raise `TypeError` when `reject_backend_kwarg` is enabled.

        Args:
            model: Model to compile.
            backend: Optional compiler backend name.

        Returns:
            Unchanged `model`.

        Raises:
            TypeError: When `self.reject_backend_kwarg` is `True` and `backend` is provided.

        """
        if self.reject_backend_kwarg and backend is not None:
            raise TypeError("backend kwarg unsupported")
        self.compiled_backends.append(backend)
        return model


def test_torch_runtime_adapter_xla_device_sync_memory_and_openxla_compile() -> None:
    """Verify `TorchRuntimeAdapter` XLA device resolution, strict hardware errors, sync, HBM, and OpenXLA compile."""
    assert pt_pkg.get_trainer() == "pytorch_trainer"
    hf_backend = PyTorchBackend.from_alias("hf_transformers")
    assert hf_backend.backend_alias == "hf_transformers"
    assert hf_backend.get_trainer() == "pytorch_trainer"

    xla_mod = FakeTorchXLAModule(device_names=("xla:0", "xla:1", "xla:2", "xla:3"))
    xm_mod = FakeTorchXLACoreModel(tpu_devices=("xla:0", "xla:1", "xla:2", "xla:3"), peak_bytes=768 * 1024 * 1024)
    fake_torch = _FakeTorchForAdapter(cuda_available=False)

    adapter = TorchRuntimeAdapter(fake_torch, xla_module=xla_mod, xm_module=xm_mod)
    assert str(FakeXLADeviceDescriptor("xla:0")) == "xla:0"
    assert adapter.get_device("cpu") == "cpu"
    assert adapter.get_device("tpu-v6e", strict_hardware=True) == "xla:0"
    assert adapter.get_device_count("tpu-v6e") == 4
    assert xm_mod.get_xla_supported_devices("CPU") == ()
    assert adapter.get_device_count("cpu") == 1
    assert adapter.get_device_count("gpu") == 1

    # Sync via torch_xla.sync() vs xm.mark_step() + xm.wait_device_ops() vs no XLA modules
    adapter.synchronize("xla:0")
    assert xla_mod.sync_calls == 1
    xm_only_adapter = TorchRuntimeAdapter(fake_torch, xla_module=None, xm_module=xm_mod)
    with patch.dict(sys.modules, {"torch_xla": None}):
        assert xm_only_adapter.get_device("tpu") == "xla:0"
        xm_only_adapter.synchronize("xla:0")
    assert xm_mod.mark_step_calls == 1
    assert xm_mod.wait_ops_calls == 1

    empty_xm = FakeTorchXLACoreModel(tpu_devices=(), peak_bytes=0)
    assert empty_xm.xla_device() == "xla:0"
    empty_xla = FakeTorchXLAModule(device_names=())
    assert empty_xla.device() == "xla:0"

    # Fallback from empty xm devices to xla_mod.devices(), plus None xm / None xla branches
    xla_fallback_adapter = TorchRuntimeAdapter(fake_torch, xla_module=xla_mod, xm_module=empty_xm)
    assert xla_fallback_adapter.get_device("tpu-v5p") == "xla:0"
    assert xla_fallback_adapter.get_device_count("tpu-v5p") == 4

    with patch.dict(sys.modules, {"torch_xla": None, "torch_xla.core.xla_model": None}):
        none_both = TorchRuntimeAdapter(fake_torch, xla_module=None, xm_module=None)
        assert none_both.get_device("tpu") == "cpu"
        assert none_both.get_device_count("tpu") == 1
        none_both.synchronize("xla:0")
        assert none_both.max_memory_mb("xla:0", 64.0) == 64.0

    # Both empty: non-strict returns cpu / 1, strict raises HardwareUnavailableError
    no_tpu_adapter = TorchRuntimeAdapter(fake_torch, xla_module=empty_xla, xm_module=empty_xm)
    assert no_tpu_adapter.get_device("tpu-v4-8", strict_hardware=False) == "cpu"
    assert no_tpu_adapter.get_device_count("tpu-v4-8") == 1
    with pytest.raises(HardwareUnavailableError, match="TPU device unavailable"):
        no_tpu_adapter.get_device("tpu-v4-8", strict_hardware=True)

    # Strict GPU unavailable when torch is None vs when CUDA/MPS are absent vs CUDA count
    with pytest.raises(HardwareUnavailableError, match="PyTorch GPU runtime unavailable"):
        TorchRuntimeAdapter(None).get_device("gpu", strict_hardware=True)

    with pytest.raises(HardwareUnavailableError, match="CUDA GPU device unavailable"):
        adapter.get_device("gpu", strict_hardware=True)

    cuda_torch = _FakeTorchForAdapter(cuda_available=True)
    cuda_adapter = TorchRuntimeAdapter(cuda_torch)
    assert cuda_adapter.get_device_count("gpu") == 8

    # Memory stats on XLA: positive peak_bytes vs 0 fallback
    assert adapter.get_memory_mb("xla:0") == pytest.approx(768.0)
    assert xla_fallback_adapter.max_memory_mb("xla:0", 42.0) == pytest.approx(42.0)

    # OpenXLA compile and TypeError fallback
    assert adapter.maybe_compile("sentinel", device="xla:0") == "sentinel"
    assert fake_torch.compiled_backends == ["openxla"]
    legacy_compile_torch = _FakeTorchForAdapter(reject_backend_kwarg=True)
    legacy_adapter = TorchRuntimeAdapter(legacy_compile_torch)
    assert legacy_adapter.compile_model("sentinel", device="xla:0") == "sentinel"
    assert legacy_compile_torch.compiled_backends == [None]


def test_pytorch_distributed_xla_pjrt_and_optimizer_step() -> None:
    """Verify `setup_distributed` for `'xla'`/`'tpu'`/`'pjrt'` and `step_optimizer_xla_aware`."""
    xla_mod = FakeTorchXLAModule()
    xm_mod = FakeTorchXLACoreModel()
    xla_rt = FakeTorchXLARuntime(ordinal=2, world=4, local_ord=1)
    assert xla_rt.local_ordinal() == 1

    class _FakeDistMod:
        """Typed `TorchDistModuleProtocol` double for XLA process group tests."""

        def __init__(self, *, initialized: bool = False, fail_init: bool = False) -> None:
            """Configure initial state and failure mode."""
            self._init = initialized
            self._fail = fail_init
            self.backends: list[str] = []

        def is_available(self) -> bool:
            """Return `True`."""
            return True

        def is_initialized(self) -> bool:
            """Return `self._init`."""
            return self._init

        def init_process_group(self, backend: str) -> None:
            """Record `backend` or raise `RuntimeError`.

            Raises:
                RuntimeError: When `self._fail` is `True`.

            """
            if self._fail:
                raise RuntimeError("pjrt init failed")
            self.backends.append(backend)
            self._init = True

        def get_rank(self) -> int:
            """Return rank `2`."""
            return 2

        def get_world_size(self) -> int:
            """Return world size `4`."""
            return 4

        def destroy_process_group(self) -> None:
            """Reset `self._init`."""
            self._init = False

    dist_ok = _FakeDistMod()
    with patch.dict(
        sys.modules,
        {
            "torch_xla": xla_mod,
            "torch_xla.core.xla_model": xm_mod,
            "torch_xla.runtime": xla_rt,
            "torch_xla.distributed.xla_backend": xla_mod,
            "torch.distributed": dist_ok,
        },
    ):
        is_dist, out_dist, dev, ord_id = pt_dist.setup_distributed("xla", torch, hardware="tpu-v6e")
        assert is_dist is True
        assert out_dist is dist_ok
        assert dev == "xla:0"
        assert ord_id == 2
        assert dist_ok.backends == ["xla"]
        assert dist_ok.is_available() is True
        assert dist_ok.get_rank() == 2
        assert dist_ok.get_world_size() == 4

        # Second call with dist_ok ALREADY initialized covers the False branch of `not dist_mod_xla.is_initialized()`
        is_dist_already, out_dist_already, _, _ = pt_dist.setup_distributed("tpu", torch)
        assert is_dist_already is True
        assert out_dist_already is dist_ok
        dist_ok.destroy_process_group()

        # Failing init_process_group + no runtime module branch
        dist_fail = _FakeDistMod(fail_init=True)
        with patch.dict(sys.modules, {"torch.distributed": dist_fail, "torch_xla.runtime": None}):
            is_dist2, out_dist2, dev2, ord2 = pt_dist.setup_distributed("pjrt", torch)
            assert is_dist2 is False
            assert out_dist2 is None
            assert dev2 == "xla:0"
            assert ord2 == 0

        # Non-distributed strategy="none" with hardware="tpu-v6e" resolves XLA candidate
        is_d_none, d_mod_none, dev_none, id_none = pt_dist.setup_distributed("none", torch, hardware="tpu-v6e")
        assert (is_d_none, d_mod_none, dev_none, id_none) == (False, None, "xla:0", 0)

        # step_optimizer_xla_aware with xm present vs xm absent
        opt = _TrackingOptimizer()
        opt.zero_grad()
        assert opt.zero_grads == 1
        pt_dist.step_optimizer_xla_aware(opt, "xla:0")
        assert xm_mod.optimizer_steps == 1
        assert opt.steps == 1
        assert xla_mod.sync_calls >= 1

        with patch.dict(sys.modules, {"torch_xla.core.xla_model": None}):
            pt_dist.step_optimizer_xla_aware(opt, "xla:0")
            assert opt.steps == 2


def test_pytorch_native_xla_benchmark_without_transformers_and_warmup_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `pytorch_native` benchmarks on XLA without `transformers` and compiles `max_new_tokens` on warmup."""
    xla_mod = FakeTorchXLAModule()
    xm_mod = FakeTorchXLACoreModel(peak_bytes=512 * 1024 * 1024)
    recorded_gen_lengths: list[int] = []

    class _XLABenchModel:
        """Typed model recording `max_new_tokens` across warmup and timed steps."""

        def __init__(self) -> None:
            """Initialize device log."""
            self.devices: list[str] = []

        def to(self, target: pt_bench.DTypeProtocol | str, /) -> Self:
            """Record `target` placement and return `self`.

            Returns:
                `self`.

            """
            self.devices.append(str(target))
            return self

        def eval(self) -> Self:
            """Return `self`."""
            return self

        def __call__(self, inputs: TorchTensorProtocol, /) -> TorchTensorProtocol:
            """Return input tensor as forward output."""
            return inputs

        def generate(
            self,
            inputs: TorchTensorProtocol,
            *,
            max_new_tokens: int = 16,
            min_new_tokens: int = 16,
        ) -> TorchTensorProtocol:
            """Record `max_new_tokens` (asserting parity with `min_new_tokens`) and return `inputs`.

            Returns:
                Unchanged `inputs`.

            """
            assert max_new_tokens == min_new_tokens
            recorded_gen_lengths.append(max_new_tokens)
            return inputs

    def _noop_compile[ModelT](model: ModelT, *, backend: str | None = None) -> ModelT:
        """Return `model` unchanged.

        Returns:
            Unchanged `model`.

        """
        _ = backend
        return model

    def _new_bench_model(_cfg: Gemma4Config) -> _XLABenchModel:
        """Construct `_XLABenchModel`.

        Returns:
            `_XLABenchModel` instance.

        """
        return _XLABenchModel()

    assert _XLABenchModel()(torch.ones((1, 2), dtype=torch.long)).shape == (1, 2)
    monkeypatch.setattr(pt_bench, "AutoModelForCausalLM", None)
    monkeypatch.setattr(torch, "compile", _noop_compile)

    with (
        patch.dict(sys.modules, {"torch_xla": xla_mod, "torch_xla.core.xla_model": xm_mod}),
        patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM", _new_bench_model),
    ):
        res = pt_bench.benchmark_model(
            "gemma-4-tiny",
            hardware="tpu-v6e",
            batch_size=2,
            backend_alias="pytorch_native",
            mode="decode",
            max_new_tokens=14,
            warmup_steps=2,
            num_runs=2,
            strict_hardware=True,
        )
    assert res["status"] == "success"
    assert res["backend"] == "pytorch_native"
    assert res.get("resolved_device") == "xla:0"
    assert res.get("device_count") == 4
    assert res.get("peak_hbm_mb") == pytest.approx(512.0)
    assert res.get("compile_latency_ms") is not None
    assert res.get("rss_mb", 0.0) > 0.0
    assert recorded_gen_lengths == [14, 14, 14, 14]
    assert xla_mod.sync_calls >= 4


def test_pytorch_train_and_dpo_xla_tpu_and_bfloat16(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `pytorch/train.py` and `pytorch/dpo.py` support XLA TPU placement, dtypes, and XLA optimizer steps."""
    xla_mod = FakeTorchXLAModule()
    xm_mod = FakeTorchXLACoreModel()

    tiny_cfg = Gemma4Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_experts=1,
    )

    placed_devices: list[str] = []

    def _xla_safe_to(self: nn.Module, target: str) -> nn.Module:
        """Record `'xla:*'` target without moving storage to unregistered XLA.

        Returns:
            `self`.

        """
        placed_devices.append(target)
        return self

    created_models: list[Gemma4ForCausalLM] = []

    def _load_xla_tiny(_name: str, config: Gemma4Config | None = None) -> Gemma4ForCausalLM:
        """Instantiate `Gemma4ForCausalLM`.

        Returns:
            Tiny `Gemma4ForCausalLM` instance.

        """
        m = Gemma4ForCausalLM(config or tiny_cfg)
        created_models.append(m)
        return m

    class _MovableTensorWrapper:
        """Tensor wrapper recording `.to('xla:0')` for DPO preference batches."""

        def __init__(self, inner: torch.Tensor) -> None:
            """Store underlying CPU tensor."""
            self.inner = inner
            self.moved_to: list[str] = []

        def to(self, device: str, /) -> torch.Tensor:
            """Record `device` and return `self.inner`.

            Returns:
                Underlying CPU `torch.Tensor`.

            """
            self.moved_to.append(device)
            return self.inner

    def _train_loader(_cfg: ETLConfig) -> dict[str, list[dict[str, torch.Tensor]]]:
        """Return single synthetic SFT batch.

        Returns:
            Loader mapping.

        """
        return {
            "loader": [
                {
                    "inputs": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                    "targets": torch.tensor([[2, 3, 4, 5]], dtype=torch.long),
                }
            ]
        }

    monkeypatch.setattr(nn.Module, "to", _xla_safe_to)
    monkeypatch.setattr(pt_train, "build_dataloader", _train_loader)
    with (
        patch.dict(sys.modules, {"torch_xla": xla_mod, "torch_xla.core.xla_model": xm_mod}),
        patch.object(Gemma4ForCausalLM, "from_pretrained", staticmethod(_load_xla_tiny)),
    ):
        for dt_name, expected_dt in (
            ("bfloat16", torch.bfloat16),
            ("float16", torch.float16),
            ("float32", torch.float32),
            ("auto", torch.float32),
        ):
            train_res = pt_train.train_model(
                TrainingConfig(
                    action="sft",
                    model_name="gemma-4-tiny",
                    dataset="synthetic",
                    epochs=1,
                    batch_size=1,
                    backend="pytorch_native",
                ),
                backend_alias="pytorch_native",
                model_config=tiny_cfg,
                dtype=dt_name,
                hardware="tpu-v6e",
                strict_hardware=True,
            )
            assert train_res["status"] == "completed"
            assert next(created_models[-1].parameters()).dtype == expected_dt
        assert placed_devices[0] == "xla:0"
        assert xm_mod.optimizer_steps >= 3

        # DPO on XLA with movable policy/ref models and movable batch tensors
        movable_input = _MovableTensorWrapper(torch.tensor([[1, 2, 3]], dtype=torch.long))

        def _dpo_loader(_cfg: ETLConfig) -> dict[str, list[dict[str, _MovableTensorWrapper]]]:
            """Return single-batch preference loader containing `_MovableTensorWrapper` tensors.

            Returns:
                Loader mapping.

            """
            return {
                "loader": [
                    {
                        "chosen_inputs": movable_input,
                        "chosen_labels": _MovableTensorWrapper(torch.tensor([[2, 3, 4]], dtype=torch.long)),
                        "rejected_inputs": _MovableTensorWrapper(torch.tensor([[1, 5, 6]], dtype=torch.long)),
                        "rejected_labels": _MovableTensorWrapper(torch.tensor([[5, 6, 7]], dtype=torch.long)),
                    }
                ]
            }

        monkeypatch.setattr(pt_dpo, "build_dataloader", _dpo_loader)
        dpo_res = pt_dpo.run_dpo(
            DPOConfig(
                model_name="gemma-4-tiny",
                dataset="custom_dpo",
                epochs=1,
                batch_size=1,
            ),
            backend_alias="pytorch_native",
            hardware="tpu-v6e",
            strict_hardware=True,
            model_config=tiny_cfg,
        )
        assert dpo_res["status"] == "completed"
        assert movable_input.moved_to == ["xla:0"]

        # Non-movable policy/ref callable models when device == "xla:0"
        param = nn.Parameter(torch.ones((16, 8)))

        class _NonMovableDPOModel:
            """Callable DPO model without `.to(...)` to cover non-movable branch on XLA."""

            def parameters(self) -> Iterator[nn.Parameter]:
                """Yield single trainable parameter.

                Returns:
                    Iterator over trainable parameters.

                """
                return iter([param])

            def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
                """Compute differentiable vocabulary logits.

                Returns:
                    Logits tensor.

                """
                emb = param[inputs % 16]
                return emb @ param.T

        def _load_non_movable(
            _model_name: str,
            *,
            backend_alias: str = "pytorch",
            model_config: Gemma4Config | None = None,
        ) -> tuple[_NonMovableDPOModel, _NonMovableDPOModel]:
            """Return pair of `_NonMovableDPOModel`.

            Returns:
                Policy and reference model pair.

            """
            _ = (backend_alias, model_config)
            return (_NonMovableDPOModel(), _NonMovableDPOModel())

        monkeypatch.setattr(pt_dpo, "_load_dpo_models", _load_non_movable)
        dpo_res2 = pt_dpo.run_dpo(
            DPOConfig(model_name="m", dataset="custom_dpo", epochs=1, batch_size=1),
            backend_alias="hf_transformers",
            hardware="tpu",
        )
        assert dpo_res2["status"] == "completed"


def test_pytorch_inference_hf_and_native_on_xla_without_device_map_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `hf_transformers` and `pytorch_native` inference place models on `xla:0` without `device_map='auto'`."""
    xla_mod = FakeTorchXLAModule()
    xm_mod = FakeTorchXLACoreModel()
    pt_infer._MODEL_CACHE.clear()
    pt_infer._HF_MODEL_CACHE.clear()

    hf_load_calls: list[tuple[str, Mapping[str, str | int | bool | None]]] = []

    def _empty_strs() -> list[str]:
        """Return empty string list.

        Returns:
            Empty list of strings.

        """
        return []

    @dataclass
    class _FakeGenOut:
        """HF generate output double."""

        sequences: torch.Tensor
        scores: tuple[torch.Tensor, ...] = ()

    @dataclass
    class _FakeHFModel:
        """Movable HF causal LM double recording `.to(...)`."""

        moved: list[str] = field(default_factory=_empty_strs)

        def to(self, device: str, /) -> Self:
            """Record `device` and return `self`.

            Returns:
                `self`.

            """
            self.moved.append(device)
            return self

        def generate(self, **kwargs: TorchTensorProtocol | int | bool) -> _FakeGenOut:
            """Return deterministic token sequence."""
            _ = kwargs
            return _FakeGenOut(sequences=torch.tensor([[1, 2, 10, 11]], dtype=torch.long))

    class _NonMovableHFModel:
        """HF causal LM double without `.to(...)` to cover non-movable branch on XLA."""

        def generate(self, **kwargs: TorchTensorProtocol | int | bool) -> _FakeGenOut:
            """Return deterministic token sequence."""
            _ = kwargs
            return _FakeGenOut(sequences=torch.tensor([[1, 2, 10, 11]], dtype=torch.long))

    class _FakeBatchEncoding(dict[str, torch.Tensor]):
        """BatchEncoding double supporting `.to(device)`."""

        def __init__(self) -> None:
            """Populate `input_ids`."""
            super().__init__(input_ids=torch.tensor([[1, 2]], dtype=torch.long))
            self.moved: list[str] = []

        @property
        def input_ids(self) -> torch.Tensor:
            """Encoded `input_ids` tensor."""
            return self["input_ids"]

        def to(self, device: str, /) -> Self:
            """Record `device` and return `self`.

            Returns:
                `self`.

            """
            self.moved.append(device)
            return self

    class _FakeHFTokenizer:
        """HF tokenizer double."""

        @staticmethod
        def from_pretrained(_name: str) -> _FakeHFTokenizer:
            """Return `_FakeHFTokenizer`."""
            return _FakeHFTokenizer()

        def __call__(self, _prompt: str, *, return_tensors: str = "pt") -> _FakeBatchEncoding:
            """Return `_FakeBatchEncoding`."""
            _ = return_tensors
            return _FakeBatchEncoding()

        def decode(self, _ids: Sequence[int] | torch.Tensor, *, skip_special_tokens: bool = True) -> str:
            """Return valid SQL query."""
            _ = skip_special_tokens
            return "SELECT 1;"

    use_movable_hf = True

    class _FakeAutoModel:
        """`AutoModelForCausalLM` double asserting `device_map` is never passed on XLA."""

        @staticmethod
        def from_pretrained(
            name: str,
            **kwargs: str | int | bool | None,
        ) -> _FakeHFModel | _NonMovableHFModel:
            """Record `kwargs` and return movable or non-movable HF model.

            Returns:
                Movable or non-movable HF model double.

            """
            hf_load_calls.append((name, dict(kwargs)))
            return _FakeHFModel() if use_movable_hf else _NonMovableHFModel()

    class _FakePeftMod:
        """Module exposing `PeftModel.from_pretrained` for XLA adapter placement."""

        class PeftModel:
            """Typed `PeftModel` loader double."""

            @staticmethod
            def from_pretrained(model: _FakeHFModel | _NonMovableHFModel, _adapter: str) -> _FakeHFModel:
                """Return movable `_FakeHFModel`."""
                return model if isinstance(model, _FakeHFModel) else _FakeHFModel()

    monkeypatch.setattr(pt_infer, "AutoTokenizer", _FakeHFTokenizer)
    monkeypatch.setattr(pt_infer, "AutoModelForCausalLM", _FakeAutoModel)

    with patch.dict(
        sys.modules,
        {"torch_xla": xla_mod, "torch_xla.core.xla_model": xm_mod, "peft": _FakePeftMod},
    ):
        res_hf = pt_infer.generate_sql(
            "google/gemma-4-E4B",
            "count rows",
            backend_alias="hf_transformers",
            hardware="tpu-v6e",
            strict_hardware=True,
            adapter_path="/tmp/lora_xla",
        )
        assert res_hf["status"] == "success"
        assert res_hf["backend"] == "hf_transformers"
        assert res_hf["sql"] == "SELECT 1;"
        assert "device_map" not in hf_load_calls[0][1]
        assert xla_mod.sync_calls >= 1

        # Non-movable HF model branch on XLA without adapter
        use_movable_hf = False
        pt_infer._HF_MODEL_CACHE.clear()
        res_hf2 = pt_infer.generate_sql(
            "google/gemma-4-E4B-v2",
            "count rows",
            backend_alias="hf_transformers",
            hardware="tpu-v5p",
        )
        assert res_hf2["status"] == "success"

        # Native PyTorch on XLA (both movable and non-movable CausalLMProtocol branches)
        class _MovableNativeProxy:
            """Movable `CausalLMProtocol` proxy intercepting `.to('xla:0')`."""

            def __init__(self) -> None:
                """Initialize device log."""
                self.moved: list[str] = []

            def to(self, dev: str, /) -> Self:
                """Record `dev` and return `self`.

                Returns:
                    `self`.

                """
                self.moved.append(dev)
                return self

            def generate(
                self,
                input_ids: TorchTensorProtocol,
                **kwargs: TorchTensorProtocol | int | bool | None,
            ) -> TorchTensorProtocol:
                """Append token IDs `[1, 2]` to `input_ids`.

                Returns:
                    Extended sequence tensor.

                """
                _ = kwargs
                assert isinstance(input_ids, torch.Tensor)
                return torch.cat([input_ids, torch.tensor([[1, 2]], dtype=torch.long)], dim=-1)

            def __call__(self, *args: ForwardArg, **kwargs: ForwardArg) -> ForwardOutput:
                """Return deterministic vocabulary logits `(1, seq_len, 64)`."""
                _ = kwargs
                seq_t = args[0]
                assert isinstance(seq_t, torch.Tensor)
                return torch.zeros((1, int(seq_t.shape[-1]), 64), dtype=torch.float32)

        class _NonMovableNativeProxy:
            """Non-movable `_NativeGenerateModelProtocol` proxy without `.to(...)`."""

            def generate(
                self,
                input_ids: TorchTensorProtocol,
                **kwargs: TorchTensorProtocol | int | bool | None,
            ) -> TorchTensorProtocol:
                """Append token IDs `[1, 2]` to `input_ids`.

                Returns:
                    Extended sequence tensor.

                """
                _ = kwargs
                assert isinstance(input_ids, torch.Tensor)
                return torch.cat([input_ids, torch.tensor([[1, 2]], dtype=torch.long)], dim=-1)

            def __call__(self, *args: ForwardArg, **kwargs: ForwardArg) -> ForwardOutput:
                """Return deterministic vocabulary logits `(1, seq_len, 64)`."""
                _ = kwargs
                seq_t = args[0]
                assert isinstance(seq_t, torch.Tensor)
                return torch.zeros((1, int(seq_t.shape[-1]), 64), dtype=torch.float32)

        proxy = _MovableNativeProxy()
        pt_infer._MODEL_CACHE.put_model("xla_native_m", (proxy, None))
        res_nat = pt_infer.generate_sql(
            "xla_native_m",
            "SELECT",
            backend_alias="pytorch_native",
            hardware="tpu-v6e",
        )
        assert res_nat["status"] == "success"
        assert proxy.moved == ["xla:0"]

        pt_infer._MODEL_CACHE.put_model("xla_native_nomove", (_NonMovableNativeProxy(), None))
        res_nat2 = pt_infer.generate_sql(
            "xla_native_nomove",
            "SELECT",
            backend_alias="pytorch_native",
            hardware="tpu-v6e",
        )
        assert res_nat2["status"] == "success"
    pt_infer._MODEL_CACHE.clear()
    pt_infer._HF_MODEL_CACHE.clear()


def test_pytorch_peft_serve_export_and_quantize_xla_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify PEFT, Serve, Export, and Quantize handle `hf_transformers`, `pytorch_native`, and XLA->CPU staging."""
    xla_mod = FakeTorchXLAModule()
    xm_mod = FakeTorchXLACoreModel()

    # 1. Export & Quantize XLA -> CPU staging via minimal protocol double
    class _XLADeviceTensorDouble:
        """Minimal tensor double reporting `device == 'xla:0'` and returning CPU tensor on `.to('cpu')`."""

        def __init__(self) -> None:
            """Initialize `staged` targets list."""
            self.staged: list[str] = []

        @property
        def device(self) -> str:
            """Report `'xla:0'`."""
            return "xla:0"

        def to(self, target: str, /) -> torch.Tensor:
            """Record `target` and return CPU `torch.Tensor`.

            Returns:
                CPU `torch.Tensor`.

            """
            self.staged.append(target)
            return torch.ones((16, 16), dtype=torch.float32)

    exp_xla = _XLADeviceTensorDouble()
    staged_export = pt_export._stage_xla_tensor_to_cpu(exp_xla)
    assert isinstance(staged_export, torch.Tensor)
    assert exp_xla.staged == ["cpu"]

    quant_xla = _XLADeviceTensorDouble()
    staged_gguf = pt_quant._stage_xla_gguf_tensor(quant_xla)
    assert isinstance(staged_gguf, torch.Tensor)
    assert quant_xla.staged == ["cpu"]

    # Also exercise full quantize_model GGUF pipeline for both hf_transformers and pytorch_native
    native_gguf = tmp_path / "native_preset.gguf"
    q_nat = pt_quant.quantize_model(
        "gemma-4-tiny",
        "gguf",
        backend_alias="pytorch_native",
        export_path=str(native_gguf),
        out_type="q8_0",
    )
    assert q_nat["status"] == "quantized_gguf"
    assert q_nat["backend"] == "pytorch_native"
    assert native_gguf.is_file()

    # 2. Serve with backend_alias="hf_transformers" and hardware="tpu-v6e" (+ invoke _create_native_app)
    captured_serve_gen: dict[str, str] = {}

    def _fake_gen_sql(*, model_name: str, prompt: str, backend_alias: str, hardware: str) -> dict[str, str]:
        """Record forwarded `backend_alias` and `hardware` from `_create_native_app`.

        Returns:
            Simulated inference dictionary.

        """
        captured_serve_gen["model_name"] = model_name
        captured_serve_gen["prompt"] = prompt
        captured_serve_gen["backend_alias"] = backend_alias
        captured_serve_gen["hardware"] = hardware
        return {"sql": "SELECT 42;", "status": "success"}

    monkeypatch.setattr(pt_infer, "generate_sql", _fake_gen_sql)
    serve_res = pt_serve.serve_model(
        "google/gemma-4-E4B",
        port=8123,
        max_batch_size=8,
        backend_alias="hf_transformers",
        hardware="tpu-v6e",
        native_fallback=True,
    )
    assert serve_res["backend"] == "pytorch"
    assert serve_res.get("backend_variant") == "hf_transformers"

    app = pt_serve._create_native_app(
        "google/gemma-4-E4B",
        4,
        backend_alias="hf_transformers",
        hardware="tpu-v6e",
    )
    assert isinstance(app, DirectRunnableAppProtocol)
    assert app.run_direct("SELECT 42") == "SELECT 42;"
    assert captured_serve_gen["backend_alias"] == "hf_transformers"
    assert captured_serve_gen["hardware"] == "tpu-v6e"

    # 3. PEFT on XLA for both hf_transformers and pytorch_native (verifying XLA -> CPU -> XLA staging on save)
    peft_moves: list[str] = []

    class _FakePeftWrappedModel:
        """Movable PEFT model double recording `.to(...)` and `.save_pretrained(...)`."""

        def to(self, dev: str, /) -> Self:
            """Record `dev` and return `self`.

            Returns:
                `self`.

            """
            peft_moves.append(dev)
            return self

        def save_pretrained(self, save_dir: str) -> None:
            """Record `save:` marker confirming `.to('cpu')` occurred immediately prior."""
            peft_moves.append(f"save:{Path(save_dir).name}")

    class _FakeAutoLoader:
        """AutoModelForCausalLM double for PEFT on XLA."""

        @staticmethod
        def from_pretrained(_name: str) -> _FakePeftWrappedModel:
            """Return `_FakePeftWrappedModel`."""
            return _FakePeftWrappedModel()

    def _fake_lora_cfg(**_kw: int | float | str | Sequence[str]) -> str:
        """Return placeholder LoRA config string.

        Returns:
            `'cfg'`.

        """
        return "cfg"

    def _fake_get_peft(model: _FakePeftWrappedModel, _cfg: str) -> _FakePeftWrappedModel:
        """Return `model` unchanged.

        Returns:
            Unchanged `model`.

        """
        return model

    monkeypatch.setattr(pt_peft, "peft", xla_mod)
    monkeypatch.setattr(pt_peft, "AutoModelForCausalLM", _FakeAutoLoader)
    monkeypatch.setattr(pt_peft, "LoraConfig", _fake_lora_cfg)
    monkeypatch.setattr(pt_peft, "get_peft_model", _fake_get_peft)

    with patch.dict(sys.modules, {"torch_xla": xla_mod, "torch_xla.core.xla_model": xm_mod}):
        peft_hf = pt_peft.apply_lora(
            "google/gemma-4-E4B",
            ["q_proj"],
            backend_alias="hf_transformers",
            hardware="tpu-v6e",
            output_dir=str(tmp_path / "hf_lora"),
        )
        assert peft_hf["backend"] == "hf_transformers"
        assert peft_moves == ["xla:0", "xla:0", "cpu", "save:hf_lora"]

        # Native PEFT on XLA with monkeypatched nn.Module.to recording device staging
        tiny_cfg = Gemma4Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            num_experts=1,
        )
        nat_moves: list[str] = []

        def _record_peft_to(self: nn.Module, target: str) -> nn.Module:
            """Record target device without moving storage to unregistered XLA.

            Returns:
                `self`.

            """
            nat_moves.append(target)
            return self

        monkeypatch.setattr(nn.Module, "to", _record_peft_to)
        nat_peft_model = Gemma4ForCausalLM(tiny_cfg)
        pt_infer.MODEL_CACHE.put_model("nat_xla_lora", (nat_peft_model, None))
        nat_lora_dir = tmp_path / "nat_lora"
        peft_nat = pt_peft.apply_lora(
            "nat_xla_lora",
            ["q_proj"],
            lora_r=2,
            backend_alias="pytorch_native",
            hardware="tpu-v6e",
            output_dir=str(nat_lora_dir),
        )
        assert peft_nat["backend"] == "pytorch_native"
        assert nat_moves == ["xla:0", "cpu", "xla:0"]
        assert (nat_lora_dir / "model.safetensors").is_file()
    pt_infer.MODEL_CACHE.clear()
