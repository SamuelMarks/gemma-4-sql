"""Behavioral and unit tests for PyTorch training pipeline without untyped mocks."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gemma_4_sql.backends.pytorch.gemma4 import Gemma4Config, Gemma4ForCausalLM
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import TorchTrainerState, TrainingConfig
from tests.conftest import requires_torch

pytestmark = requires_torch


@pytest.fixture(autouse=True)
def _restore_train_mod(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore `train_module` globals after each test."""
    import gemma_4_sql.backends.pytorch.train as train_module

    monkeypatch.setattr(train_module, "torch", train_module.torch)
    monkeypatch.setattr(train_module, "nn", train_module.nn)
    monkeypatch.setattr(train_module, "optim", train_module.optim)
    monkeypatch.setattr(train_module, "Gemma4ForCausalLM", train_module.Gemma4ForCausalLM)


def _make_tiny_native_model() -> Gemma4ForCausalLM:
    """Create a minimal native `Gemma4ForCausalLM` instance for fast CPU training tests.

    Returns:
        Minimal `Gemma4ForCausalLM` instance.
    """
    cfg = Gemma4Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
        layer_types=["sliding_attention"],
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
    )
    return Gemma4ForCausalLM(cfg)


def test_real_tiny_pytorch_training_step() -> None:
    """Verify forward pass, cross-entropy loss, backward pass, and optimizer step on a real tiny `Gemma4ForCausalLM`."""
    import torch

    import gemma_4_sql.backends.pytorch.train as train_module

    model = _make_tiny_native_model()
    initial_head = model.lm_head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    batch = {
        "inputs": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "targets": torch.tensor([[2, 3, 4, 5]], dtype=torch.long),
    }
    state = TorchTrainerState(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        dataloader=[batch],
        epochs=2,
        device="cpu",
    )
    loss_val = train_module._run_training_epochs(state)
    assert isinstance(loss_val, float)
    assert loss_val > 0.0
    assert not torch.allclose(model.lm_head.weight.detach(), initial_head)


def test_setup_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_setup_distributed` handles DDP/FSDP initialization, CUDA detection, and single-device fallback."""
    import torch

    import gemma_4_sql.backends.pytorch.train as train_module

    init_backends: list[str] = []
    cuda_available = True

    class _FakeCuda:
        """Typed `torch.cuda` double."""

        def is_available(self) -> bool:
            """Return `cuda_available`."""
            return cuda_available

        def device_count(self) -> int:
            """Return `1`."""
            return 1

        def set_device(self, _idx: int) -> None:
            """No-op `set_device`."""

    class _FakeTorch:
        """Typed `torch` double exposing `.cuda` and `.device`."""

        cuda = _FakeCuda()
        device = torch.device

    with (
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torch.distributed.init_process_group", side_effect=init_backends.append),
        patch("torch.distributed.get_rank", return_value=0),
    ):
        monkeypatch.setattr(train_module, "torch", _FakeTorch())

        is_d, _d_mod, _dev, _d_id = train_module._setup_distributed("ddp")
        assert is_d is True
        assert init_backends == ["nccl"]

        cuda_available = False
        is_d2, _d_mod2, _dev2, _d_id2 = train_module._setup_distributed("ddp")
        assert is_d2 is True

        is_d_none, d_mod_none, _dev_none, _d_id_none = train_module._setup_distributed("none")
        assert is_d_none is False
        assert d_mod_none is None

    init_skip_calls: list[str] = []
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.init_process_group", side_effect=init_skip_calls.append),
        patch("torch.distributed.get_rank", return_value=0),
    ):
        train_module._setup_distributed("ddp")
        assert init_skip_calls == []


def test_run_training_epochs() -> None:
    """Verify `_run_training_epochs` updates `torch.nn.Module` parameters across tuple and object output formats."""
    import torch

    import gemma_4_sql.backends.pytorch.train as train_module

    class TupleTinyLM(torch.nn.Module):
        """Tiny causal LM returning a tuple `(logits,)`."""

        def __init__(self) -> None:
            """Initialize embedding and linear head."""
            super().__init__()
            self.emb = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
            """Return 1-tuple of logits."""
            return (self.head(self.emb(x)),)

    class ObjOut:
        """Output container with a `.logits` tensor attribute."""

        def __init__(self, logits: torch.Tensor) -> None:
            """Store logits tensor."""
            self.logits = logits

    class ObjectTinyLM(torch.nn.Module):
        """Tiny causal LM returning an object with `.logits`."""

        def __init__(self) -> None:
            """Initialize embedding and linear head."""
            super().__init__()
            self.emb = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16)

        def forward(self, x: torch.Tensor) -> ObjOut:
            """Return `ObjOut` wrapping logits."""
            return ObjOut(self.head(self.emb(x)))

    batch = {
        "inputs": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "targets": torch.tensor([[2, 3, 4]], dtype=torch.long),
    }

    tuple_model = TupleTinyLM()
    w_before = tuple_model.head.weight.detach().clone()
    opt1 = torch.optim.SGD(tuple_model.parameters(), lr=0.1)
    state1 = TorchTrainerState(
        model=tuple_model,
        optimizer=opt1,
        criterion=torch.nn.CrossEntropyLoss(),
        dataloader=[batch],
        epochs=2,
        device="cpu",
    )
    loss1 = train_module._run_training_epochs(state1)
    assert loss1 > 0.0
    assert not torch.allclose(tuple_model.head.weight.detach(), w_before)

    obj_model = ObjectTinyLM()
    w2_before = obj_model.head.weight.detach().clone()
    opt2 = torch.optim.SGD(obj_model.parameters(), lr=0.1)
    state2 = TorchTrainerState(
        model=obj_model,
        optimizer=opt2,
        criterion=torch.nn.CrossEntropyLoss(),
        dataloader=[batch],
        epochs=2,
        device="cpu",
    )
    loss2 = train_module._run_training_epochs(state2)
    assert loss2 > 0.0
    assert not torch.allclose(obj_model.head.weight.detach(), w2_before)


def test_wrap_model_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_wrap_model_distributed` wraps models with DDP, FSDP, or returns unwrapped model."""
    import gemma_4_sql.backends.pytorch.distributed as dist_mod
    import gemma_4_sql.backends.pytorch.train as train_module

    class _FakeParallelMod:
        """Typed module double exposing `DistributedDataParallel` and `FullyShardedDataParallel`."""

        def DistributedDataParallel(self, model: str, *, device_ids: list[int] | None = None) -> str:
            """Return wrapped DDP tag."""
            _ = device_ids
            return f"ddp({model})"

        def FullyShardedDataParallel(self, model: str, *, device_id: int | None = None) -> str:
            """Return wrapped FSDP tag."""
            _ = device_id
            return f"fsdp({model})"

    class _FakeCuda:
        """Typed CUDA double with `is_available() -> True`."""

        def is_available(self) -> bool:
            """Return `True`."""
            return True

    class _FakeTorch:
        """Typed `torch` double."""

        cuda = _FakeCuda()

    fake_parallel = _FakeParallelMod()

    def _get_fake_parallel(_name: str) -> _FakeParallelMod:
        """Return `fake_parallel`."""
        return fake_parallel

    monkeypatch.setattr(dist_mod, "optional_import", _get_fake_parallel)
    monkeypatch.setattr(train_module, "torch", _FakeTorch())
    m_stub = "m"
    assert train_module._wrap_model_distributed(m_stub, "ddp", 0) == "ddp(m)"
    assert train_module._wrap_model_distributed(m_stub, "fsdp", 0) == "fsdp(m)"
    assert train_module._wrap_model_distributed(m_stub, "none", 0) == "m"


def test_cleanup_distributed() -> None:
    """Verify `cleanup_distributed` destroys initialized process groups and ignores `None`."""
    import gemma_4_sql.backends.pytorch.train as train_module

    destroyed_calls = 0

    class _FakeDistModule:
        """Typed `TorchDistModuleProtocol` double."""

        def is_initialized(self) -> bool:
            """Return `True`."""
            return True

        def init_process_group(self, backend: str) -> None:
            """No-op `init_process_group`."""
            _ = backend

        def get_rank(self) -> int:
            """Return `0`."""
            return 0

        def destroy_process_group(self) -> None:
            """Increment `destroyed_calls`."""
            nonlocal destroyed_calls
            destroyed_calls += 1

    fake_dist = _FakeDistModule()
    fake_dist.init_process_group("gloo")
    assert fake_dist.get_rank() == 0
    train_module.cleanup_distributed(fake_dist)
    assert destroyed_calls == 1
    train_module.cleanup_distributed(None)


def test_execute_train(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_execute_train` runs real parameter updates on a tiny PyTorch model for native and HF aliases."""
    import torch

    import gemma_4_sql.backends.pytorch.train as train_module
    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4ForCausalLM

    batch = {
        "inputs": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "targets": torch.tensor([[2, 3, 4]], dtype=torch.long),
    }

    native_model = _make_tiny_native_model()
    w_before = native_model.lm_head.weight.detach().clone()

    def _from_pretrained_native(
        _cls: type[Gemma4ForCausalLM], *_a: str, **_k: Gemma4Config | None
    ) -> Gemma4ForCausalLM:
        """Return `native_model`."""
        return native_model

    monkeypatch.setattr(Gemma4ForCausalLM, "from_pretrained", classmethod(_from_pretrained_native))

    with (
        patch.object(train_module, "build_dataloader", return_value={}),
        pytest.raises(ValueError, match="Invalid dataloader"),
    ):
        train_module._execute_train("m", "real_ds", 1, 1e-2, "none", backend_alias="pytorch_native")

    with patch.object(train_module, "build_dataloader", return_value={"loader": [batch]}):
        status, loss = train_module._execute_train("m", "d", 1, 1e-2, "none", backend_alias="pytorch_native")
        assert status == "completed"
        assert loss > 0.0
        assert not torch.allclose(native_model.lm_head.weight.detach(), w_before)

        hf_model = _make_tiny_native_model()
        w_hf_before = hf_model.lm_head.weight.detach().clone()

        class _FakeHFLoader:
            """Typed `AutoModelForCausalLM` double."""

            @classmethod
            def from_pretrained(cls, _name: str) -> Gemma4ForCausalLM:
                """Return `hf_model`."""
                return hf_model

        monkeypatch.setattr(train_module, "Gemma4ForCausalLM", _FakeHFLoader)

        status2, loss2 = train_module._execute_train("m", "d", 1, 1e-2, "none", backend_alias="pytorch")
        assert status2 == "completed"
        assert loss2 > 0.0
        assert not torch.allclose(hf_model.lm_head.weight.detach(), w_hf_before)

    monkeypatch.setattr(train_module, "torch", None)
    with pytest.raises(DependencyMissingError):
        train_module._execute_train("m", "d", 1, 1e-2, "none")


def test_train_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `train_model` returns structured dict, propagates `TrainingError` when requested, and checks deps."""
    import gemma_4_sql.backends.pytorch.train as train_module

    class _DummyLoader:
        """Minimal `Gemma4ForCausalLM` sentinel."""

    monkeypatch.setattr(train_module, "Gemma4ForCausalLM", _DummyLoader)
    config = TrainingConfig(model_name="model")
    should_fail = False

    def _fake_exec(*_a: str | float, **_k: str | int | bool | None) -> tuple[str, float]:
        """Return `('completed', 0.5)` or raise `ValueError`.

        Raises:
            ValueError: If `should_fail` is True.
        """
        if should_fail:
            raise ValueError("err")
        return ("completed", 0.5)

    monkeypatch.setattr(train_module, "_execute_train", _fake_exec)
    res = train_module.train_model(config)
    assert res["status"] == "completed"
    assert res["final_loss"] == 0.5

    should_fail = True
    res2 = train_module.train_model(config)
    assert "failed: err" in res2["status"]
    with pytest.raises(TrainingError, match="err"):
        train_module.train_model(config, raise_on_error=True)

    monkeypatch.setattr(train_module, "torch", None)
    with pytest.raises(DependencyMissingError):
        train_module.train_model(config)
