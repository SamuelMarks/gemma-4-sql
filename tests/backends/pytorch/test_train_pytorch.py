"""Behavioral and unit tests for PyTorch training pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import TrainerState, TrainingConfig
from tests.conftest import requires_torch

pytestmark = requires_torch


@pytest.fixture(autouse=True)
def _restore_train_mod(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore train_module globals after each test."""
    import gemma_4_sql.backends.pytorch.train as train_module

    monkeypatch.setattr(train_module, "torch", train_module.torch)
    monkeypatch.setattr(train_module, "nn", train_module.nn)
    monkeypatch.setattr(train_module, "optim", train_module.optim)
    monkeypatch.setattr(train_module, "Gemma4ForCausalLM", train_module.Gemma4ForCausalLM)


def _make_tiny_native_model() -> Any:
    """Create a minimal native Gemma4ForCausalLM instance for fast CPU training tests."""
    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4Config, Gemma4ForCausalLM

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
    """Verify forward pass, cross-entropy loss, backward pass, and optimizer step on a real tiny Gemma4ForCausalLM."""
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
    state: Any = TrainerState(
        policy_model=model,
        optimizer=optimizer,
        criterion=criterion,
        dataloader=[batch],
        epochs=2,
        device=torch.device("cpu"),
    )
    loss_val = train_module._run_training_epochs(state)
    assert isinstance(loss_val, float)
    assert loss_val > 0.0
    assert not torch.allclose(model.lm_head.weight.detach(), initial_head)


def test_setup_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _setup_distributed handles DDP/FSDP initialization, CUDA detection, and single-device fallback."""
    import gemma_4_sql.backends.pytorch.train as train_module

    mock_dist = MagicMock()
    mock_dist.is_initialized.return_value = False
    mock_dist.get_rank.return_value = 0

    with (
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torch.distributed.init_process_group") as mock_init,
        patch("torch.distributed.get_rank", return_value=0),
    ):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        monkeypatch.setattr(train_module, "torch", mock_torch)

        is_d, _d_mod, _dev, _d_id = train_module._setup_distributed("ddp")
        assert is_d is True
        mock_init.assert_called_once_with("nccl")

        mock_torch.cuda.is_available.return_value = False
        is_d2, _d_mod2, _dev2, _d_id2 = train_module._setup_distributed("ddp")
        assert is_d2 is True

        is_d_none, d_mod_none, _dev_none, _d_id_none = train_module._setup_distributed("none")
        assert is_d_none is False
        assert d_mod_none is None

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.init_process_group") as mock_init_skip,
        patch("torch.distributed.get_rank", return_value=0),
    ):
        train_module._setup_distributed("ddp")
        mock_init_skip.assert_not_called()


def test_run_training_epochs() -> None:
    """Verify _run_training_epochs updates real torch.nn.Module parameters across tuple and object output formats."""
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
            """Return ObjOut wrapping logits."""
            return ObjOut(self.head(self.emb(x)))

    batch = {
        "inputs": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "targets": torch.tensor([[2, 3, 4]], dtype=torch.long),
    }

    tuple_model = TupleTinyLM()
    w_before = tuple_model.head.weight.detach().clone()
    opt1 = torch.optim.SGD(tuple_model.parameters(), lr=0.1)
    state1: Any = TrainerState(
        policy_model=tuple_model,
        optimizer=opt1,
        criterion=torch.nn.CrossEntropyLoss(),
        dataloader=[batch],
        epochs=2,
        device=torch.device("cpu"),
    )
    loss1 = train_module._run_training_epochs(state1)
    assert loss1 > 0.0
    assert not torch.allclose(tuple_model.head.weight.detach(), w_before)

    obj_model = ObjectTinyLM()
    w2_before = obj_model.head.weight.detach().clone()
    opt2 = torch.optim.SGD(obj_model.parameters(), lr=0.1)
    state2: Any = TrainerState(
        policy_model=obj_model,
        optimizer=opt2,
        criterion=torch.nn.CrossEntropyLoss(),
        dataloader=[batch],
        epochs=2,
        device=torch.device("cpu"),
    )
    loss2 = train_module._run_training_epochs(state2)
    assert loss2 > 0.0
    assert not torch.allclose(obj_model.head.weight.detach(), w2_before)


def test_wrap_model_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _wrap_model_distributed wraps models with DDP, FSDP, or returns unwrapped model."""
    import gemma_4_sql.backends.pytorch.train as train_module

    with patch("importlib.import_module") as mock_import:
        mock_ddp = MagicMock()
        mock_import.return_value.DistributedDataParallel = mock_ddp
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        monkeypatch.setattr(train_module, "torch", mock_torch)
        m_stub: Any = "m"
        train_module._wrap_model_distributed(m_stub, "ddp", 0)

        mock_fsdp = MagicMock()
        mock_import.return_value.FullyShardedDataParallel = mock_fsdp
        train_module._wrap_model_distributed(m_stub, "fsdp", 0)

        res = train_module._wrap_model_distributed(m_stub, "none", 0)
        assert res == "m"


def test_cleanup_distributed() -> None:
    """Verify _cleanup_distributed destroys initialized process groups and ignores None."""
    import gemma_4_sql.backends.pytorch.train as train_module

    mock_dist = MagicMock()
    mock_dist.is_initialized.return_value = True
    train_module._cleanup_distributed(mock_dist)
    mock_dist.destroy_process_group.assert_called_once()

    train_module._cleanup_distributed(None)


def test_execute_train(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _execute_train runs real parameter updates on a tiny PyTorch model for native and HF aliases."""
    import torch

    import gemma_4_sql.backends.pytorch.train as train_module
    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4ForCausalLM

    batch = {
        "inputs": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "targets": torch.tensor([[2, 3, 4]], dtype=torch.long),
    }

    native_model = _make_tiny_native_model()
    w_before = native_model.lm_head.weight.detach().clone()
    monkeypatch.setattr(Gemma4ForCausalLM, "from_pretrained", classmethod(lambda _cls, *_a, **_k: native_model))

    with patch.object(train_module, "build_dataloader", return_value={"loader": [batch]}):
        status, loss = train_module._execute_train("m", "d", 1, 1e-2, "none", backend_alias="pytorch_native")
        assert status == "completed"
        assert loss > 0.0
        assert not torch.allclose(native_model.lm_head.weight.detach(), w_before)

        hf_model = _make_tiny_native_model()
        w_hf_before = hf_model.lm_head.weight.detach().clone()
        mock_hf_cls = MagicMock()
        mock_hf_cls.from_pretrained.return_value = hf_model
        monkeypatch.setattr(train_module, "Gemma4ForCausalLM", mock_hf_cls)

        status2, loss2 = train_module._execute_train("m", "d", 1, 1e-2, "none", backend_alias="pytorch")
        assert status2 == "completed"
        assert loss2 > 0.0
        assert not torch.allclose(hf_model.lm_head.weight.detach(), w_hf_before)

    with (
        patch.object(train_module, "build_dataloader", return_value={}),
        pytest.raises(ValueError, match="Invalid dataloader"),
    ):
        train_module._execute_train("m", "d", 1, 1e-2, "none")

    monkeypatch.setattr(train_module, "torch", None)
    with pytest.raises(DependencyMissingError):
        train_module._execute_train("m", "d", 1, 1e-2, "none")


def test_train_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify train_model returns structured dictionary, propagates TrainingError when requested, and checks deps."""
    import gemma_4_sql.backends.pytorch.train as train_module

    monkeypatch.setattr(train_module, "Gemma4ForCausalLM", MagicMock())
    config = TrainingConfig(model_name="model")

    with patch.object(train_module, "_execute_train") as mock_exec:
        mock_exec.return_value = ("completed", 0.5)

        res = train_module.train_model(config)
        assert res["status"] == "completed"
        assert res["final_loss"] == 0.5

        mock_exec.side_effect = ValueError("err")
        res2 = train_module.train_model(config)
        assert "failed: err" in res2["status"]
        with pytest.raises(TrainingError, match="err"):
            train_module.train_model(config, raise_on_error=True)

    monkeypatch.setattr(train_module, "torch", None)
    with pytest.raises(DependencyMissingError):
        train_module.train_model(config)
