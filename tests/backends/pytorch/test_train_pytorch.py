"""Tests for PyTorch train."""

from unittest.mock import MagicMock, patch

from tests.conftest import requires_torch

pytestmark = requires_torch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import TrainerState, TrainingConfig


@pytest.fixture(autouse=True)
def _restore_train_mod(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure train_module globals are restored after each test."""
    import gemma_4_sql.backends.pytorch.train as train_module

    monkeypatch.setattr(train_module, "torch", train_module.torch)
    monkeypatch.setattr(train_module, "nn", train_module.nn)
    monkeypatch.setattr(train_module, "optim", train_module.optim)
    monkeypatch.setattr(train_module, "Gemma4ForCausalLM", train_module.Gemma4ForCausalLM)


def test_real_tiny_pytorch_training_step() -> None:
    """Verify forward pass, cross-entropy loss, backward pass, and optimizer step on a real tiny Gemma4ForCausalLM."""
    import torch

    import gemma_4_sql.backends.pytorch.train as train_module
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
    model = Gemma4ForCausalLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    batch = {
        "inputs": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "targets": torch.tensor([[2, 3, 4, 5]], dtype=torch.long),
    }
    state = TrainerState(
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


def test_setup_distributed():
    """Test _setup_distributed."""
    import gemma_4_sql.backends.pytorch.train as train_module

    mock_dist = MagicMock()
    mock_dist.is_initialized.return_value = False
    mock_dist.get_rank.return_value = 0

    with patch("torch.distributed.is_initialized", return_value=False), patch("torch.distributed.init_process_group") as mock_init, patch("torch.distributed.get_rank", return_value=0):
        train_module.torch = MagicMock()
        train_module.torch.cuda.is_available.return_value = True
        train_module.torch.cuda.device_count.return_value = 1

        is_d, d_mod, dev, d_id = train_module._setup_distributed("ddp")
        assert is_d is True
        mock_init.assert_called_once_with("nccl")

        # Test not available
        train_module.torch.cuda.is_available.return_value = False
        is_d2, d_mod2, dev2, d_id2 = train_module._setup_distributed("ddp")

        # Test initialized
        mock_dist.is_initialized.return_value = True
        mock_dist.init_process_group.reset_mock()
        train_module._setup_distributed("ddp")
        mock_dist.init_process_group.assert_not_called()

        # Test not distributed
        is_d, d_mod, dev, d_id = train_module._setup_distributed("none")
        assert is_d is False
        assert d_mod is None


def test_run_training_epochs():
    """Test _run_training_epochs."""
    import gemma_4_sql.backends.pytorch.train as train_module

    state = MagicMock()
    state.epochs = 2
    state.device = "cpu"

    mock_batch = {"inputs": MagicMock(), "targets": MagicMock()}
    state.dataloader = [mock_batch]

    mock_model = MagicMock()
    mock_loss = MagicMock()
    mock_loss.item.return_value = 0.5
    state.criterion.return_value = mock_loss

    # Check outputs tuple
    mock_model.return_value = (MagicMock(),)
    state.policy_model = mock_model

    res = train_module._run_training_epochs(state)
    assert res == 0.5

    # Check outputs obj
    class Obj:
        """Execute and verify `Obj` behavior and invariants."""

        logits = MagicMock()

    mock_model.return_value = Obj()
    res2 = train_module._run_training_epochs(state)
    assert res2 == 0.5


def test_wrap_model_distributed():
    """Test _wrap_model_distributed."""
    import gemma_4_sql.backends.pytorch.train as train_module

    with patch("importlib.import_module") as mock_import:
        mock_ddp = MagicMock()
        mock_import.return_value.DistributedDataParallel = mock_ddp
        train_module.torch = MagicMock()
        train_module.torch.cuda.is_available.return_value = True
        train_module._wrap_model_distributed("m", "ddp", 0)

        mock_fsdp = MagicMock()
        mock_import.return_value.FullyShardedDataParallel = mock_fsdp
        train_module._wrap_model_distributed("m", "fsdp", 0)

        res = train_module._wrap_model_distributed("m", "none", 0)
        assert res == "m"


def test_cleanup_distributed():
    """Test _cleanup_distributed."""
    import gemma_4_sql.backends.pytorch.train as train_module

    mock_dist = MagicMock()
    mock_dist.is_initialized.return_value = True
    train_module._cleanup_distributed(mock_dist)
    mock_dist.destroy_process_group.assert_called_once()

    train_module._cleanup_distributed(None)


def test_execute_train():
    """Test _execute_train."""
    import gemma_4_sql.backends.pytorch.train as train_module

    train_module.torch = MagicMock()
    train_module.nn = MagicMock()
    train_module.optim = MagicMock()
    train_module.Gemma4ForCausalLM = MagicMock()

    with patch("gemma_4_sql.backends.pytorch.train._setup_distributed") as mock_setup:
        mock_setup.return_value = (False, None, "cpu", 0)

        with patch("gemma_4_sql.backends.pytorch.train.build_dataloader") as mock_build:
            mock_build.return_value = {"loader": [1, 2]}

            with patch("gemma_4_sql.backends.pytorch.train._run_training_epochs") as mock_run:
                mock_run.return_value = 0.5

                # Test native
                with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM") as mock_native:
                    mock_native.from_pretrained.return_value.to.return_value = MagicMock()
                    res = train_module._execute_train("m", "d", 1, 0.1, "none", backend_alias="pytorch_native")
                    assert res[0] == "completed"

                # Test HF
                res2 = train_module._execute_train("m", "d", 1, 0.1, "none", backend_alias="pytorch")
                assert res2[0] == "completed"

                # Test bad dataloader
                mock_build.return_value = {}
                with pytest.raises(ValueError):
                    train_module._execute_train("m", "d", 1, 0.1, "none")

    train_module.torch = None
    with pytest.raises(DependencyMissingError):
        train_module._execute_train("m", "d", 1, 0.1, "none")


def test_train_model():
    """Test train_model."""
    import gemma_4_sql.backends.pytorch.train as train_module

    train_module.torch = MagicMock()
    train_module.nn = MagicMock()
    train_module.optim = MagicMock()
    train_module.Gemma4ForCausalLM = MagicMock()

    config = TrainingConfig(model_name="model")

    with patch("gemma_4_sql.backends.pytorch.train._execute_train") as mock_exec:
        mock_exec.return_value = ("completed", 0.5)

        res = train_module.train_model(config)
        assert res["status"] == "completed"

        mock_exec.side_effect = ValueError("err")
        res2 = train_module.train_model(config)
        assert "failed: err" in res2["status"]

    train_module.torch = None
    with pytest.raises(DependencyMissingError):
        train_module.train_model(config)
