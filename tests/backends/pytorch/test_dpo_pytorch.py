"""Tests for PyTorch dpo."""

from unittest.mock import MagicMock, patch

from tests.conftest import requires_torch

pytestmark = requires_torch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import DPOConfig


@pytest.fixture(autouse=True)
def _restore_dpo_mod(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore dpo_module globals after each test."""
    import gemma_4_sql.backends.pytorch.dpo as dpo_module

    monkeypatch.setattr(dpo_module, "torch", dpo_module.torch)
    monkeypatch.setattr(dpo_module, "nn", dpo_module.nn)
    monkeypatch.setattr(dpo_module, "optim", dpo_module.optim)
    monkeypatch.setattr(dpo_module, "functional", dpo_module.functional)


def test_real_pytorch_dpo_loss_and_step() -> None:
    """Verify DPO loss and optimization step with real PyTorch tensors and nn.Linear models."""
    import torch

    import gemma_4_sql.backends.pytorch.dpo as dpo_module

    p_chosen = torch.tensor([-1.0, -1.2], requires_grad=True)
    p_rejected = torch.tensor([-2.0, -2.5], requires_grad=True)
    r_chosen = torch.tensor([-1.1, -1.3])
    r_rejected = torch.tensor([-1.9, -2.4])

    loss, chosen_rew, rejected_rew = dpo_module.dpo_loss(p_chosen, p_rejected, r_chosen, r_rejected, beta=0.1)
    assert float(loss.item()) > 0.0
    assert float(chosen_rew.mean().item()) > float(rejected_rew.mean().item())

    policy_model = torch.nn.Linear(4, 2)
    ref_model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(policy_model.parameters(), lr=0.05)
    batch = {
        "chosen_inputs": torch.ones((2, 4)),
        "rejected_inputs": torch.zeros((2, 4)),
    }
    step_loss = dpo_module._run_dpo_step(policy_model, ref_model, optimizer, batch, 0.1)  # type: ignore[arg-type]
    assert float(step_loss.item()) > 0.0


def test_dpo_loss():
    """Test dpo_loss."""
    import gemma_4_sql.backends.pytorch.dpo as dpo_module

    dpo_module.torch = MagicMock()
    dpo_module.functional = MagicMock()

    with patch("gemma_4_sql.backends.pytorch.dpo.generic_dpo_loss") as mock_generic:
        mock_generic.return_value = (1, 2, 3)
        res = dpo_module.dpo_loss(1, 2, 3, 4, 0.1)
        assert res == (1, 2, 3)
        mock_generic.assert_called_once_with(1, 2, 3, 4, 0.1, dpo_module.functional.logsigmoid)

    dpo_module.torch = None
    res = dpo_module.dpo_loss(1, 2, 3, 4, 0.1)
    assert res == (0.0, 0.0, 0.0)


def test_run_dpo_step():
    """Test _run_dpo_step."""
    import gemma_4_sql.backends.pytorch.dpo as dpo_module

    dpo_module.torch = MagicMock()

    mock_policy = MagicMock()
    mock_ref = MagicMock()
    mock_optim = MagicMock()

    mock_policy.return_value.mean.return_value = "mean"
    mock_ref.return_value.mean.return_value = "mean"

    batch = {"chosen_inputs": "c", "rejected_inputs": "r"}

    with patch("gemma_4_sql.backends.pytorch.dpo.dpo_loss") as mock_loss:
        mock_l = MagicMock()
        mock_loss.return_value = (mock_l, 0, 0)

        res = dpo_module._run_dpo_step(mock_policy, mock_ref, mock_optim, batch, 0.1)
        assert res == mock_l

        mock_optim.zero_grad.assert_called_once()
        mock_l.backward.assert_called_once()
        mock_optim.step.assert_called_once()

        # Test without methods
        del mock_optim.zero_grad
        del mock_l.backward
        del mock_optim.step

        mock_policy.return_value = "no_mean"
        mock_ref.return_value = "no_mean"

        res2 = dpo_module._run_dpo_step(mock_policy, mock_ref, mock_optim, batch, 0.1)
        assert res2 == mock_l


def test_run_training_epochs():
    """Test _run_training_epochs."""
    import gemma_4_sql.backends.pytorch.dpo as dpo_module

    state = MagicMock()
    state.dataloader = [1, 2]
    state.epochs = 2

    with patch("gemma_4_sql.backends.pytorch.dpo._run_dpo_step") as mock_step:
        mock_l = MagicMock()
        mock_l.item.return_value = 0.5
        mock_step.return_value = mock_l

        res = dpo_module._run_training_epochs(state)
        assert res == 0.5
        assert mock_step.call_count == 4

        # test float
        mock_step.return_value = 0.5
        res = dpo_module._run_training_epochs(state)
        assert res == 0.5


def test_run_dpo():
    """Test run_dpo."""
    import gemma_4_sql.backends.pytorch.dpo as dpo_module

    dpo_module.torch = MagicMock()
    dpo_module.nn = MagicMock()
    dpo_module.optim = MagicMock()

    config = DPOConfig(model_name="model", dataset="dataset")

    with patch("builtins.__import__") as mock_import:
        mock_cls = MagicMock()
        mock_cls.Gemma4ForCausalLM.from_pretrained.return_value = MagicMock()
        mock_import.return_value = mock_cls

        with patch.object(dpo_module, "build_dataloader") as mock_build:
            mock_build.return_value = {"loader": [1, 2]}

            with patch.object(dpo_module, "_run_training_epochs") as mock_run:
                mock_run.return_value = 0.5

                res = dpo_module.run_dpo(config)
                assert res["status"] == "completed"
                assert res["final_loss"] == 0.5

                # Test bad loader
                mock_build.return_value = {}
                res = dpo_module.run_dpo(config)
                assert "failed: Invalid dataloader" in res["status"]

                # Test error from epochs
                mock_build.return_value = {"loader": [1]}
                mock_run.side_effect = RuntimeError("error")
                res = dpo_module.run_dpo(config)
                assert "failed: error" in res["status"]

        # Test model load fail
        mock_cls.Gemma4ForCausalLM.from_pretrained.side_effect = ImportError("error")
        res = dpo_module.run_dpo(config)
        assert "failed: Failed to load model" in res["status"]

    dpo_module.torch = None
    with pytest.raises(DependencyMissingError):
        dpo_module.run_dpo(config)
