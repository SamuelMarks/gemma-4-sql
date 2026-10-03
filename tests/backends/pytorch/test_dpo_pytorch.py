from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.pytorch import dpo
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import DPOConfig, TrainerState


def test_dpo_loss():
    # Test torch missing
    with patch("gemma_4_sql.backends.pytorch.dpo.torch", None):
        assert dpo.dpo_loss(None, None, None, None) == (0.0, 0.0, 0.0)

    mock_torch = MagicMock()
    mock_functional = MagicMock()
    with patch("gemma_4_sql.backends.pytorch.dpo.torch", mock_torch), patch("gemma_4_sql.backends.pytorch.dpo.functional", mock_functional), patch("gemma_4_sql.backends.pytorch.dpo.generic_dpo_loss") as mock_generic:
        mock_generic.return_value = (1.0, 0.5, 0.5)
        res = dpo.dpo_loss("pc", "pr", "rc", "rr", 0.2)
        assert res == (1.0, 0.5, 0.5)
        mock_generic.assert_called_with("pc", "pr", "rc", "rr", 0.2, mock_functional.logsigmoid)


def test_run_dpo_step():
    mock_torch = MagicMock()

    class MockModelContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    mock_torch.no_grad.return_value = MockModelContext()

    # Model that returns tensors with a mean method
    def mock_model(inputs):
        t = MagicMock()
        t.mean.return_value = f"mean_{inputs}"
        return t

    policy = MagicMock(side_effect=mock_model)
    ref = MagicMock(side_effect=mock_model)

    opt = MagicMock()

    batch = {"chosen_inputs": "ci", "rejected_inputs": "ri"}

    with patch("gemma_4_sql.backends.pytorch.dpo.torch", mock_torch), patch("gemma_4_sql.backends.pytorch.dpo.dpo_loss") as mock_dpo_loss:
        mock_loss = MagicMock()
        mock_dpo_loss.return_value = (mock_loss, None, None)

        loss = dpo._run_dpo_step(policy, ref, opt, batch, 0.5)

        opt.zero_grad.assert_called_once()
        opt.step.assert_called_once()
        mock_loss.backward.assert_called_once()
        mock_dpo_loss.assert_called_with("mean_ci", "mean_ri", "mean_ci", "mean_ri", 0.5)
        assert loss == mock_loss

    # Test without mean, item, backward, step etc
    def mock_model_no_mean(inputs):
        return inputs

    policy2 = MagicMock(side_effect=mock_model_no_mean)
    ref2 = MagicMock(side_effect=mock_model_no_mean)
    opt2 = MagicMock(spec=[])  # No zero_grad, step

    with patch("gemma_4_sql.backends.pytorch.dpo.torch", mock_torch), patch("gemma_4_sql.backends.pytorch.dpo.dpo_loss") as mock_dpo_loss:
        mock_loss2 = "simple_loss"  # no backward
        mock_dpo_loss.return_value = (mock_loss2, None, None)

        loss2 = dpo._run_dpo_step(policy2, ref2, opt2, batch, 0.5)
        assert loss2 == "simple_loss"
        mock_dpo_loss.assert_called_with("ci", "ri", "ci", "ri", 0.5)


def test_run_training_epochs():
    batch1 = {"b": 1}
    batch2 = {"b": 2}
    dataloader = [batch1, batch2]

    state = TrainerState(dataloader=dataloader, epochs=2, policy_model="p", ref_model="r", optimizer="o", beta=0.1)

    with patch("gemma_4_sql.backends.pytorch.dpo._run_dpo_step") as mock_step:
        # 2 epochs * 2 batches = 4 steps
        # return loss objects that have .item()
        mock_loss = MagicMock()
        mock_loss.item.return_value = 1.0
        mock_step.return_value = mock_loss

        final_loss = dpo._run_training_epochs(state)
        # Epoch 1: 1.0 + 1.0 = 2.0
        # Epoch 2: 1.0 + 1.0 = 2.0
        # Final loss: 2.0 / 2 = 1.0
        assert final_loss == 1.0

        # Test loss without item
        mock_step.return_value = 2.0
        final_loss2 = dpo._run_training_epochs(state)
        # Epoch 1: 4.0
        # Epoch 2: 4.0
        # Final: 2.0
        assert final_loss2 == 2.0


def test_run_dpo():
    config = DPOConfig(model_name="m", dataset="d")

    # Test dependencies missing
    with patch("gemma_4_sql.backends.pytorch.dpo.torch", None), pytest.raises(DependencyMissingError):
        dpo.run_dpo(config)

    mock_torch = MagicMock()
    mock_nn = MagicMock()
    mock_optim = MagicMock()

    with patch("gemma_4_sql.backends.pytorch.dpo.torch", mock_torch), patch("gemma_4_sql.backends.pytorch.dpo.nn", mock_nn), patch("gemma_4_sql.backends.pytorch.dpo.optim", mock_optim):
        # Test model loading failure
        with patch("builtins.__import__", side_effect=ImportError):
            res = dpo.run_dpo(config)
            assert res["status"] == "failed: Failed to load model m"
            assert res["final_loss"] == 0.0

        # Test invalid dataloader
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        orig_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "transformers.models.gemma4":
                return MagicMock(Gemma4ForCausalLM=mock_model_cls)
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import), patch("gemma_4_sql.backends.pytorch.dpo.build_dataloader", return_value={"loader": None}):
            res = dpo.run_dpo(config)
            assert res["status"].startswith("failed: Invalid dataloader")
            assert res["final_loss"] == 0.0

        # Test successful execution
        with patch("builtins.__import__", side_effect=fake_import), patch("gemma_4_sql.backends.pytorch.dpo.build_dataloader", return_value={"loader": [1, 2]}), patch("gemma_4_sql.backends.pytorch.dpo._run_training_epochs", return_value=3.14):
            res = dpo.run_dpo(config)
            assert res["status"] == "completed"
            assert res["final_loss"] == 3.14
