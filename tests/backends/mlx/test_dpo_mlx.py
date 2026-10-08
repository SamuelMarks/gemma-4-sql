"""Module docstring."""

from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.mlx.dpo as mlx_dpo
from gemma_4_sql.backends.mlx.dpo import _run_dpo_step, _run_training_epochs, dpo_loss, run_dpo
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import DPOConfig, TrainerState


@pytest.fixture(autouse=True)
def mock_mlx_deps(monkeypatch):
    """Docstring for mock_mlx_deps."""
    mock_mlx = MagicMock()
    mock_mx = MagicMock()
    mock_nn = MagicMock()
    mock_mx_nn = MagicMock()
    mock_optim = MagicMock()
    mock_load = MagicMock()
    monkeypatch.setattr(mlx_dpo, "mlx", mock_mlx)
    monkeypatch.setattr(mlx_dpo, "mx", mock_mx)
    monkeypatch.setattr(mlx_dpo, "nn", mock_nn)
    monkeypatch.setattr(mlx_dpo, "mx_nn", mock_mx_nn)
    monkeypatch.setattr(mlx_dpo, "optim", mock_optim)
    monkeypatch.setattr(mlx_dpo, "load", mock_load)
    return mock_mx, mock_mx_nn


def test_dpo_loss_missing_deps(monkeypatch):
    """Docstring for test_dpo_loss_missing_deps."""
    monkeypatch.setattr(mlx_dpo, "mx", None)
    assert dpo_loss(1, 1, 1, 1) == (0.0, 0.0, 0.0)


def test_dpo_loss(mock_mlx_deps):
    """Docstring for test_dpo_loss."""
    mock_mx, mock_mx_nn = mock_mlx_deps
    mock_mx.negative.side_effect = lambda x: -x
    mock_mx_nn.losses.log_sigmoid = lambda x: -x

    # Generic DPO loss mock
    with patch("gemma_4_sql.backends.mlx.dpo.generic_dpo_loss") as mock_generic:
        mock_generic.return_value = (1.0, 2.0, 3.0)
        res = dpo_loss(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        assert res == (1.0, 2.0, 3.0)


def test_dpo_loss_fallback_log_sig(mock_mlx_deps):
    """Docstring for test_dpo_loss_fallback_log_sig."""
    mock_mx, mock_mx_nn = mock_mlx_deps
    del mock_mx.negative  # fallback to -x
    del mock_mx_nn.losses.log_sigmoid

    with patch("gemma_4_sql.backends.mlx.dpo.generic_dpo_loss") as mock_generic:

        def call_generic(pc, pr, rc, rr, b, log_sig):
            """Docstring for call_generic."""
            # Test fallback log_sig
            assert log_sig(5) == -5
            return (1.0, 2.0, 3.0)

        mock_generic.side_effect = call_generic
        res = dpo_loss(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        assert res == (1.0, 2.0, 3.0)


def test_run_dpo_step():
    """Docstring for test_run_dpo_step."""
    policy_model = MagicMock()
    ref_model = MagicMock()
    optimizer = MagicMock()
    batch = {"chosen_inputs": 1, "rejected_inputs": 2}

    policy_model.return_value.mean.return_value = MagicMock()
    ref_model.return_value.mean.return_value = MagicMock()

    with patch("gemma_4_sql.backends.mlx.dpo.dpo_loss") as mock_loss:
        mock_loss_val = MagicMock()
        mock_loss.return_value = (mock_loss_val, 0, 0)

        res = _run_dpo_step(policy_model, ref_model, optimizer, batch, 0.1)
        assert res == mock_loss_val
        mock_loss_val.backward.assert_called_once()
        optimizer.step.assert_called_once()
        optimizer.zero_grad.assert_called_once()


def test_run_dpo_step_no_mean_no_backward():
    """Docstring for test_run_dpo_step_no_mean_no_backward."""
    policy_model = MagicMock()
    ref_model = MagicMock()
    optimizer = MagicMock()
    del optimizer.zero_grad
    del optimizer.step

    # Return raw mocks not having mean
    pi_ch = MagicMock(spec=[])
    policy_model.return_value = pi_ch

    with patch("gemma_4_sql.backends.mlx.dpo.dpo_loss") as mock_loss:
        mock_loss_val = MagicMock(spec=[])
        mock_loss.return_value = (mock_loss_val, 0, 0)

        res = _run_dpo_step(policy_model, ref_model, optimizer, {"chosen_inputs": 1, "rejected_inputs": 2}, 0.1)
        assert res == mock_loss_val


def test_run_training_epochs():
    """Docstring for test_run_training_epochs."""
    dataloader = [{"chosen_inputs": 1, "rejected_inputs": 2}]
    state = TrainerState(dataloader=dataloader, epochs=2, policy_model=MagicMock(), ref_model=MagicMock(), optimizer=MagicMock(), beta=0.1)

    with patch("gemma_4_sql.backends.mlx.dpo._run_dpo_step") as mock_step:
        mock_loss = MagicMock()
        mock_loss.item.return_value = 5.0
        mock_step.return_value = mock_loss

        loss = _run_training_epochs(state)
        assert loss == 5.0


def test_run_training_epochs_no_item():
    """Docstring for test_run_training_epochs_no_item."""
    dataloader = [{"chosen_inputs": 1, "rejected_inputs": 2}]
    state = TrainerState(dataloader=dataloader, epochs=1, policy_model=MagicMock(), ref_model=MagicMock(), optimizer=MagicMock(), beta=0.1)

    with patch("gemma_4_sql.backends.mlx.dpo._run_dpo_step") as mock_step:
        mock_loss = 5.0  # no item()
        mock_step.return_value = mock_loss

        loss = _run_training_epochs(state)
        assert loss == 5.0


def test_run_dpo_missing_deps(monkeypatch):
    """Docstring for test_run_dpo_missing_deps."""
    monkeypatch.setattr(mlx_dpo, "mlx", None)
    with pytest.raises(DependencyMissingError, match="MLX dependencies are missing."):
        run_dpo(DPOConfig(model_name="model", dataset="ds"))


def test_run_dpo_success(monkeypatch):
    """Docstring for test_run_dpo_success."""
    monkeypatch.setattr(mlx_dpo, "load", MagicMock(return_value=(MagicMock(), MagicMock())))

    with patch("gemma_4_sql.backends.mlx.dpo.build_dataloader") as mock_build:
        mock_loader = MagicMock()
        mock_loader.__iter__.return_value = iter([{"chosen_inputs": 1}])
        mock_build.return_value = {"loader": mock_loader}

        with patch("gemma_4_sql.backends.mlx.dpo._run_training_epochs") as mock_epochs:
            mock_epochs.return_value = 1.0

            res = run_dpo(DPOConfig(model_name="model", dataset="ds", epochs=1))
            assert res["status"] == "completed"
            assert res["final_loss"] == 1.0


def test_run_dpo_tuple_load(monkeypatch):
    """Docstring for test_run_dpo_tuple_load."""
    mock_load = MagicMock()
    mock_load.return_value = [MagicMock(), MagicMock()]  # test list return
    monkeypatch.setattr(mlx_dpo, "load", mock_load)

    with patch("gemma_4_sql.backends.mlx.dpo.build_dataloader") as mock_build:
        mock_loader = MagicMock()
        mock_build.return_value = {"loader": mock_loader}

        with patch("gemma_4_sql.backends.mlx.dpo._run_training_epochs"):
            res = run_dpo(DPOConfig(model_name="model", dataset="ds", epochs=1))
            assert res["status"] == "completed"


def test_run_dpo_invalid_dataloader(monkeypatch):
    """Docstring for test_run_dpo_invalid_dataloader."""
    monkeypatch.setattr(mlx_dpo, "load", MagicMock(return_value=MagicMock()))
    with patch("gemma_4_sql.backends.mlx.dpo.build_dataloader") as mock_build:
        mock_build.return_value = {"loader": None}  # Invalid loader

        res = run_dpo(DPOConfig(model_name="model", dataset="ds"))
        assert "failed" in res["status"]


def test_run_dpo_exception(monkeypatch):
    """Docstring for test_run_dpo_exception."""
    monkeypatch.setattr(mlx_dpo, "load", MagicMock(side_effect=RuntimeError("Fail")))
    res = run_dpo(DPOConfig(model_name="model", dataset="ds"))
    assert "failed: Fail" in res["status"]
