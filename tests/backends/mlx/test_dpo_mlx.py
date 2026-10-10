"""Tests for mlx dpo."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import DPOConfig


def test_mlx_dpo_imports():
    """Test mlx dpo imports fallback."""
    # Test mlx core fails
    with patch.dict(sys.modules, {"mlx": None, "mlx.core": None, "mlx.nn": None, "mlx.optimizers": None, "mlx_lm": MagicMock()}):
        if "gemma_4_sql.backends.mlx.dpo" in sys.modules:
            del sys.modules["gemma_4_sql.backends.mlx.dpo"]
        import gemma_4_sql.backends.mlx.dpo as dpo_module

        assert dpo_module.mlx is None
        assert dpo_module.mx is None

    # Test mlx_lm fails
    with patch.dict(sys.modules, {"mlx": MagicMock(), "mlx.core": MagicMock(), "mlx.nn": MagicMock(), "mlx.optimizers": MagicMock(), "mlx_lm": None}):
        if "gemma_4_sql.backends.mlx.dpo" in sys.modules:
            del sys.modules["gemma_4_sql.backends.mlx.dpo"]
        import gemma_4_sql.backends.mlx.dpo as dpo_module

        assert dpo_module.load is None


def test_dpo_loss():
    """Test dpo_loss."""
    import gemma_4_sql.backends.mlx.dpo as dpo_module

    dpo_module.mx = MagicMock()
    dpo_module.mx_nn = MagicMock(spec=[])

    with patch.object(dpo_module, "generic_dpo_loss") as mock_generic:
        mock_generic.return_value = (1, 2, 3)
        res = dpo_module.dpo_loss(1, 2, 3, 4, 0.1)
        assert res == (1, 2, 3)

        # Test fallback log_sig_fn
        log_sig_fn = mock_generic.call_args[0][5]

        # Trigger fallback logic with mx.negative
        dpo_module.mx.negative.return_value = "neg"
        assert log_sig_fn(1) == "neg"

        # Trigger fallback logic without mx.negative
        del dpo_module.mx.negative

        class MockVal:
            """Docstring for MockVal."""

            def __neg__(self):
                """Docstring for __neg__."""
                return "neg2"

        assert log_sig_fn(MockVal()) == "neg2"

    dpo_module.mx = None
    res = dpo_module.dpo_loss(1, 2, 3, 4, 0.1)
    assert res == (0.0, 0.0, 0.0)


def test_run_dpo_step():
    """Test _run_dpo_step."""
    import gemma_4_sql.backends.mlx.dpo as dpo_module

    mock_policy = MagicMock()
    mock_policy.return_value = MagicMock()
    mock_policy.return_value.mean.return_value = "logps"

    mock_ref = MagicMock()
    mock_ref.return_value = MagicMock()
    mock_ref.return_value.mean.return_value = "logps"

    mock_optimizer = MagicMock()

    batch = {"chosen_inputs": "ci", "rejected_inputs": "ri"}

    with patch.object(dpo_module, "dpo_loss") as mock_dpo_loss:
        mock_loss = MagicMock()
        mock_dpo_loss.return_value = (mock_loss, 0, 0)

        res = dpo_module._run_dpo_step(mock_policy, mock_ref, mock_optimizer, batch, 0.1)

        assert res == mock_loss
        mock_optimizer.zero_grad.assert_called_once()
        mock_loss.backward.assert_called_once()
        mock_optimizer.step.assert_called_once()

        # Test without hasattr
        del mock_optimizer.zero_grad
        del mock_loss.backward
        del mock_optimizer.step

        mock_policy.return_value = "no_mean"
        mock_ref.return_value = "no_mean"

        res = dpo_module._run_dpo_step(mock_policy, mock_ref, mock_optimizer, batch, 0.1)
        assert res == mock_loss


def test_run_training_epochs():
    """Test _run_training_epochs."""
    import gemma_4_sql.backends.mlx.dpo as dpo_module

    state = MagicMock()
    state.dataloader = [1, 2]
    state.epochs = 2
    state.policy_model = "p"
    state.ref_model = "r"
    state.optimizer = "o"
    state.beta = 0.1

    with patch.object(dpo_module, "_run_dpo_step") as mock_step:
        mock_step.return_value = MagicMock(item=lambda: 0.5)

        loss = dpo_module._run_training_epochs(state)
        assert loss == 0.5
        assert mock_step.call_count == 4


def test_run_dpo():
    """Test run_dpo."""
    import gemma_4_sql.backends.mlx.dpo as dpo_module

    dpo_module.mlx = MagicMock()
    dpo_module.mx = MagicMock()
    dpo_module.nn = MagicMock()
    dpo_module.optim = MagicMock()
    dpo_module.load = MagicMock()

    config = DPOConfig(model_name="model", dataset="dataset")

    dpo_module.load.__call__ = MagicMock(return_value=("model1", "tok1"))

    with patch.object(dpo_module, "build_dataloader") as mock_build_dataloader:
        mock_build_dataloader.return_value = {"loader": [1, 2]}

        with patch.object(dpo_module, "_run_training_epochs") as mock_run_epochs:
            mock_run_epochs.return_value = 0.5

            res = dpo_module.run_dpo(config)
            assert res["status"] == "completed"
            assert res["final_loss"] == 0.5

            # Test error
            mock_run_epochs.side_effect = RuntimeError("error")
            res2 = dpo_module.run_dpo(config)
            assert "failed: error" in res2["status"]

        # Test invalid dataloader - None
        mock_build_dataloader.return_value = {}
        res3 = dpo_module.run_dpo(config)
        assert "failed: Invalid dataloader" in res3["status"]

        # Test invalid dataloader - no __iter__
        mock_build_dataloader.return_value = {"loader": MagicMock(spec=[])}
        res4 = dpo_module.run_dpo(config)
        assert "failed: Invalid dataloader" in res4["status"]

    dpo_module.mlx = None
    with pytest.raises(DependencyMissingError):
        dpo_module.run_dpo(config)


def test_mlx_dpo_successful_imports():
    """Test mlx dpo successful imports."""
    import sys
    from unittest.mock import MagicMock

    mock_mlx = MagicMock()
    mock_core = MagicMock()
    mock_nn = MagicMock()
    mock_optim = MagicMock()
    mock_lm = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "mlx": mock_mlx,
            "mlx.core": mock_core,
            "mlx.nn": mock_nn,
            "mlx.optimizers": mock_optim,
            "mlx_lm": mock_lm,
        },
    ):
        if "gemma_4_sql.backends.mlx.dpo" in sys.modules:
            del sys.modules["gemma_4_sql.backends.mlx.dpo"]
        import gemma_4_sql.backends.mlx.dpo as dpo_module

        assert dpo_module.mlx is mock_mlx
        assert dpo_module.mx is not None
        assert dpo_module.nn is not None
        assert dpo_module.optim is not None
        assert dpo_module.load is not None
