"""Module docstring."""

import builtins
import importlib
from unittest.mock import MagicMock, patch

import gemma_4_sql.backends.maxtext.dpo as mod


def test_maxtext_dpo_import_error():
    """Docstring for test_maxtext_dpo_import_error."""
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name in ("jax", "jax.numpy", "optax"):
            raise ImportError(f"Mock missing {name}")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        importlib.reload(mod)
        assert mod.jax is None
        assert mod.jnp is None
        assert mod.optax is None

    importlib.reload(mod)  # Restore


def test_maxtext_dpo_train_step():
    """Docstring for test_maxtext_dpo_train_step."""

    def fake_value_and_grad(fn):
        """Docstring for fake_value_and_grad."""

        def return_fn(*args):
            """Docstring for return_fn."""
            return "loss", "grads"

        return return_fn

    with patch.object(mod, "jax", MagicMock(value_and_grad=fake_value_and_grad)):
        with patch.object(mod, "optax", MagicMock()):
            with patch.object(mod, "_dpo_step_loss", return_value="loss"):
                train_step = mod._get_train_step_fn(MagicMock(), MagicMock(), MagicMock(), 0.1)
                train_step({}, {}, {})


def test_maxtext_dpo_compute_logps_with_numpy(monkeypatch):
    """Docstring for test_maxtext_dpo_compute_logps_with_numpy."""
    import gemma_4_sql.backends.maxtext.dpo as mod

    mock_jnp = MagicMock()
    mock_jnp.sum = MagicMock(return_value="mock_sum")
    monkeypatch.setattr(mod, "jnp", mock_jnp)

    model = MagicMock()
    model.apply.return_value = MagicMock()

    res = mod._compute_logps(model, MagicMock(), MagicMock(), MagicMock())
    assert res == "mock_sum"


def test_maxtext_dpo_wrapper(monkeypatch):
    """Docstring for test_maxtext_dpo_wrapper."""
    import gemma_4_sql.backends.maxtext.dpo as mod

    mock_jax = MagicMock()

    def mock_vag(fn):
        """Docstring for mock_vag."""

        def inner(*args, **kwargs):
            # args[0] = policy_params, args[1] = ref_params, args[2] = batch
            """Docstring for inner."""
            res = fn(args[0], args[1], args[2])
            return (res, "grads")

        return inner

    mock_jax.value_and_grad = mock_vag

    # We must also patch jit, so that when it jits our return_fn, it just returns it
    mock_jax.jit = lambda fn: fn
    monkeypatch.setattr(mod, "jax", mock_jax)

    monkeypatch.setattr(mod, "_dpo_step_loss", MagicMock(return_value="mock_loss"))

    mock_opt = MagicMock()
    mock_opt.update.return_value = ("updates", "state")

    mock_optax = MagicMock()
    mock_optax.apply_updates.return_value = "new_params"
    monkeypatch.setattr(mod, "optax", mock_optax)

    step_fn = mod._get_train_step_fn(MagicMock(), MagicMock(), mock_opt, 0.1)

    res = step_fn("pol", "opt", "ref", "batch")
    assert res[0] == "new_params"
    assert res[2] == "mock_loss"
