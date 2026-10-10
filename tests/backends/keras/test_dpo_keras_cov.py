"""Module docstring."""

from unittest.mock import MagicMock

import pytest


def test_run_dpo_missing_keras(monkeypatch):
    """Docstring for test_run_dpo_missing_keras."""
    from gemma_4_sql.backends.keras import dpo
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(dpo, "keras", None)
    config = MagicMock()
    with pytest.raises(DependencyMissingError):
        dpo.run_dpo(config)


def test_run_dpo_missing_tf(monkeypatch):
    """Docstring for test_run_dpo_missing_tf."""
    from gemma_4_sql.backends.keras import dpo
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(dpo, "keras", MagicMock())
    monkeypatch.setattr(dpo, "tf", None)
    config = MagicMock()
    with pytest.raises(DependencyMissingError):
        dpo.run_dpo(config)


def test_dpo_loss(monkeypatch):
    """Docstring for test_dpo_loss."""
    from gemma_4_sql.backends.keras import dpo

    monkeypatch.setattr(dpo, "tf", None)
    assert dpo.dpo_loss(None, None, None, None) == (0.0, 0.0, 0.0)

    mock_tf = MagicMock()
    monkeypatch.setattr(dpo, "tf", mock_tf)
    monkeypatch.setattr("gemma_4_sql.backends.keras.dpo.generic_dpo_loss", MagicMock(return_value="loss"))
    assert dpo.dpo_loss(None, None, None, None) == "loss"


def test_compute_logps(monkeypatch):
    """Docstring for test_compute_logps."""
    from gemma_4_sql.backends.keras import dpo

    monkeypatch.setattr(dpo, "tf", None)
    assert dpo._compute_logps(None, None, None) == 0.0

    mock_tf = MagicMock()
    monkeypatch.setattr(dpo, "tf", mock_tf)
    model = MagicMock()
    res = dpo._compute_logps(model, MagicMock(), MagicMock())
    assert res is not None


def test_compute_logps_with_numpy(monkeypatch):
    """Docstring for test_compute_logps_with_numpy."""
    from gemma_4_sql.backends.keras import dpo

    mock_tf = MagicMock()

    class FakeTensor:
        """Docstring for FakeTensor."""

        def __init__(self, val):
            """Docstring for __init__."""
            self.val = val

        def __eq__(self, other):
            """Docstring for __eq__."""
            return self.val == other

        def numpy(self):
            """Docstring for numpy."""
            return self.val

        def __mul__(self, other):
            """Docstring for __mul__."""
            return FakeTensor(self.val * other)

    mock_tf.reduce_sum = lambda x, axis: FakeTensor(-1.0) if x == "case1" else FakeTensor(-1.5)
    mock_tf.cast = lambda x, y: x
    mock_tf.expand_dims = lambda x, axis: x
    mock_tf.gather = lambda x, y, batch_dims: x
    mock_tf.squeeze = lambda x, axis: x

    class FakeNN:
        """Docstring for FakeNN."""

    mock_tf.nn = FakeNN()
    mock_tf.nn.log_softmax = lambda x, axis: "case1"

    monkeypatch.setattr(dpo, "tf", mock_tf)

    model = MagicMock(return_value=1.0)

    res = dpo._compute_logps(model, MagicMock(), MagicMock())
    assert res == -1.0

    mock_tf.nn.log_softmax = lambda x, axis: "case2"
    assert dpo._compute_logps(model, MagicMock(), MagicMock()) == -1.5


def test_get_train_step_fn(monkeypatch):
    """Docstring for test_get_train_step_fn."""
    from gemma_4_sql.backends.keras import dpo

    monkeypatch.setattr(dpo, "tf", None)
    fn = dpo._get_train_step_fn(None, None, None, 0.1)
    assert fn(None) == 0.0

    mock_tf = MagicMock()

    class MockTape:
        """Docstring for MockTape."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

        def __enter__(self):
            """Docstring for __enter__."""
            return MagicMock()

        def __exit__(self, *args, **kwargs):
            """Docstring for __exit__."""

    mock_tf.GradientTape = MockTape
    mock_tf.function = lambda x: x
    monkeypatch.setattr(dpo, "tf", mock_tf)

    monkeypatch.setattr(dpo, "_compute_logps", MagicMock(return_value=1.0))
    monkeypatch.setattr(dpo, "dpo_loss", MagicMock(return_value=(1.0, 0, 0)))

    fn2 = dpo._get_train_step_fn(MagicMock(), MagicMock(), MagicMock(), 0.1)

    batch = {"chosen_inputs": 1, "chosen_labels": 1, "rejected_inputs": 1, "rejected_labels": 1}
    assert fn2(batch) == 1.0


def test_run_training_epochs():
    """Docstring for test_run_training_epochs."""
    from gemma_4_sql.backends.keras.dpo import _run_training_epochs
    from gemma_4_sql.type_hints import TrainerState

    def mock_train_step(b):
        """Docstring for mock_train_step."""
        m = MagicMock()
        m.numpy.return_value = 1.0
        return m

    state = TrainerState(dataloader=[1, 2], epochs=1, train_step=mock_train_step)
    assert _run_training_epochs(state) == 1.0

    def mock_train_step_no_numpy(b):
        """Docstring for mock_train_step_no_numpy."""
        return 2.0

    state2 = TrainerState(dataloader=[1, 2], epochs=1, train_step=mock_train_step_no_numpy)
    assert _run_training_epochs(state2) == 2.0

    def mock_train_step_err(b):
        """Docstring for mock_train_step_err."""
        return "err"

    state3 = TrainerState(dataloader=[1, 2], epochs=1, train_step=mock_train_step_err)
    assert _run_training_epochs(state3) == 0.0


def test_execute_dpo_missing(monkeypatch):
    """Docstring for test_execute_dpo_missing."""
    from gemma_4_sql.backends.keras import dpo
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(dpo, "keras", None)
    with pytest.raises(DependencyMissingError):
        dpo._execute_dpo("m", "d", 0.1, 1, 0.1)


def test_execute_dpo_error(monkeypatch):
    """Docstring for test_execute_dpo_error."""
    from gemma_4_sql.backends.keras import dpo

    monkeypatch.setattr(dpo, "keras", MagicMock())
    monkeypatch.setattr(dpo, "tf", MagicMock())

    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "keras_nlp.models":
            raise ValueError("simulated error")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ValueError):
        dpo._execute_dpo("m", "d", 0.1, 1, 0.1)


def test_execute_dpo(monkeypatch):
    """Docstring for test_execute_dpo."""
    from gemma_4_sql.backends.keras import dpo

    monkeypatch.setattr(dpo, "keras", MagicMock())
    monkeypatch.setattr(dpo, "tf", MagicMock())

    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "keras_nlp.models":
            mock_models = MagicMock()
            return mock_models
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    monkeypatch.setattr(dpo, "_get_train_step_fn", MagicMock())

    # bad dataloader
    monkeypatch.setattr(dpo, "build_dataloader", MagicMock(return_value={}))
    with pytest.raises(ValueError):
        dpo._execute_dpo("m", "d", 0.1, 1, 0.1)

    # good dataloader
    monkeypatch.setattr(dpo, "build_dataloader", MagicMock(return_value={"loader": [1, 2]}))
    monkeypatch.setattr(dpo, "_run_training_epochs", MagicMock(return_value=1.0))
    status, loss = dpo._execute_dpo("m", "d", 0.1, 1, 0.1)
    assert status == "completed"
    assert loss == 1.0


def test_run_dpo(monkeypatch):
    """Docstring for test_run_dpo."""
    from gemma_4_sql.backends.keras import dpo

    monkeypatch.setattr(dpo, "keras", MagicMock())
    monkeypatch.setattr(dpo, "tf", MagicMock())

    # error
    monkeypatch.setattr(dpo, "_execute_dpo", MagicMock(side_effect=ValueError("simulated")))
    config = MagicMock()
    res = dpo.run_dpo(config)
    assert "failed: simulated" in res["status"]

    # success
    monkeypatch.setattr(dpo, "_execute_dpo", MagicMock(return_value=("completed", 1.0)))
    res2 = dpo.run_dpo(config)
    assert res2["status"] == "completed"
    assert res2["final_loss"] == 1.0


def test_compute_logps_tf_none(monkeypatch):
    """Docstring for test_compute_logps_tf_none."""
    from gemma_4_sql.backends.keras import dpo

    monkeypatch.setattr(dpo, "tf", None)
    assert dpo._compute_logps(None, None, None) == 0.0


def test_get_train_step_fn_tf_none_more(monkeypatch):
    """Docstring for test_get_train_step_fn_tf_none_more."""
