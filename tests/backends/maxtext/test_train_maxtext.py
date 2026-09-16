"""Tests for MaxText training pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
from typing_extensions import Self

import gemma_4_sql.backends.maxtext.train as tr
from gemma_4_sql.backends.maxtext.train import train_model
from gemma_4_sql.type_hints import TrainingConfig


class MockJnpTensor:
    """Initialize class MockJnpTensor."""

    def __init__(self, shape: object) -> None:
        """Initialize function __init__.

        Args:
        ----
        shape: Description of shape.

        """
        self.shape = shape

    def item(self) -> object:
        """Initialize function item.

        Returns:
            object: Description of return.

        """
        return 0.35


class MockJnp:
    """Initialize class MockJnp."""

    int32 = 1

    @staticmethod
    def zeros(shape: object, **_kwargs: object) -> object:
        """Initialize function zeros.

        Args:
        ----
        shape: Description of shape.
        dtype: Description of dtype.
        **kwargs: Description of kwargs.


        Returns:
            object: Description of return.

        """
        return MockJnpTensor(shape)

    @staticmethod
    def mean(x: object) -> object:
        """Initialize function mean.

        Args:
        ----
        x: Description of x.

        """


class MockJaxRandom:
    """Initialize class MockJaxRandom."""

    @staticmethod
    def mock_prngkey(seed: object) -> object:
        """Initialize function prngkey.

        Args:
        ----
        seed: Description of seed.


        Returns:
            object: Description of return.

        """
        return seed

    PRNGKey = mock_prngkey


class MockJax:
    """Initialize class MockJax."""

    random = MockJaxRandom()

    @staticmethod
    def jit(fn: object) -> object:
        """Initialize function jit.

        Args:
        ----
        fn: Description of fn.


        Returns:
            object: Description of return.

        """
        return fn

    @staticmethod
    def value_and_grad(fn: object) -> object:
        """Initialize function value_and_grad.

        Args:
        ----
        fn: Description of fn.


        Returns:
            object: Description of return.

        """

        def wrapper(*args: object, **kwargs: object) -> object:
            """Initialize function wrapper.

            Args:
            ----
            args: Description of args.
            kwargs: Description of kwargs.


            Returns:
                object: Description of return.

            """
            _ = fn(*args, **kwargs)
            return (MockJnpTensor((1,)), "grads")

        return wrapper


class MockOptax:
    """Initialize class MockOptax."""

    @staticmethod
    def adamw(_lr: object) -> object:
        """Initialize function adamw.

        Returns:
            object: Description of return.

        """

        class MockOpt:
            """Initialize class MockOpt."""

            def init(self, _params: object) -> object:
                """Initialize function init.

                Returns:
                    object: Description of return.

                """
                return "opt_state"

            def update(self, _grads: object, _opt_state: object, _params: object) -> object:
                """Initialize function update.

                Returns:
                    object: Description of return.

                """
                return ("updates", "opt_state")

        return MockOpt()

    @staticmethod
    def softmax_cross_entropy_with_integer_labels(_logits: object, _labels: object) -> object:
        """Initialize function softmax_cross_entropy_with_integer_labels.

        Returns:
            object: Description of return.

        """
        return MockJnpTensor((1,))

    @staticmethod
    def apply_updates(params: object, _updates: object) -> object:
        """Initialize function apply_updates.

        Args:
        ----
        params: Description of params.


        Returns:
            object: Description of return.

        """
        return params


class MockGemma4Model:
    """Initialize class MockGemma4Model."""

    def __init__(self, name: object) -> None:
        """Initialize function __init__.

        Args:
        ----
        name: Description of name.

        """

    def init(self, _rng: object, _inputs: object) -> object:
        """Initialize function init.

        Returns:
            object: Description of return.

        """
        return "params"

    def apply(self, _params: object, _inputs: object) -> object:
        """Initialize function apply.

        Returns:
            object: Description of return.

        """
        return MockJnpTensor((1,))


@pytest.fixture
def _mock_maxtext_env(monkeypatch: object) -> object:
    """Initialize function mock_maxtext_env.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(tr, "jax", MockJax())
    monkeypatch.setattr(tr, "jnp", MockJnp())
    monkeypatch.setattr(tr, "optax", MockOptax())
    monkeypatch.setattr(tr, "Gemma4Model", MockGemma4Model)

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Initialize function mock_build_dataloader.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.


        Returns:
            object: Description of return.

        """
        return {"loader": [{"inputs": MockJnpTensor((1,)), "targets": MockJnpTensor((1,))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_train_model_maxtext_real() -> object:
    """Initialize function test_train_model_maxtext_real.

    Raises:
        AssertionError: Description.

    """
    res = train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1))
    if not res["backend"] == "maxtext":
        raise AssertionError


def test_train_model_maxtext_missing() -> object:
    """Initialize function test_train_model_maxtext_missing.

    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    orig_jax = tr.jax
    tr.jax = None
    with pytest.raises(DependencyMissingError):
        train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1))
    tr.jax = orig_jax


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_train_model_maxtext_error(monkeypatch: object) -> object:
    """Initialize function test_train_model_maxtext_error.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """

    def mock_raise_error(*_args: object, **_kwargs: object) -> object:
        """Initialize function Exception.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.


        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(tr, "build_dataloader", Exception)
    train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1))


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_train_model_maxtext_no_loader_fallback(monkeypatch: object) -> object:
    """Initialize function test_train_model_maxtext_no_loader_fallback.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Initialize function mock_build_dataloader.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.


        Returns:
            object: Description of return.

        """
        return {"loader": None}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)
    train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1))


def test_train_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    m_train = __import__("gemma_4_sql.backends.maxtext.train", fromlist=[""])
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(m_train)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "maxtext.train", None)
    importlib.reload(m_train)
    monkeypatch.undo()
    importlib.reload(m_train)


class MockMaxTextTrain:
    """Provide class docstring."""

    @staticmethod
    def main(*args: object, **kwargs: object) -> None:
        """Execute function."""


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_train_model_maxtext_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    m_train = __import__("gemma_4_sql.backends.maxtext.train", fromlist=[""])
    monkeypatch.setattr(m_train, "maxtext_train", MockMaxTextTrain())
    res = m_train.train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1, extra_kwargs={"test_mode": False}))
    if res["status"] != "completed":
        raise AssertionError


def test_train_imports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful import of all MaxText training dependencies."""
    import importlib
    import sys
    import types

    import gemma_4_sql.backends.maxtext.train as m_train

    mock_mod = types.ModuleType("maxtext")
    mock_train = types.ModuleType("maxtext.train")
    mock_models = types.ModuleType("maxtext.models")
    mock_gemma4 = types.ModuleType("maxtext.models.gemma4")
    mock_gemma4.Gemma4Model = type("MockModel", (), {})  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "maxtext", mock_mod)
    monkeypatch.setitem(sys.modules, "maxtext.train", mock_train)
    monkeypatch.setitem(sys.modules, "maxtext.models", mock_models)
    monkeypatch.setitem(sys.modules, "maxtext.models.gemma4", mock_gemma4)

    importlib.reload(m_train)
    assert m_train.Gemma4Model is not None
    assert m_train.maxtext_train is not None

    monkeypatch.undo()
    importlib.reload(m_train)


def test_maxtext_train_step_nojit_and_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _get_train_step_fn without jax.jit and _execute_train missing dependencies."""
    import gemma_4_sql.backends.maxtext.train as m_train
    from gemma_4_sql.exceptions import DependencyMissingError

    # Test without jax.jit
    class MockNoJitJax:
        """Test class for MockNoJitJax."""

    monkeypatch.setattr(m_train, "jax", MockNoJitJax())
    step_fn = m_train._get_train_step_fn(None, None)
    assert callable(step_fn)

    # Test _execute_train missing dependencies
    monkeypatch.setattr(m_train, "jax", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing for training"):
        m_train._execute_train("mod", "ds", 1, 1e-4, False)


def test_initialize_jax_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX distributed coordination service initialization."""
    # test_mode returns False
    assert not tr._initialize_jax_distributed(test_mode=True)

    # jax is None returns False
    monkeypatch.setattr(tr, "jax", None)
    assert not tr._initialize_jax_distributed(test_mode=False)

    # jax has distributed and initialize succeeds
    calls: list[dict[str, object]] = []

    class MockDistributed:
        """Mock JAX distributed module."""

        @staticmethod
        def initialize(**kwargs: object) -> None:
            """Execute initialize."""
            calls.append(kwargs)

    class MockJaxDist:
        """Mock JAX module with distributed support."""

        distributed = MockDistributed()

    monkeypatch.setattr(tr, "jax", MockJaxDist())
    res = tr._initialize_jax_distributed(
        coordinator_address="10.0.0.1:1234",
        num_processes=4,
        process_id=1,
        test_mode=False,
    )
    assert res is True
    assert len(calls) == 1
    assert calls[0] == {"coordinator_address": "10.0.0.1:1234", "num_processes": 4, "process_id": 1}

    # error during initialize logs warning and returns False
    class MockFailingDistributed:
        """Mock failing distributed module."""

        @staticmethod
        def initialize(**kwargs: object) -> None:
            """Raise exception on initialize."""
            raise RuntimeError("Already initialized")

    class MockFailingJaxDist:
        """Mock JAX module with failing distributed initialization."""

        distributed = MockFailingDistributed()

    monkeypatch.setattr(tr, "jax", MockFailingJaxDist())
    assert tr._initialize_jax_distributed(test_mode=False) is False


def test_save_maxtext_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test saving checkpoint using Orbax CheckpointManager."""
    from gemma_4_sql.exceptions import DependencyMissingError, ExportError

    # Missing ocp
    monkeypatch.setattr(tr, "ocp", None)
    with pytest.raises(DependencyMissingError, match="Orbax checkpoint dependency"):
        tr.save_maxtext_checkpoint(tmp_path, 1, {"weights": 1})

    # Successful save
    saved: list[tuple[int, object]] = []

    class MockCheckpointManager:
        """Mock Orbax CheckpointManager."""

        def __init__(self, directory: object, checkpointer: object, options: object) -> None:
            """Initialize MockCheckpointManager."""

        def __enter__(self) -> Self:
            """Enter context."""
            return self

        def __exit__(self, *args: object) -> None:
            """Exit context."""

        def save(self, step: int, item: object) -> None:
            """Save item."""
            saved.append((step, item))

    class MockOcp:
        """Mock Orbax checkpoint module."""

        CheckpointManagerOptions = staticmethod(lambda **kwargs: kwargs)
        PyTreeCheckpointer = staticmethod(lambda: "pytree")
        CheckpointManager = MockCheckpointManager

    monkeypatch.setattr(tr, "ocp", MockOcp())
    saved_path = tr.save_maxtext_checkpoint(
        tmp_path / "ckpt",
        step=10,
        params={"p": 1},
        opt_state={"opt": 2},
    )
    assert saved_path == (tmp_path / "ckpt").resolve()
    assert len(saved) == 1
    assert saved[0][0] == 10
    assert saved[0][1] == {"params": {"p": 1}, "opt_state": {"opt": 2}}

    # Failing save raises ExportError
    class MockFailingCheckpointManager:
        """Mock failing Orbax CheckpointManager."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize MockFailingCheckpointManager."""

        def __enter__(self) -> Self:
            """Enter context."""
            return self

        def __exit__(self, *args: object) -> None:
            """Exit context."""

        def save(self, step: int, item: object) -> None:
            """Raise exception on save."""
            raise OSError("disk full")

    MockOcp.CheckpointManager = MockFailingCheckpointManager
    with pytest.raises(ExportError, match="Failed to persist Orbax checkpoint"):
        tr.save_maxtext_checkpoint(tmp_path / "fail_ckpt", 1, {"p": 1})


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_execute_train_cluster_distributed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test multi-host cluster execution invoking maxtext_train.main with Gin config and Orbax checkpoint."""
    invoked_cli: list[list[str]] = []
    saved_ckpts: list[tuple[int, object]] = []

    class MockClusterTrain:
        """Mock MaxText cluster training entrypoint."""

        @staticmethod
        def main(args: list[str]) -> None:
            """Capture CLI args."""
            invoked_cli.append(args)

    class MockCheckpointManager:
        """Mock CheckpointManager."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize."""

        def __enter__(self) -> Self:
            """Enter context."""
            return self

        def __exit__(self, *args: object) -> None:
            """Exit context."""

        def save(self, step: int, item: object) -> None:
            """Save item."""
            saved_ckpts.append((step, item))

    class MockOcp:
        """Mock Orbax."""

        CheckpointManagerOptions = staticmethod(lambda **kwargs: kwargs)
        PyTreeCheckpointer = staticmethod(lambda: "pytree")
        CheckpointManager = MockCheckpointManager

    monkeypatch.setattr(tr, "maxtext_train", MockClusterTrain())
    monkeypatch.setattr(tr, "ocp", MockOcp())

    cfg = TrainingConfig(
        model_name="gemma-4-7b",
        dataset="spider",
        epochs=3,
        batch_size=4,
        extra_kwargs={
            "coordinator_address": "127.0.0.1:8080",
            "num_processes": 8,
            "process_id": 0,
            "checkpoint_dir": str(tmp_path / "cluster_ckpt"),
        },
    )

    status, loss = tr._execute_train(cfg, test_mode=False, local_step_mode=False)
    assert status == "completed"
    assert loss == 0.0
    assert len(invoked_cli) == 1
    assert invoked_cli[0][0] == "train.py"
    assert Path(invoked_cli[0][1]).is_file()
    assert len(saved_ckpts) == 1
    assert saved_ckpts[0][0] == 3


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_execute_train_local_step_mode_with_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test explicit local_step_mode executing lightweight loop and persisting Orbax checkpoint."""
    saved_ckpts: list[tuple[int, object]] = []

    class MockCheckpointManager:
        """Mock CheckpointManager."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize."""

        def __enter__(self) -> Self:
            """Enter context."""
            return self

        def __exit__(self, *args: object) -> None:
            """Exit context."""

        def save(self, step: int, item: object) -> None:
            """Save item."""
            saved_ckpts.append((step, item))

    class MockOcp:
        """Mock Orbax."""

        CheckpointManagerOptions = staticmethod(lambda **kwargs: kwargs)
        PyTreeCheckpointer = staticmethod(lambda: "pytree")
        CheckpointManager = MockCheckpointManager

    monkeypatch.setattr(tr, "ocp", MockOcp())

    status, loss = tr._execute_train(
        "gemma-4",
        dataset="dat",
        epochs=2,
        local_step_mode=True,
        checkpoint_dir=str(tmp_path / "local_ckpt"),
    )
    assert status == "completed"
    assert loss == 0.35
    assert len(saved_ckpts) == 1
    assert saved_ckpts[0][0] == 2


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_execute_train_missing_gemma4_model_in_local_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test DependencyMissingError when Gemma4Model is None in local step loop."""
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(tr, "Gemma4Model", None)
    monkeypatch.setattr(tr, "maxtext_train", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing for training"):
        tr._execute_train("gemma-4", local_step_mode=True)


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_train_model_returns_checkpoint_dir(tmp_path: Path) -> None:
    """Test that train_model includes checkpoint_dir in return dict when configured."""
    cfg = TrainingConfig(
        model_name="gemma-4",
        dataset="ds",
        epochs=1,
        extra_kwargs={"checkpoint_dir": str(tmp_path / "ckpt_dir"), "test_mode": True},
    )
    res = tr.train_model(cfg)
    assert res["status"] == "completed"
    assert res["checkpoint_dir"] == str(tmp_path / "ckpt_dir")


def test_train_model_missing_gemma_and_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test DependencyMissingError when both Gemma4Model and maxtext_train are None."""
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(tr, "Gemma4Model", None)
    monkeypatch.setattr(tr, "maxtext_train", None)
    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        tr.train_model(TrainingConfig())
