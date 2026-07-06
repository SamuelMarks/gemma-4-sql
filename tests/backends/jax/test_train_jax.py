"""Tests for JAX training pipeline."""

import pytest

import gemma_4_sql.backends.jax.train as tr
from gemma_4_sql.backends.jax.train import train_model
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
        kwargs: Description of kwargs.


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
    def prngkey(seed: object) -> object:
        """Initialize function prngkey.

        Args:
        ----
        seed: Description of seed.


        Returns:
            object: Description of return.

        """
        return seed


class MockJaxSharding:
    """Initialize class MockJaxSharding."""

    class Mesh:
        """Initialize class Mesh."""

        def __init__(self, devices: object, axis_names: object) -> None:
            """Execute function."""
            self.devices = devices
            self.axis_names = axis_names

    class NamedSharding:
        """Initialize class NamedSharding."""

        def __init__(self, mesh: object, spec: object) -> None:
            """Execute function."""
            self.mesh = mesh
            self.spec = spec

    class PartitionSpec:
        """Initialize class PartitionSpec."""

        def __init__(self, *args: object) -> None:
            """Execute function."""
            self.args = args


class MockJax:
    """Initialize class MockJax."""

    random = MockJaxRandom()
    sharding = MockJaxSharding()

    @staticmethod
    def devices() -> list[str]:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return ["cpu"]

    @staticmethod
    def device_put(x: object, _sharding: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return x

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
    def warmup_cosine_decay_schedule(**_kwargs: object) -> object:
        """Initialize function warmup_cosine_decay_schedule.

        Returns:
            object: Description of return.

        """
        return "schedule"

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


class MockGemma4Config:
    """Initialize class MockGemma4Config."""

    @staticmethod
    def gemma4_e2b() -> object:
        """Initialize function gemma4_e2b.

        Returns:
            object: Description of return.

        """
        return "mock_config"


class MockGemma4ForCausalLM:
    """Initialize class MockGemma4ForCausalLM."""

    def __init__(self, config: object, **_kwargs: object) -> None:
        """Initialize function __init__.

        Args:
        ----
        config: Description of config.
        _kwargs: Description of kwargs.

        """
        self.config = config

    def __call__(self, _inputs: object) -> object:
        """Initialize function __call__.

        Returns:
            object: Description of return.

        """
        return MockJnpTensor((1,))


class MockNNXOptimizer:
    """Initialize class MockNNXOptimizer."""

    def __init__(self, model: object, optax_optimizer: object) -> None:
        """Initialize function __init__.

        Args:
        ----
        model: Description of model.
        optax_optimizer: Description of optax_optimizer.

        """
        self.model = model
        self.optax_optimizer = optax_optimizer

    def update(self, grads: object) -> object:
        """Initialize function update.

        Args:
        ----
        grads: Description of grads.

        """


class MockNNX:
    """Initialize class MockNNX."""

    class Rngs:
        """Initialize class Rngs."""

        def __init__(self, seed: object) -> None:
            """Initialize function __init__.

            Args:
            ----
            seed: Description of seed.

            """
            self.seed = seed

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

    Optimizer = MockNNXOptimizer


@pytest.fixture
def _mock_jax_env(monkeypatch: object) -> object:
    """Initialize function mock_jax_env.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(tr, "jax", MockJax())
    monkeypatch.setattr(tr, "jnp", MockJnp())
    monkeypatch.setattr(tr, "optax", MockOptax())
    monkeypatch.setattr(tr, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(tr, "Gemma4Config", MockGemma4Config)
    monkeypatch.setattr(tr, "nnx", MockNNX())

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


@pytest.mark.usefixtures("_mock_jax_env")
def test_train_model_jax_real() -> object:
    """Initialize function test_train_model_jax_real.

    Raises:
        AssertionError: Description.

    """
    res = train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1))
    if not res["backend"] == "jax":
        raise AssertionError


def test_train_model_jax_missing() -> object:
    """Initialize function test_train_model_jax_missing.

    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    orig_jax = tr.jax
    tr.jax = None
    with pytest.raises(DependencyMissingError, match="JAX dependencies are missing for training."):
        train_model(TrainingConfig(action="sft", model_name="mod", dataset="dat", epochs=2, learning_rate=0.1))
    tr.jax = orig_jax


@pytest.mark.usefixtures("_mock_jax_env")
def test_train_model_jax_error(monkeypatch: object) -> object:
    """Initialize function test_train_model_jax_error.

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


@pytest.mark.usefixtures("_mock_jax_env")
def test_train_model_jax_no_loader_fallback(monkeypatch: object) -> object:
    """Initialize function test_train_model_jax_no_loader_fallback.

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
    mdl = __import__("gemma_4_sql.backends.jax.train", fromlist=[""])
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "flax", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    importlib.reload(mdl)
