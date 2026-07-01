"""Tests for JAX training pipeline."""

import pytest

import gemma_4_sql.backends.jax.train as tr
from gemma_4_sql.backends.jax.train import train_model


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
        """Initialize function item."""
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

        """
        return seed


class MockJaxSharding:
    """Initialize class MockJaxSharding."""

    class Mesh:
        """Initialize class Mesh."""

        def __init__(self, devices: object, axis_names: object) -> None:
            self.devices = devices
            self.axis_names = axis_names

    class NamedSharding:
        """Initialize class NamedSharding."""

        def __init__(self, mesh: object, spec: object) -> None:
            self.mesh = mesh
            self.spec = spec

    class PartitionSpec:
        """Initialize class PartitionSpec."""

        def __init__(self, *args: object) -> None:
            self.args = args


class MockJax:
    """Initialize class MockJax."""

    random = MockJaxRandom()
    sharding = MockJaxSharding()

    @staticmethod
    def devices() -> list[str]:
        return ["cpu"]

    @staticmethod
    def device_put(x: object, _sharding: object) -> object:
        return x

    @staticmethod
    def jit(fn: object) -> object:
        """Initialize function jit.

        Args:
        ----
        fn: Description of fn.

        """
        return fn

    @staticmethod
    def value_and_grad(fn: object) -> object:
        """Initialize function value_and_grad.

        Args:
        ----
        fn: Description of fn.

        """

        def wrapper(*args: object, **kwargs: object) -> object:
            """Initialize function wrapper.

            Args:
            ----
            args: Description of args.
            kwargs: Description of kwargs.

            """
            _ = fn(*args, **kwargs)  # type: ignore[operator]
            return (MockJnpTensor((1,)), "grads")

        return wrapper


class MockOptax:
    """Initialize class MockOptax."""

    @staticmethod
    def warmup_cosine_decay_schedule(**_kwargs: object) -> object:
        """Initialize function warmup_cosine_decay_schedule."""
        return "schedule"

    @staticmethod
    def adamw(_lr: object) -> object:
        """Initialize function adamw.

        Args:
        ----
        lr: Description of lr.

        """

        class MockOpt:
            """Initialize class MockOpt."""

            def init(self, _params: object) -> object:
                """Initialize function init.

                Args:
                ----
                params: Description of params.

                """
                return "opt_state"

            def update(self, _grads: object, _opt_state: object, _params: object) -> object:
                """Initialize function update.

                Args:
                ----
                grads: Description of grads.
                opt_state: Description of opt_state.
                params: Description of params.

                """
                return ("updates", "opt_state")

        return MockOpt()

    @staticmethod
    def softmax_cross_entropy_with_integer_labels(_logits: object, _labels: object) -> object:
        """Initialize function softmax_cross_entropy_with_integer_labels.

        Args:
        ----
        logits: Description of logits.
        labels: Description of labels.

        """
        return MockJnpTensor((1,))

    @staticmethod
    def apply_updates(params: object, _updates: object) -> object:
        """Initialize function apply_updates.

        Args:
        ----
        params: Description of params.
        updates: Description of updates.

        """
        return params


class MockGemma4Config:
    """Initialize class MockGemma4Config."""

    @staticmethod
    def gemma4_e2b() -> object:
        """Initialize function gemma4_e2b."""
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

    def __call__(self, inputs: object) -> object:
        """Initialize function __call__.

        Args:
        ----
        inputs: Description of inputs.

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

        """
        return fn

    @staticmethod
    def value_and_grad(fn: object) -> object:
        """Initialize function value_and_grad.

        Args:
        ----
        fn: Description of fn.

        """

        def wrapper(*args: object, **kwargs: object) -> object:
            """Initialize function wrapper.

            Args:
            ----
            args: Description of args.
            kwargs: Description of kwargs.

            """
            _ = fn(*args, **kwargs)  # type: ignore[operator]
            return (MockJnpTensor((1,)), "grads")

        return wrapper

    Optimizer = MockNNXOptimizer


@pytest.fixture
def _mock_jax_env(monkeypatch: object) -> object:  # type: ignore[return]
    """Initialize function mock_jax_env.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(tr, "jax", MockJax())  # type: ignore[attr-defined]
    monkeypatch.setattr(tr, "jnp", MockJnp())  # type: ignore[attr-defined]
    monkeypatch.setattr(tr, "optax", MockOptax())  # type: ignore[attr-defined]
    monkeypatch.setattr(tr, "Gemma4ForCausalLM", MockGemma4ForCausalLM)  # type: ignore[attr-defined]
    monkeypatch.setattr(tr, "Gemma4Config", MockGemma4Config)  # type: ignore[attr-defined]
    monkeypatch.setattr(tr, "nnx", MockNNX())  # type: ignore[attr-defined]

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Initialize function mock_build_dataloader.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        return {"loader": [{"inputs": MockJnpTensor((1,)), "targets": MockJnpTensor((1,))}]}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)  # type: ignore[attr-defined]


@pytest.mark.usefixtures("_mock_jax_env")
def test_train_model_jax_real() -> object:  # type: ignore[return]
    """Initialize function test_train_model_jax_real.

    Args:
    ----
    mock_jax_env: Description of mock_jax_env.

    """
    res = train_model("sft", "mod", "dat", 2, 0.1)
    if not res["backend"] == "jax":
        raise AssertionError


def test_train_model_jax_missing() -> object:  # type: ignore[return]
    """Initialize function test_train_model_jax_missing."""
    orig_jax = tr.jax  # type: ignore[attr-defined]
    tr.jax = None  # type: ignore[attr-defined]
    res = train_model("sft", "mod", "dat", 2, 0.1)
    if not res["status"] == "mocked_missing_jax":
        raise AssertionError
    tr.jax = orig_jax  # type: ignore[attr-defined]


@pytest.mark.usefixtures("_mock_jax_env")
def test_train_model_jax_error(monkeypatch: object) -> object:  # type: ignore[return]
    """Initialize function test_train_model_jax_error.

    Args:
    ----
    mock_jax_env: Description of mock_jax_env.
    monkeypatch: Description of monkeypatch.

    """

    def raise_error(*_args: object, **_kwargs: object) -> object:
        """Initialize function raise_error.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(tr, "build_dataloader", raise_error)  # type: ignore[attr-defined]
    train_model("sft", "mod", "dat", 2, 0.1)


@pytest.mark.usefixtures("_mock_jax_env")
def test_train_model_jax_no_loader_fallback(monkeypatch: object) -> object:  # type: ignore[return]
    """Initialize function test_train_model_jax_no_loader_fallback.

    Args:
    ----
    mock_jax_env: Description of mock_jax_env.
    monkeypatch: Description of monkeypatch.

    """

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Initialize function mock_build_dataloader.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        return {"loader": None}

    monkeypatch.setattr(tr, "build_dataloader", mock_build_dataloader)  # type: ignore[attr-defined]
    train_model("sft", "mod", "dat", 2, 0.1)


def test_train_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    import gemma_4_sql.backends.jax.train as mdl

    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(mdl)

    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "flax", None)
    importlib.reload(mdl)

    monkeypatch.undo()
    importlib.reload(mdl)
