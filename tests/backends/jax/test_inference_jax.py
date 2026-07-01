"""Tests for JAX inference logic."""

import pytest

import gemma_4_sql.backends.jax.inference as inf
from gemma_4_sql.backends.jax.inference import generate_sql, jax_beam_search


class MockArray:
    """Mock JAX Array."""

    def __init__(self: object, data: object) -> None:
        """Initialize function __init__.

        Args:
        ----
        data: Description of data.

        """
        self.data = data if isinstance(data, list) else [data]

    @property
    def shape(self: object) -> object:
        """Initialize function shape."""
        if isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))
        return (len(self.data),)

    def __getitem__(self: object, idx: object) -> object:
        """Magic method docstring."""
        if isinstance(idx, MockArray):
            return MockArray([self.data[i] for i in getattr(idx, "data", [])])
        try:
            expected_len = 2
            if isinstance(idx, tuple) and len(idx) == expected_len:
                return self.data[idx[0]][idx[1]]
            return MockArray(self.data[idx])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError):
            return MockArray(self.data)

    def tolist(self: object) -> object:
        """Initialize function tolist."""
        return self.data

    def item(self: object) -> object:
        """Initialize function item."""
        return self.data[0] if isinstance(self.data, list) else self.data

    def reshape(self: object, *shape: object) -> object:
        """Initialize function reshape.

        Args:
        ----
        shape: Description of shape.

        """
        if shape == (1, 1):
            val = self.data[0] if isinstance(self.data, list) else self.data
            return MockArray([[val]])
        return self


class MockJNP:
    """Mock JNP."""

    def arange(self: object, val: object) -> object:
        """Initialize function arange.

        Args:
        ----
        val: Description of val.

        """
        return MockArray([0] * val)

    def array(self: object, data: object, _dtype: object = None, **_kwargs: object) -> object:
        """Initialize function array.

        Args:
        ----
        data: Description of data.
        dtype: Description of dtype.

        """
        return MockArray(data)

    int32 = 1

    def concatenate(self: object, arrays: object, axis: object = 0) -> object:
        """Initialize function concatenate.

        Args:
        ----
        arrays: Description of arrays.
        axis: Description of axis.

        """
        if axis == -1:
            res = [arrays[0].data[i] + arrays[1].data[i] for i in range(len(arrays[0].data))]  # type: ignore[index]
            return MockArray(res)
        return MockArray([a.data for a in arrays])  # type: ignore[attr-defined]

    def argsort(self: object, array: object) -> object:
        """Initialize function argsort.

        Args:
        ----
        array: Description of array.

        """
        d = array.data  # type: ignore[attr-defined]
        return MockArray(sorted(range(len(d)), key=lambda x: d[x]))


class MockNN:
    """Initialize class MockNN."""

    def log_softmax(self: object, x: object, _axis: object = -1, **_kwargs: object) -> object:
        """Initialize function log_softmax.

        Args:
        ----
        x: Description of x.
        axis: Description of axis.

        """
        return x


class MockJAX:
    """Mock JAX."""

    nn = MockNN()


class MockGemma4Config:
    """Initialize class MockGemma4Config."""

    @staticmethod
    def gemma4_e2b() -> object:
        """Initialize function gemma4_e2b."""
        return "mock_config"


class MockGemma4ForCausalLM:
    """Initialize class MockGemma4ForCausalLM."""

    def __init__(self: object, config: object, _rngs: object = None, **_kwargs: object) -> None:
        """Initialize function __init__.

        Args:
        ----
        config: Description of config.
        rngs: Description of rngs.

        """
        self.config = config

    def __call__(self: object, _seq: object, _positions: object = None) -> object:
        """Initialize function __call__.

        Args:
        ----
        seq: Description of seq.
        positions: Description of positions.

        """
        logits = [0.0] * 300
        logits[100] = 10.0
        return MockArray([logits])


class MockNNX:
    """Initialize class MockNNX."""

    class Rngs:
        """Initialize class Rngs."""

        def __init__(self: object, seed: object) -> None:
            """Initialize function __init__.

            Args:
            ----
            seed: Description of seed.

            """
            self.seed = seed


@pytest.fixture
def _mock_jax_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function mock_jax_env.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(inf, "jax", MockJAX())
    monkeypatch.setattr(inf, "jnp", MockJNP())
    monkeypatch.setattr(inf, "Gemma4ForCausalLM", MockGemma4ForCausalLM)
    monkeypatch.setattr(inf, "Gemma4Config", MockGemma4Config)
    monkeypatch.setattr(inf, "nnx", MockNNX())


@pytest.mark.usefixtures("_mock_jax_env")
def test_generate_sql_success() -> None:
    """Initialize function test_generate_sql_success.

    Args:
    ----
    mock_jax_env: Description of mock_jax_env.

    """
    res = generate_sql("mock-model", "test prompt", beam_width=2, max_length=3)
    if not res["status"] == "success":
        raise AssertionError
    if not res["backend"] == "jax":
        raise AssertionError
    if not isinstance(res["sql"], str):
        raise TypeError


def test_generate_sql_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function test_generate_sql_missing_deps.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(inf, "jax", None)
    res = generate_sql("mock-model", "test prompt")
    if not res["status"] == "mocked_missing_jax":
        raise AssertionError


@pytest.mark.usefixtures("_mock_jax_env")
def test_jax_beam_search() -> None:
    """Initialize function test_jax_beam_search.

    Args:
    ----
    mock_jax_env: Description of mock_jax_env.

    """
    jnp_mock = MockJNP()

    def mock_apply_fn(seq: MockArray, _positions: object = None) -> MockArray:
        """Initialize function mock_apply_fn.

        Args:
        ----
        seq: Description of seq.
        positions: Description of positions.

        """
        logits = [0.0] * 300
        seq_len = len(seq.data[0]) if isinstance(seq.data[0], list) else len(seq.data)
        if seq_len == 1:
            logits[5] = 10.0
        else:
            logits[299] = 10.0
        return MockArray([logits])

    input_ids = jnp_mock.array([[1]])
    result = jax_beam_search(model_apply_fn=mock_apply_fn, input_ids=input_ids, beam_width=2, max_length=5, eos_token_id=299)
    if not result.tolist() == [[1, 5, 299]]:
        raise AssertionError


def test_inference_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    import gemma_4_sql.backends.jax.inference as mdl

    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(mdl)

    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "flax", None)
    importlib.reload(mdl)

    monkeypatch.undo()
    importlib.reload(mdl)
