"""Tests for JAX inference logic."""

from unittest.mock import MagicMock

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
        """Magic method docstring.

        Returns:
            object: Description of return.

        """
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
        """Initialize function tolist.

        Returns:
            object: Description of return.

        """
        return self.data

    def item(self: object) -> object:
        """Initialize function item.

        Returns:
            object: Description of return.

        """
        return self.data[0] if isinstance(self.data, list) else self.data

    def reshape(self: object, *shape: object) -> object:
        """Initialize function reshape.

        Args:
        ----
        shape: Description of shape.


        Returns:
            object: Description of return.

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


        Returns:
            object: Description of return.

        """
        return MockArray([0] * val)

    def array(self: object, data: object, _dtype: object = None, **_kwargs: object) -> object:
        """Initialize function array.

        Args:
        ----
        data: Description of data.
        dtype: Description of dtype.


        Returns:
            object: Description of return.

        """
        return MockArray(data)

    int32 = 1

    def concatenate(self: object, arrays: object, axis: object = 0) -> object:
        """Initialize function concatenate.

        Args:
        ----
        arrays: Description of arrays.
        axis: Description of axis.


        Returns:
            object: Description of return.

        """
        if axis == -1:
            res = [arrays[0].data[i] + arrays[1].data[i] for i in range(len(arrays[0].data))]
            return MockArray(res)
        return MockArray([a.data for a in arrays])

    def argsort(self: object, array: object) -> object:
        """Initialize function argsort.

        Args:
        ----
        array: Description of array.


        Returns:
            object: Description of return.

        """
        d = getattr(array, "data", array)
        if isinstance(d, list) and len(d) > 0 and isinstance(d[0], list):
            d = d[0]
        return MockArray(sorted(range(len(d)), key=lambda x: d[x]))


class MockNN:
    """Initialize class MockNN."""

    def log_softmax(self: object, x: object, _axis: object = -1, **_kwargs: object) -> object:
        """Initialize function log_softmax.

        Args:
        ----
        x: Description of x.
        axis: Description of axis.


        Returns:
            object: Description of return.

        """
        if isinstance(x, MockArray):
            return x
        return MockArray(x)


class MockJAX:
    """Mock JAX."""

    nn = MockNN()

    @staticmethod
    def jit(fn: object, *args: object, **kwargs: object) -> object:
        """Mock jit compilation.

        Args:
            fn: Function to compile.
            *args: Positional args.
            **kwargs: Keyword args.

        Returns:
            The input function unmodified.
        """
        return fn


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

        Returns:
            object: Description of return.

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
def test_generate_sql_orbax_checkpoint_restore(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate_sql restores checkpoint when model path exists.

    Args:
        tmp_path: Pytest tmp_path fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    import sys
    import types
    from pathlib import Path

    ckpt_dir = Path(str(tmp_path)) / "orbax_ckpt"
    ckpt_dir.mkdir()

    class MockCheckpointer:
        """Mock checkpointer."""

        def restore(self, path: Path) -> dict:
            """Restore mock weights."""
            return {"weights": 1}

    orbax_mod = types.ModuleType("orbax")
    ocp_mod = types.ModuleType("orbax.checkpoint")
    ocp_mod.PyTreeCheckpointer = MockCheckpointer
    orbax_mod.checkpoint = ocp_mod
    monkeypatch.setitem(sys.modules, "orbax", orbax_mod)
    monkeypatch.setitem(sys.modules, "orbax.checkpoint", ocp_mod)
    inf._MODEL_CACHE.clear()
    res = inf.generate_sql(str(ckpt_dir), "SELECT 1")
    assert res["status"] == "success"


@pytest.mark.usefixtures("_mock_jax_env")
def test_generate_sql_success() -> None:
    """Initialize function test_generate_sql_success.

    Raises:
        AssertionError: Description.


        TypeError: Description.

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


    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(inf, "jax", None)
    with pytest.raises(DependencyMissingError, match=r"JAX inference dependencies are missing\."):
        generate_sql("mock-model", "test prompt")


@pytest.mark.usefixtures("_mock_jax_env")
def test_jax_beam_search() -> None:
    """Initialize function test_jax_beam_search.

    Raises:
        AssertionError: Description.

    """
    jnp_mock = MockJNP()

    def mock_apply_fn(seq: MockArray, _positions: object = None) -> MockArray:
        """Initialize function mock_apply_fn.

        Args:
        ----
        seq: Description of seq.


        Returns:
            object: Description of return.

        """
        logits = [0.0] * 300
        seq_len = len(seq.data[0]) if isinstance(seq.data[0], list) else len(seq.data)
        if seq_len == 1:
            logits[5] = 10.0
        else:
            logits[299] = 10.0
        return MockArray([logits])

    input_ids = jnp_mock.array([[1]])
    (result, _score) = jax_beam_search(model_apply_fn=mock_apply_fn, input_ids=input_ids, beam_width=2, max_length=5, eos_token_id=299)
    if not result.tolist() == [[1, 5, 299]]:
        raise AssertionError


def test_real_jax_beam_search() -> None:
    """Test native JAX beam search using genuine JAX arrays."""
    real_jax = pytest.importorskip("jax")
    real_jnp = real_jax.numpy

    def mock_model(seq: object, _positions: object) -> object:
        """Apply mock model returning realistic 3D logits.

        Args:
            seq: Input sequence array.
            _positions: Position array.

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        batch_size = seq.shape[0]
        seq_len = seq.shape[1]
        vocab_size = 50
        logits = real_jnp.zeros((batch_size, seq_len, vocab_size))
        if seq_len == 1:
            logits = logits.at[0, -1, 7].set(10.0)
        else:
            logits = logits.at[0, -1, 42].set(10.0)
        return logits

    input_ids = real_jnp.array([[1]])
    best_seq, score = jax_beam_search(
        model_apply_fn=mock_model,
        input_ids=input_ids,
        beam_width=2,
        max_length=5,
        eos_token_id=42,
    )
    assert [int(x) for x in best_seq[0]] == [1, 7, 42]
    assert score <= 0.0
    assert score > -1.0


def test_inference_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    mdl = __import__("gemma_4_sql.backends.jax.inference", fromlist=[""])
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "flax", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    importlib.reload(mdl)


def test_jax_inference_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """Test test_mode, logits shapes, caching, and orbax loading."""
    import gemma_4_sql.backends.jax.inference as jax_inf

    # 1. test_mode
    res = jax_inf.generate_sql("model", "select *", test_mode=True)
    assert res["status"] == "success"
    assert res["sql"] == "SELECT * FROM jax_table"

    # 2. 1D logits and 2D logits in _compute_step_probs
    real_jax = pytest.importorskip("jax")
    real_jnp = real_jax.numpy
    idx1, _prob1 = jax_inf._compute_step_probs(real_jnp.zeros((10,)), 2)
    assert len(idx1) == 2
    idx2, _prob2 = jax_inf._compute_step_probs(real_jnp.zeros((3, 10)), 2)
    assert len(idx2) == 2

    # 3. jax without jit in _beam_search_step
    monkeypatch.setattr(jax_inf, "jax", type("MockJax", (), {"nn": real_jax.nn})())
    beams = jax_inf._beam_search_step(real_jnp.array([[1]]), 0.0, lambda s, p: real_jnp.zeros((1, 1, 10)), 2)
    assert len(beams) == 2

    # 4. _MODEL_CACHE hit and orbax restore
    model_dir = tmp_path / "test_model_dir"
    model_dir.mkdir()
    fake_model = MagicMock()
    fake_model.return_value = real_jnp.zeros((1, 1, 10))
    monkeypatch.setattr(jax_inf, "Gemma4ForCausalLM", lambda *a, **k: fake_model)

    import flax.nnx as real_nnx

    mock_update = MagicMock()
    monkeypatch.setattr(real_nnx, "update", mock_update)
    monkeypatch.setattr(jax_inf, "nnx", real_nnx)

    import orbax.checkpoint as ocp

    mock_cp = MagicMock()
    mock_cp.restore.return_value = {"weights": 1}
    monkeypatch.setattr(ocp, "PyTreeCheckpointer", lambda: mock_cp)

    jax_inf._MODEL_CACHE.clear()
    res1 = jax_inf.generate_sql(str(model_dir), "select", beam_width=1, max_length=1)
    assert res1["status"] == "success"
    assert str(model_dir) in jax_inf._MODEL_CACHE
    mock_update.assert_called_once()

    # Second run hits _MODEL_CACHE
    res2 = jax_inf.generate_sql(str(model_dir), "select", beam_width=1, max_length=1)
    assert res2["status"] == "success"
