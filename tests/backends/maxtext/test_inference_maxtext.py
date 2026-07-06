"""Tests for MaxText inference logic."""

from typing import NoReturn as Never

import pytest

import gemma_4_sql.backends.maxtext.inference as inf
from gemma_4_sql.backends.maxtext.inference import generate_sql, maxtext_beam_search


class MockArray:
    """Mock JAX Array for MaxText."""

    def __init__(self: object, data: object) -> None:
        """Initialize function __init__.

        Args:
        ----
        data: Description of data.

        """
        self.data = data if isinstance(data, list) else [data]

    @property
    def shape(self) -> tuple[int, ...]:
        """Execute function."""
        if isinstance(self.data, list) and len(self.data) > 0 and isinstance(self.data[0], list):
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
        d = array.data
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
        return x


class MockJAX:
    """Mock JAX."""

    nn = MockNN()

    @staticmethod
    def jit(fn: object, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return fn


class MockGemma4Model:
    """Mock Gemma 4 Model."""

    def __init__(self: object, name: object) -> None:
        """Initialize function __init__.

        Args:
        ----
        name: Description of name.

        """
        self.name = name

    def apply(self: object, _seq: object) -> object:
        """Initialize function apply.

        Returns:
            object: Description of return.

        """
        logits = [0.0] * 300
        logits[100] = 10.0
        return MockArray([logits])


@pytest.fixture
def _mock_maxtext_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function mock_maxtext_env.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(inf, "jax", MockJAX())
    monkeypatch.setattr(inf, "jnp", MockJNP())
    monkeypatch.setattr(inf, "Gemma4Model", MockGemma4Model)


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_generate_sql_success() -> None:
    """Initialize function test_generate_sql_success.

    Raises:
        AssertionError: Description.


        TypeError: Description.

    """
    res = generate_sql("mock-model", "test prompt", beam_width=2, max_length=3)
    if not res["status"] == "success":
        raise AssertionError
    if not res["backend"] == "maxtext":
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
    with pytest.raises(DependencyMissingError):
        generate_sql("mock-model", "test prompt")


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_maxtext_beam_search() -> None:
    """Initialize function test_maxtext_beam_search.

    Raises:
        AssertionError: Description.

    """
    jnp_mock = MockJNP()

    def mock_apply_fn(seq: MockArray) -> MockArray:
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
    (result, _score) = maxtext_beam_search(model_apply_fn=mock_apply_fn, input_ids=input_ids, beam_width=2, max_length=5, eos_token_id=299)
    if not result.tolist() == [[1, 5, 299]]:
        raise AssertionError


def test_inference_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    m_inf = __import__("gemma_4_sql.backends.maxtext.inference", fromlist=[""])
    monkeypatch.setitem(sys.modules, "jax", None)
    importlib.reload(m_inf)
    monkeypatch.undo()
    importlib.reload(m_inf)


class MockTokenizer:
    """Provide class docstring."""

    vocab_size = 10

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Execute function."""

    def encode(self, _x: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return [1]

    def decode(self, _x: object) -> str:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return "decoded"


class MockResult:
    """Provide class docstring."""

    def tolist(self) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return [1]

    @property
    def shape(self) -> object:
        """Execute function."""
        return (1, 1)

    def __len__(self) -> int:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return 1


def test_inference_no_jit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    m_inf = __import__("gemma_4_sql.backends.maxtext.inference", fromlist=[""])
    monkeypatch.setattr(m_inf, "jax", type("M", (), {"jit": lambda x, **_kwargs: x, "random": type("R", (), {"PRNGKey": lambda x: x})}))
    monkeypatch.setattr(m_inf, "jnp", type("M", (), {"array": lambda x, **_kwargs: x, "int32": 1}))
    monkeypatch.setattr(m_inf, "Gemma4Model", lambda *_args, **_kwargs: type("M", (), {"init": lambda *_args: None, "apply": lambda *_args, **_kwargs: None})())
    monkeypatch.setattr(m_inf, "SQLTokenizer", MockTokenizer)
    monkeypatch.setattr(m_inf, "maxtext_beam_search", lambda *_args, **_kwargs: ([MockResult()], 0.95))
    res = m_inf.generate_sql("m", "prompt", test_mode=True, use_jit=False)
    if res["status"] != "success":
        raise AssertionError


def test_inference_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    m_inf = __import__("gemma_4_sql.backends.maxtext.inference", fromlist=[""])
    monkeypatch.setattr(m_inf, "jax", type("M", (), {"jit": lambda x, **_kwargs: x, "random": type("R", (), {"PRNGKey": lambda x: x})}))
    monkeypatch.setattr(m_inf, "jnp", type("M", (), {"array": lambda x, **_kwargs: x, "int32": 1}))
    monkeypatch.setattr(m_inf, "Gemma4Model", lambda *_args, **_kwargs: type("M", (), {"init": lambda *_args: None, "apply": lambda *_args, **_kwargs: None})())

    def raise_err(*_args: object, **_kwargs: object) -> Never:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(m_inf, "SQLTokenizer", MockTokenizer)
    monkeypatch.setattr(m_inf, "maxtext_beam_search", lambda *_args, **_kwargs: [type("M", (), {"tolist": lambda _self: [1]})()])
    res = m_inf.generate_sql("m", "prompt", test_mode=True, use_jit=False)
    if "failed" not in res["status"]:
        raise AssertionError
