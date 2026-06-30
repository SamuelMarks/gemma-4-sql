"""Tests for MaxText inference logic."""

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

        Args:
        ----
        seq: Description of seq.

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

    Args:
    ----
    mock_maxtext_env: Description of mock_maxtext_env.

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

    """
    monkeypatch.setattr(inf, "jax", None)
    res = generate_sql("mock-model", "test prompt")
    if not res["status"] == "mocked_missing_maxtext":
        raise AssertionError


@pytest.mark.usefixtures("_mock_maxtext_env")
def test_maxtext_beam_search() -> None:
    """Initialize function test_maxtext_beam_search.

    Args:
    ----
    mock_maxtext_env: Description of mock_maxtext_env.

    """
    jnp_mock = MockJNP()

    def mock_apply_fn(seq: MockArray) -> MockArray:
        """Initialize function mock_apply_fn.

        Args:
        ----
        seq: Description of seq.

        """
        logits = [0.0] * 300
        seq_len = len(seq.data[0]) if isinstance(seq.data[0], list) else len(seq.data)
        if seq_len == 1:
            logits[5] = 10.0
        else:
            logits[299] = 10.0
        return MockArray([logits])

    input_ids = jnp_mock.array([[1]])
    result = maxtext_beam_search(model_apply_fn=mock_apply_fn, input_ids=input_ids, beam_width=2, max_length=5, eos_token_id=299)
    if not result.tolist() == [[1, 5, 299]]:
        raise AssertionError
