"""Tests for Keras inference logic."""

import typing

import gemma_4_sql.backends.keras.inference as inf
import pytest
from gemma_4_sql.backends.keras.inference import generate_sql, keras_beam_search


class MockTensor:
    """Initialize class MockTensor."""

    def __init__(self: typing.Any, data: object, dtype: object = None) -> None:
        """Initialize function __init__.

        Args:
        ----
        data: Description of data.
        dtype: Description of dtype.

        """
        self.data = data if isinstance(data, list) else [data]
        self.dtype = dtype

    def numpy(self: typing.Any) -> object:
        """Initialize function numpy."""
        return self.data[0] if len(self.data) == 1 and (not isinstance(self.data[0], list)) else self

    def __getitem__(self: typing.Any, idx: object) -> object:
        """Initialize function __getitem__.

        Args:
        ----
        idx: Description of idx.

        """
        if isinstance(idx, tuple) and len(idx) == int("2"):
            try:
                return MockTensor(self.data[idx[0]][idx[1]])
            except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
                return MockTensor(self.data[idx[0]])
        if isinstance(idx, slice):
            return MockTensor(self.data[idx])
        return MockTensor(self.data[idx])

    def tolist(self: typing.Any) -> object:
        """Initialize function tolist."""
        return self.data


class MockNN:
    """Initialize class MockNN."""

    def log_softmax(self: typing.Any, x: object, _axis: object = -1) -> object:
        """Initialize function log_softmax.

        Args:
        ----
        x: Description of x.
        axis: Description of axis.

        """
        return x


class MockMath:
    """Initialize class MockMath."""

    def top_k(self: typing.Any, log_probs: object, k: object = 1) -> object:
        """Initialize function top_k.

        Args:
        ----
        log_probs: Description of log_probs.
        k: Description of k.

        """
        d = log_probs.data  # type: ignore[attr-defined]
        indices = sorted(range(len(d)), key=lambda x: d[x], reverse=True)[:k]  # type: ignore[misc]
        probs = [d[i] for i in indices]
        return (MockTensor(probs), MockTensor(indices))


class MockTF:
    """Initialize class MockTF."""

    int32 = 1
    float32 = 2
    nn = MockNN()
    math = MockMath()

    def constant(self: typing.Any, data: object, dtype: object = None) -> object:
        """Initialize function constant.

        Args:
        ----
        data: Description of data.
        dtype: Description of dtype.

        """
        return MockTensor(data, dtype)

    def reshape(self: typing.Any, tensor: object, shape: object) -> object:
        """Initialize function reshape.

        Args:
        ----
        tensor: Description of tensor.
        shape: Description of shape.

        """
        if shape == (1, 1):
            val = tensor.data[0] if isinstance(tensor.data, list) else tensor.data  # type: ignore[attr-defined]
            return MockTensor([[val]])
        return tensor

    def cast(self: typing.Any, tensor: object, _dtype: object) -> object:
        """Initialize function cast.

        Args:
        ----
        tensor: Description of tensor.
        dtype: Description of dtype.

        """
        return tensor

    def concat(self: typing.Any, arrays: object, axis: object = 0) -> object:
        """Initialize function concat.

        Args:
        ----
        arrays: Description of arrays.
        axis: Description of axis.

        """
        if axis == -1:
            res = [arrays[0].data[i] + arrays[1].data[i] for i in range(len(arrays[0].data))]  # type: ignore[index]
            return MockTensor(res)
        return MockTensor([a.data for a in arrays])  # type: ignore[attr-defined]

    def scatter_nd(self: typing.Any, indices: object, updates: object, shape: object) -> object:
        """Initialize function scatter_nd.

        Args:
        ----
        indices: Description of indices.
        updates: Description of updates.
        shape: Description of shape.

        """
        data = [0.0] * shape[1]  # type: ignore[index]
        data[indices[0][1]] = updates[0]  # type: ignore[index]
        return MockTensor([data])


class MockKerasModel:
    """Initialize class MockKerasModel."""

    def __init__(self: typing.Any, vocab_size: int, eos_token_id: int) -> None:
        """Initialize function __init__.

        Args:
        ----
        vocab_size: Description of vocab_size.
        eos_token_id: Description of eos_token_id.

        """
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

    def __call__(self: typing.Any, _x: object) -> object:
        """Initialize function __call__.

        Args:
        ----
        x: Description of x.

        """
        logits = [0.0] * self.vocab_size
        logits[100] = 10.0
        return MockTensor([logits])


@pytest.fixture()
def _mock_keras_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function mock_keras_env.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(inf, "tf", MockTF())
    monkeypatch.setattr(inf, "keras", object())


@pytest.mark.usefixtures("_mock_keras_env")
def test_generate_sql_success() -> None:
    """Initialize function test_generate_sql_success.

    Args:
    ----
    mock_keras_env: Description of mock_keras_env.

    """
    res = generate_sql("mock-model", "test prompt", beam_width=2, max_length=3)
    if not res["status"] == "success":
        raise AssertionError
    if not res["backend"] == "keras":
        raise AssertionError
    if not isinstance(res["sql"], str):
        raise TypeError


def test_generate_sql_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function test_generate_sql_missing_deps.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(inf, "keras", None)
    res = generate_sql("mock-model", "test prompt")
    if not res["status"] == "mocked_missing_keras":
        raise AssertionError


@pytest.mark.usefixtures("_mock_keras_env")
def test_keras_beam_search() -> None:
    """Initialize function test_keras_beam_search.

    Args:
    ----
    mock_keras_env: Description of mock_keras_env.

    """
    tf_mock = MockTF()

    def mock_model(seq: MockTensor) -> MockTensor:
        """Initialize function mock_model.

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
        return MockTensor([logits])

    input_ids = tf_mock.constant([[1]])
    result = keras_beam_search(model=mock_model, input_ids=input_ids, beam_width=2, max_length=5, eos_token_id=299)
    if not result.tolist() == [[1, 5, 299]]:
        raise AssertionError
