"""Tests for MaxText inference logic."""

from typing import Any

import pytest

import gemma_4_sql.backends.maxtext.inference as inf
from gemma_4_sql.backends.maxtext.inference import generate_sql, maxtext_beam_search


class MockArray:
    """Mock JAX Array for MaxText."""

    def __init__(self, data: Any):
        self.data = data if isinstance(data, list) else [data]

    def __getitem__(self, idx):
        if isinstance(idx, MockArray):
            return MockArray([self.data[i] for i in idx.data])
        if isinstance(idx, slice):
            return MockArray(self.data[idx])
        if isinstance(idx, tuple):
            if len(idx) == 2:
                try:
                    return self.data[idx[0]][idx[1]]
                except:
                    return self.data[idx[0]]
        return MockArray(self.data[idx])

    def tolist(self):
        return self.data

    def item(self):
        return self.data[0] if isinstance(self.data, list) else self.data

    def reshape(self, *shape):
        if shape == (1, 1):
            val = self.data[0] if isinstance(self.data, list) else self.data
            return MockArray([[val]])
        return self


class MockJNP:
    """Mock JNP."""

    def array(self, data, dtype=None):
        return MockArray(data)

    int32 = 1

    def concatenate(self, arrays, axis=0):
        if axis == -1:
            res = []
            for i in range(len(arrays[0].data)):
                res.append(arrays[0].data[i] + arrays[1].data[i])
            return MockArray(res)
        return MockArray([a.data for a in arrays])

    def argsort(self, array):
        d = array.data
        return MockArray(sorted(range(len(d)), key=lambda x: d[x]))


class MockNN:
    def log_softmax(self, x, axis=-1):
        return x


class MockJAX:
    """Mock JAX."""

    nn = MockNN()


class MockGemma4Model:
    """Mock Gemma 4 Model."""

    def __init__(self, name):
        self.name = name

    def apply(self, seq):
        logits = [0.0] * 300
        logits[100] = 10.0
        return MockArray([logits])


@pytest.fixture
def mock_maxtext_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inf, "jax", MockJAX())
    monkeypatch.setattr(inf, "jnp", MockJNP())
    monkeypatch.setattr(inf, "Gemma4Model", MockGemma4Model)


def test_generate_sql_success(mock_maxtext_env: None) -> None:
    res = generate_sql("mock-model", "test prompt", beam_width=2, max_length=3)
    assert res["status"] == "success"
    assert res["backend"] == "maxtext"
    assert isinstance(res["sql"], str)


def test_generate_sql_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inf, "jax", None)
    res = generate_sql("mock-model", "test prompt")
    assert res["status"] == "mocked_missing_maxtext"


def test_maxtext_beam_search(mock_maxtext_env: None) -> None:
    jnp_mock = MockJNP()

    def mock_apply_fn(seq: MockArray) -> MockArray:
        logits = [0.0] * 300
        seq_len = len(seq.data[0]) if isinstance(seq.data[0], list) else len(seq.data)
        if seq_len == 1:
            logits[5] = 10.0
        else:
            logits[299] = 10.0  # EOS
        return MockArray([logits])

    input_ids = jnp_mock.array([[1]])
    result = maxtext_beam_search(
        model_apply_fn=mock_apply_fn,
        input_ids=input_ids,
        beam_width=2,
        max_length=5,
        eos_token_id=299,
    )

    assert result.tolist() == [[1, 5, 299]]
