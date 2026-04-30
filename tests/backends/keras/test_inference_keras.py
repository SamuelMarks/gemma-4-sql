"""Tests for Keras inference logic."""

from typing import Any

import pytest

import gemma_4_sql.backends.keras.inference as inf
from gemma_4_sql.backends.keras.inference import (
    generate_sql,
    keras_beam_search,
)


class MockTensor:
    def __init__(self, data: Any, dtype=None):
        self.data = data if isinstance(data, list) else [data]
        self.dtype = dtype

    def numpy(self):
        # mock numpy for simple tests
        return (
            self.data[0]
            if len(self.data) == 1 and not isinstance(self.data[0], list)
            else self
        )

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) == 2:
            try:
                return MockTensor(self.data[idx[0]][idx[1]])
            except:
                return MockTensor(self.data[idx[0]])
        if isinstance(idx, slice):
            return MockTensor(self.data[idx])
        return MockTensor(self.data[idx])

    def tolist(self):
        return self.data


class MockNN:
    def log_softmax(self, x, axis=-1):
        return x


class MockMath:
    def top_k(self, log_probs, k=1):
        d = log_probs.data
        indices = sorted(range(len(d)), key=lambda x: d[x], reverse=True)[:k]
        probs = [d[i] for i in indices]
        return MockTensor(probs), MockTensor(indices)


class MockTF:
    int32 = 1
    float32 = 2
    nn = MockNN()
    math = MockMath()

    def constant(self, data, dtype=None):
        return MockTensor(data, dtype)

    def reshape(self, tensor, shape):
        if shape == (1, 1):
            val = tensor.data[0] if isinstance(tensor.data, list) else tensor.data
            return MockTensor([[val]])
        return tensor

    def cast(self, tensor, dtype):
        return tensor

    def concat(self, arrays, axis=0):
        if axis == -1:
            res = []
            for i in range(len(arrays[0].data)):
                res.append(arrays[0].data[i] + arrays[1].data[i])
            return MockTensor(res)
        return MockTensor([a.data for a in arrays])

    def scatter_nd(self, indices, updates, shape):
        # simple mock
        data = [0.0] * shape[1]
        data[indices[0][1]] = updates[0]
        return MockTensor([data])


class MockKerasModel:
    def __init__(self, vocab_size: int, eos_token_id: int):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

    def __call__(self, x):
        logits = [0.0] * self.vocab_size
        logits[100] = 10.0
        return MockTensor([logits])


@pytest.fixture
def mock_keras_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inf, "tf", MockTF())
    monkeypatch.setattr(inf, "keras", object())


def test_generate_sql_success(mock_keras_env: None) -> None:
    res = generate_sql("mock-model", "test prompt", beam_width=2, max_length=3)
    assert res["status"] == "success"
    assert res["backend"] == "keras"
    assert isinstance(res["sql"], str)


def test_generate_sql_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inf, "keras", None)
    res = generate_sql("mock-model", "test prompt")
    assert res["status"] == "mocked_missing_keras"


def test_keras_beam_search(mock_keras_env: None) -> None:
    tf_mock = MockTF()

    def mock_model(seq: MockTensor) -> MockTensor:
        logits = [0.0] * 300
        seq_len = len(seq.data[0]) if isinstance(seq.data[0], list) else len(seq.data)
        if seq_len == 1:
            logits[5] = 10.0
        else:
            logits[299] = 10.0  # EOS
        return MockTensor([logits])

    input_ids = tf_mock.constant([[1]])
    result = keras_beam_search(
        model=mock_model,
        input_ids=input_ids,
        beam_width=2,
        max_length=5,
        eos_token_id=299,
    )

    assert result.tolist() == [[1, 5, 299]]
