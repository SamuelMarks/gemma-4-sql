"""Tests for PyTorch inference."""

from typing import Any

import pytest

import gemma_4_sql.backends.pytorch.inference as inf
from gemma_4_sql.backends.pytorch.inference import (
    generate_sql,
    pytorch_beam_search,
)


class MockTensor:
    def __init__(self, data: Any, dtype=None):
        self.data = data if isinstance(data, list) else [data]

    def clone(self):
        import copy

        return MockTensor(copy.deepcopy(self.data))

    def item(self):
        return self.data[0] if isinstance(self.data, list) else self.data

    def unsqueeze(self, dim):
        # for our mock, unsqueeze doesn't change underlying data shape significantly
        # but topk returns single values, so it's [val]. unsqueeze(0).unsqueeze(0) -> [[[val]]]
        # we'll just keep it flat or single nested
        if not isinstance(self.data, list):
            return MockTensor([self.data])
        return self

    def tolist(self):
        # ensure fully flat list of ints for tokenization
        def flatten(lst):
            res = []
            for item in lst:
                if isinstance(item, list):
                    res.extend(flatten(item))
                else:
                    res.append(item)
            return res

        return flatten(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) == 2:
            try:
                return MockTensor(self.data[idx[0]][idx[1]])
            except:
                return MockTensor(self.data[idx[0]])
        if isinstance(idx, slice):
            return MockTensor(self.data[idx])
        return MockTensor(self.data[idx])


class MockTorch:
    long = 1

    @staticmethod
    def tensor(data, dtype):
        return MockTensor(data)

    @staticmethod
    def log_softmax(x, dim=-1):
        return x

    @staticmethod
    def topk(log_probs, k):
        d = log_probs.data
        indices = sorted(range(len(d)), key=lambda x: d[x], reverse=True)[:k]
        probs = [d[i] for i in indices]
        return MockTensor(probs), MockTensor(indices)

    @staticmethod
    def cat(tensors, dim=-1):
        res = []
        for t in tensors:
            d = t.data
            if isinstance(d, list) and len(d) > 0 and isinstance(d[0], list):
                # extract inner lists
                d = [item for sub in d for item in sub]
            if not isinstance(d, list):
                d = [d]
            res.extend(d)
        return MockTensor([res])


class MockGemma4ForCausalLM:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, seq):
        logits = [0.0] * 300
        logits[100] = 10.0
        return MockTensor([logits])


@pytest.fixture
def mock_torch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inf, "torch", MockTorch())
    monkeypatch.setattr(inf, "Gemma4ForCausalLM", MockGemma4ForCausalLM)


def test_inference_pytorch_real(mock_torch_env: None) -> None:
    res = generate_sql("mock", "hi", beam_width=1, max_length=2)
    assert res["status"] == "success"
    assert "mock" == res["model"]


def test_inference_pytorch_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inf, "torch", None)
    res = generate_sql("mock", "hi")
    assert res["status"] == "mocked_missing_torch"


def test_pytorch_beam_search(mock_torch_env: None) -> None:
    torch_mock = MockTorch()

    def mock_model(seq: MockTensor) -> MockTensor:
        logits = [0.0] * 300
        seq_len = len(seq.data[0]) if isinstance(seq.data[0], list) else len(seq.data)
        if seq_len == 1:
            logits[5] = 10.0
        else:
            logits[299] = 10.0  # EOS
        return MockTensor([logits])

    input_ids = torch_mock.tensor([[1]], dtype=torch_mock.long)
    result = pytorch_beam_search(
        model=mock_model,
        input_ids=input_ids,
        beam_width=2,
        max_length=5,
        eos_token_id=299,
    )

    assert result.tolist() == [1, 5, 299]
