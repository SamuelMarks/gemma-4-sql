"""Tests for PyTorch inference."""

import typing

import gemma_4_sql.backends.pytorch.inference as inf
import pytest
from gemma_4_sql.backends.pytorch.inference import generate_sql, pytorch_beam_search


class MockTensor:
    """Initialize class MockTensor."""

    def __init__(self: typing.Any, data: object, _dtype: object = None) -> None:
        """Initialize function __init__.

        Args:
        ----
        data: Description of data.
        dtype: Description of dtype.

        """
        self.data = data if isinstance(data, list) else [data]

    def clone(self: typing.Any) -> object:
        """Initialize function clone."""
        copy = __import__("copy")
        return MockTensor(copy.deepcopy(self.data))

    def item(self: typing.Any) -> object:
        """Initialize function item."""
        return self.data[0] if isinstance(self.data, list) else self.data

    def unsqueeze(self: typing.Any, _dim: object) -> object:
        """Initialize function unsqueeze.

        Args:
        ----
        dim: Description of dim.

        """
        if not isinstance(self.data, list):
            return MockTensor([self.data])
        return self

    def tolist(self: typing.Any) -> object:
        """Initialize function tolist."""

        def flatten(lst: object) -> object:
            """Initialize function flatten.

            Args:
            ----
            lst: Description of lst.

            """
            res = []
            for item in lst:  # type: ignore[attr-defined]
                if isinstance(item, list):
                    res.extend(flatten(item))
                else:
                    res.append(item)
            return res

        return flatten(self.data)

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
        try:
            return MockTensor(self.data[idx])
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, OSError):
            return MockTensor(self.data)


class MockTorch:
    """Initialize class MockTorch."""

    long = 1

    @staticmethod
    def tensor(data: object, _dtype: object = None, **_kwargs: object) -> object:
        """Initialize function tensor.

        Args:
        ----
        data: Description of data.
        dtype: Description of dtype.

        """
        return MockTensor(data)

    @staticmethod
    def log_softmax(x: object, _dim: object = -1, **_kwargs: object) -> object:
        """Initialize function log_softmax.

        Args:
        ----
        x: Description of x.
        dim: Description of dim.

        """
        return x

    @staticmethod
    def topk(log_probs: object, k: object) -> object:
        """Initialize function topk.

        Args:
        ----
        log_probs: Description of log_probs.
        k: Description of k.

        """
        d = log_probs.data  # type: ignore[attr-defined]
        indices = sorted(range(len(d)), key=lambda x: d[x], reverse=True)[:k]  # type: ignore[misc]
        probs = [d[i] for i in indices]
        return (MockTensor(probs), MockTensor(indices))

    @staticmethod
    def cat(tensors: object, _dim: object = -1, **_kwargs: object) -> object:
        """Initialize function cat.

        Args:
        ----
        tensors: Description of tensors.
        dim: Description of dim.

        """
        res = []
        for t in tensors:  # type: ignore[attr-defined]
            d = t.data
            if isinstance(d, list) and len(d) > 0 and isinstance(d[0], list):
                d = [item for sub in d for item in sub]
            if not isinstance(d, list):
                d = [d]
            res.extend(d)
        return MockTensor([res])


class MockGemma4ForCausalLM:
    """Initialize class MockGemma4ForCausalLM."""

    @classmethod
    def from_pretrained(cls: typing.Any, *_args: object, **_kwargs: object) -> object:
        """Initialize function from_pretrained.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        return cls()

    def __call__(self: typing.Any, _seq: object) -> object:
        """Initialize function __call__.

        Args:
        ----
        seq: Description of seq.

        """
        logits = [0.0] * 300
        logits[100] = 10.0
        return MockTensor([logits])


@pytest.fixture()
def _mock_torch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function mock_torch_env.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(inf, "torch", MockTorch())
    monkeypatch.setattr(inf, "Gemma4ForCausalLM", MockGemma4ForCausalLM)


@pytest.mark.usefixtures("_mock_torch_env")
def test_inference_pytorch_real() -> None:
    """Initialize function test_inference_pytorch_real.

    Args:
    ----
    mock_torch_env: Description of mock_torch_env.

    """
    res = generate_sql("mock", "hi", beam_width=1, max_length=2)
    if not res["status"] == "success":
        raise AssertionError
    if not res["model"] == "mock":
        raise AssertionError


def test_inference_pytorch_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function test_inference_pytorch_missing_deps.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(inf, "torch", None)
    res = generate_sql("mock", "hi")
    if not res["status"] == "mocked_missing_torch":
        raise AssertionError


@pytest.mark.usefixtures("_mock_torch_env")
def test_pytorch_beam_search() -> None:
    """Initialize function test_pytorch_beam_search.

    Args:
    ----
    mock_torch_env: Description of mock_torch_env.

    """
    torch_mock = MockTorch()

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

    input_ids = torch_mock.tensor([[1]], dtype=torch_mock.long)  # type: ignore[call-arg]
    result = pytorch_beam_search(model=mock_model, input_ids=input_ids, beam_width=2, max_length=5, eos_token_id=299)
    if not result.tolist() == [1, 5, 299]:
        raise AssertionError
