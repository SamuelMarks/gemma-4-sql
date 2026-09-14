"""Tests for PyTorch inference."""

from __future__ import annotations

import typing

import pytest

import gemma_4_sql.backends.pytorch.inference as pt_inf
from gemma_4_sql.backends.pytorch.inference import generate_sql


class MockTorch:
    """Provide class docstring."""


class MockAutoModelForCausalLM:
    """Provide class docstring."""


class MockAutoTokenizer:
    """Provide class docstring."""


def test_inference_pytorch_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(pt_inf, "torch", MockTorch())
    monkeypatch.setattr(pt_inf, "AutoModelForCausalLM", MockAutoModelForCausalLM)
    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockAutoTokenizer)
    res = generate_sql("mock", "hi", beam_width=1, max_length=2, test_mode=True)
    if not res["status"] == "success":
        raise AssertionError
    if not res["model"] == "mock":
        raise AssertionError


def test_inference_pytorch_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(pt_inf, "torch", None)
    with pytest.raises(DependencyMissingError, match=r"PyTorch dependencies are missing\."):
        generate_sql("mock", "hi")


def test_inference_pytorch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(pt_inf, "torch", MockTorch())
    monkeypatch.setattr(pt_inf, "AutoModelForCausalLM", MockAutoModelForCausalLM)
    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockAutoTokenizer)

    def raise_err(*_args: object, **_kwargs: object) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(pt_inf, "_run_generation", raise_err)
    res = generate_sql("mock", "hi", test_mode=True)
    if "failed" not in str(res["status"]):
        raise AssertionError


class MockTokenizerObj:
    """Provide class docstring."""

    def __call__(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return type("M", (), {"to": lambda *_a, **_k: {"input_ids": [1]}})()

    def decode(self, *_args: object, **_kwargs: object) -> str:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return "prompt SELECT * FROM x"


class MockTokenizer:
    """Provide class docstring."""

    @classmethod
    def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockTokenizerObj()


class MockModelObj:
    """Provide class docstring."""

    device = "cpu"

    def generate(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """

        class MockOutputs:
            """Provide class docstring."""

            sequences: typing.ClassVar = [[1, 2]]
            sequences_scores: typing.ClassVar = [type("T", (), {"item": lambda _self: 0.99})()]

        return MockOutputs()


class MockModel:
    """Provide class docstring."""

    @classmethod
    def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockModelObj()


def test_inference_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    m_inf = __import__("gemma_4_sql.backends.pytorch.inference", fromlist=[""])
    monkeypatch.setattr(m_inf, "torch", object())
    monkeypatch.setattr(m_inf, "AutoTokenizer", MockTokenizer)
    monkeypatch.setattr(m_inf, "AutoModelForCausalLM", MockModel)
    res = m_inf.generate_sql("m", "prompt", test_mode=False)
    if res["status"] != "success":
        raise AssertionError
    if res["sql"] != "SELECT * FROM x":
        raise AssertionError


def test_inference_pytorch_native_with_sql_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test pytorch_native inference when AutoTokenizer is None.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import gemma_4_sql.backends.pytorch.inference as pt_inf
    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4Config

    monkeypatch.setattr(pt_inf, "AutoTokenizer", None)
    tiny_cfg = Gemma4Config(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        intermediate_size=128,
    )
    res = pt_inf.generate_sql("test_model", "SELECT 1", max_length=2, backend_alias="pytorch_native", config=tiny_cfg)
    assert res["status"] == "success"
    assert res["backend"] == "pytorch_native"


def test_inference_sequences_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test inference sequences fallback branch when input_ids has no shape.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import torch

    import gemma_4_sql.backends.pytorch.inference as pt_inf

    class MockTokenizerNoShape:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            return cls()

        def __call__(self, _prompt: str, return_tensors: str = "pt") -> object:
            class Inputs(dict):
                def __init__(self) -> None:
                    super().__init__({"input_ids": [1, 2]})

                def to(self, _dev: object) -> object:
                    return self

            return Inputs()

        def decode(self, _tokens: object, skip_special_tokens: bool = True) -> str:
            return "prompt SELECT * FROM fallback"

    class MockModelOutputs:
        def __init__(self) -> None:
            self.device = "cpu"
            self.sequences = ["prompt SELECT * FROM fallback"]
            self.sequences_scores = [torch.tensor(0.9)]

        def generate(self, **kwargs: object) -> object:
            return self

    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockTokenizerNoShape)
    monkeypatch.setattr(pt_inf, "AutoModelForCausalLM", type("M", (), {"from_pretrained": lambda *a, **k: MockModelOutputs()}))
    res = pt_inf.generate_sql("model", "prompt", test_mode=False)
    assert res["status"] == "success"
    assert res["sql"] == "SELECT * FROM fallback"

    # Test fallback when generated text doesn't start with prompt
    class MockModelOutputsNoPrefix:
        def __init__(self) -> None:
            self.device = "cpu"
            self.sequences = ["SELECT * FROM no_prefix"]

        def generate(self, **kwargs: object) -> object:
            return self

    class MockTokenizerNoPrefix(MockTokenizerNoShape):
        def decode(self, _tokens: object, skip_special_tokens: bool = True) -> str:
            return "SELECT * FROM no_prefix"

    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockTokenizerNoPrefix)
    monkeypatch.setattr(pt_inf, "AutoModelForCausalLM", type("M", (), {"from_pretrained": lambda *a, **k: MockModelOutputsNoPrefix()}))
    res2 = pt_inf.generate_sql("model", "prompt", test_mode=False)
    assert res2["sql"] == "SELECT * FROM no_prefix"


def test_inference_pytorch_native_with_autotokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test pytorch_native inference with AutoTokenizer working and with error."""
    import torch

    import gemma_4_sql.backends.pytorch.inference as pt_inf
    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4Config

    tiny_cfg = Gemma4Config(vocab_size=128, hidden_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=32, intermediate_size=128)

    class MockNativeTokenizer:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            return cls()

        def __call__(self, _prompt: str, return_tensors: str = "pt") -> object:
            return type("Inputs", (), {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)})()

        def decode(self, _tokens: object, skip_special_tokens: bool = True) -> str:
            return "SELECT * FROM native_table"

    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockNativeTokenizer)
    res = pt_inf.generate_sql("model", "prompt", max_length=2, backend_alias="pytorch_native", config=tiny_cfg)
    assert res["status"] == "success"
    assert res["sql"] == "SELECT * FROM native_table"

    # Test AutoTokenizer raising ValueError
    class MockErrorTokenizer:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            raise ValueError("tokenizer load error")

    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockErrorTokenizer)
    res_err = pt_inf.generate_sql("model", "prompt", max_length=2, backend_alias="pytorch_native", config=tiny_cfg)
    assert res_err["status"] == "success"


def test_inference_token_slicing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test standard token-level slicing branch in PyTorch inference.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import torch

    import gemma_4_sql.backends.pytorch.inference as pt_inf

    class MockTokenizerWithShape:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            return cls()

        def __call__(self, _prompt: str, return_tensors: str = "pt") -> object:
            class Inputs(dict):
                def __init__(self) -> None:
                    super().__init__({"input_ids": torch.tensor([[1, 2]])})
                    self.input_ids = torch.tensor([[1, 2]])

                def to(self, _dev: object) -> object:
                    return self

            return Inputs()

        def decode(self, _tokens: object, skip_special_tokens: bool = True) -> str:
            return "SELECT * FROM sliced_tokens"

    class MockModelSlicing:
        def __init__(self) -> None:
            self.device = "cpu"
            self.sequences = [torch.tensor([1, 2, 3, 4])]
            self.sequences_scores = [torch.tensor(0.95)]

        def generate(self, **kwargs: object) -> object:
            return self

    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockTokenizerWithShape)
    monkeypatch.setattr(pt_inf, "AutoModelForCausalLM", type("M", (), {"from_pretrained": lambda *a, **k: MockModelSlicing()}))
    res = pt_inf.generate_sql("model", "prompt", test_mode=False)
    assert res["status"] == "success"
    assert res["sql"] == "SELECT * FROM sliced_tokens"
