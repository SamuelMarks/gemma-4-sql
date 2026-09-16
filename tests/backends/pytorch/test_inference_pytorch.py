"""Tests for PyTorch inference."""

from __future__ import annotations

import typing
from collections import UserDict
from pathlib import Path

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
    """Test PyTorch generate_sql top-level handler."""
    monkeypatch.setattr(pt_inf, "torch", MockTorch())
    monkeypatch.setattr(pt_inf, "AutoModelForCausalLM", MockAutoModelForCausalLM)
    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockAutoTokenizer)
    monkeypatch.setattr(pt_inf, "_run_generation", lambda *a, **k: ("SELECT * FROM t", 0.95))
    res = generate_sql("mock", "hi", beam_width=1, max_length=2)
    assert res["status"] == "success"
    assert res["model"] == "mock"
    assert res["sql"] == "SELECT * FROM t"
    assert res["confidence_score"] == pytest.approx(0.95)


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
        """Test class for MockTokenizerNoShape."""

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            """Execute from pretrained helper."""
            return cls()

        def __call__(self, _prompt: str, return_tensors: str = "pt") -> object:
            """Initialize __call__."""

            class Inputs(UserDict):
                """Test class for Inputs."""

                def __init__(self) -> None:
                    """Initialize __init__."""
                    super().__init__({"input_ids": [1, 2]})

                def to(self, _dev: object) -> object:
                    """Execute to helper."""
                    return self

            return Inputs()

        def decode(self, _tokens: object, skip_special_tokens: bool = True) -> str:
            """Execute decode helper."""
            return "prompt SELECT * FROM fallback"

    class MockModelOutputs:
        """Test class for MockModelOutputs."""

        def __init__(self) -> None:
            """Initialize __init__."""
            self.device = "cpu"
            self.sequences = ["prompt SELECT * FROM fallback"]
            self.sequences_scores = [torch.tensor(0.9)]

        def generate(self, **kwargs: object) -> object:
            """Execute generate helper."""
            return self

    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockTokenizerNoShape)
    monkeypatch.setattr(pt_inf, "AutoModelForCausalLM", type("M", (), {"from_pretrained": lambda *a, **k: MockModelOutputs()}))
    res = pt_inf.generate_sql("model", "prompt", test_mode=False)
    assert res["status"] == "success"
    assert res["sql"] == "SELECT * FROM fallback"

    # Test fallback when generated text doesn't start with prompt
    class MockModelOutputsNoPrefix:
        """Test class for MockModelOutputsNoPrefix."""

        def __init__(self) -> None:
            """Initialize __init__."""
            self.device = "cpu"
            self.sequences = ["SELECT * FROM no_prefix"]

        def generate(self, **kwargs: object) -> object:
            """Execute generate helper."""
            return self

    class MockTokenizerNoPrefix(MockTokenizerNoShape):
        """Test class for MockTokenizerNoPrefix."""

        def decode(self, _tokens: object, skip_special_tokens: bool = True) -> str:
            """Execute decode helper."""
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
        """Test class for MockNativeTokenizer."""

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            """Execute from pretrained helper."""
            return cls()

        def __call__(self, _prompt: str, return_tensors: str = "pt") -> object:
            """Initialize __call__."""
            return type("Inputs", (), {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)})()

        def decode(self, _tokens: object, skip_special_tokens: bool = True) -> str:
            """Execute decode helper."""
            return "SELECT * FROM native_table"

    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockNativeTokenizer)
    res = pt_inf.generate_sql("model", "prompt", max_length=2, backend_alias="pytorch_native", config=tiny_cfg)
    assert res["status"] == "success"
    assert res["sql"] == "SELECT * FROM native_table"

    # Test AutoTokenizer raising ValueError
    class MockErrorTokenizer:
        """Test class for MockErrorTokenizer."""

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            """Execute from pretrained helper."""
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
        """Test class for MockTokenizerWithShape."""

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            """Execute from pretrained helper."""
            return cls()

        def __call__(self, _prompt: str, return_tensors: str = "pt") -> object:
            """Initialize __call__."""

            class Inputs(UserDict):
                """Test class for Inputs."""

                def __init__(self) -> None:
                    """Initialize __init__."""
                    super().__init__({"input_ids": torch.tensor([[1, 2]])})
                    self.input_ids = torch.tensor([[1, 2]])

                def to(self, _dev: object) -> object:
                    """Execute to helper."""
                    return self

            return Inputs()

        def decode(self, _tokens: object, skip_special_tokens: bool = True) -> str:
            """Execute decode helper."""
            return "SELECT * FROM sliced_tokens"

    class MockModelSlicing:
        """Test class for MockModelSlicing."""

        def __init__(self) -> None:
            """Initialize __init__."""
            self.device = "cpu"
            self.sequences = [torch.tensor([1, 2, 3, 4])]
            self.sequences_scores = [torch.tensor(0.95)]

        def generate(self, **kwargs: object) -> object:
            """Execute generate helper."""
            return self

    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockTokenizerWithShape)
    monkeypatch.setattr(pt_inf, "AutoModelForCausalLM", type("M", (), {"from_pretrained": lambda *a, **k: MockModelSlicing()}))
    res = pt_inf.generate_sql("model", "prompt", test_mode=False)
    assert res["status"] == "success"
    assert res["sql"] == "SELECT * FROM sliced_tokens"


def test_inference_pytorch_with_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch inference loading adapter_path."""
    import sys

    import torch

    import gemma_4_sql.backends.pytorch.inference as pt_inf

    class MockTokenizerWithShape:
        """Test class for MockTokenizerWithShape."""

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            """Execute from pretrained helper."""
            return cls()

        def __call__(self, _prompt: str, return_tensors: str = "pt") -> object:
            """Initialize __call__."""

            class Inputs(UserDict):
                """Test class for Inputs."""

                def __init__(self) -> None:
                    """Initialize __init__."""
                    super().__init__({"input_ids": torch.tensor([[1, 2]])})
                    self.input_ids = torch.tensor([[1, 2]])

                def to(self, _dev: object) -> object:
                    """Execute to helper."""
                    return self

            return Inputs()

        def decode(self, _tokens: object, skip_special_tokens: bool = True) -> str:
            """Execute decode helper."""
            return "SELECT * FROM adapter_tokens"

    class MockModelSlicing:
        """Test class for MockModelSlicing."""

        def __init__(self) -> None:
            """Initialize __init__."""
            self.device = "cpu"
            self.sequences = [torch.tensor([1, 2, 3, 4])]
            self.sequences_scores = [torch.tensor(0.95)]

        def generate(self, **kwargs: object) -> object:
            """Execute generate helper."""
            return self

    class MockPeftModel:
        """Test class for MockPeftModel."""

        @classmethod
        def from_pretrained(cls, model: object, path: str) -> object:
            """Execute from pretrained helper."""
            return model

    monkeypatch.setitem(sys.modules, "peft", type("PeftMod", (), {"PeftModel": MockPeftModel}))
    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockTokenizerWithShape)
    monkeypatch.setattr(pt_inf, "AutoModelForCausalLM", type("M", (), {"from_pretrained": lambda *a, **k: MockModelSlicing()}))
    res = pt_inf.generate_sql("model", "prompt", adapter_path="/fake/path", test_mode=False)
    assert res["status"] == "success"

    # Test adapter loading failure
    class FailingPeftModel:
        """Test class for FailingPeftModel."""

        @classmethod
        def from_pretrained(cls, model: object, path: str) -> object:
            """Execute from pretrained helper."""
            raise ValueError("bad adapter")

    monkeypatch.setitem(sys.modules, "peft", type("PeftMod", (), {"PeftModel": FailingPeftModel}))
    res_fail = pt_inf.generate_sql("model", "prompt", lora_path="/fake/bad/path", test_mode=False)
    assert res_fail["status"] == "success"


def test_inference_pytorch_empty_sql_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test InferenceError when Hugging Face generation decodes into empty SQL."""
    import torch

    class Inputs(UserDict):
        def __init__(self) -> None:
            super().__init__({"input_ids": torch.tensor([[1, 2]])})
            self.input_ids = torch.tensor([[1, 2]])

        def to(self, _dev: object) -> Inputs:
            return self

    class MockEmptyTokenizer:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            return cls()

        def __call__(self, _prompt: str, return_tensors: str = "pt") -> object:
            return Inputs()

        def decode(self, _tokens: object, skip_special_tokens: bool = True) -> str:
            return ""

    class MockEmptyModel:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            return cls()

        def generate(self, **kwargs: object) -> object:
            return type("Outputs", (), {"sequences": torch.tensor([[1, 2]]), "sequences_scores": [torch.tensor(-0.5)]})()

    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockEmptyTokenizer)
    monkeypatch.setattr(pt_inf, "AutoModelForCausalLM", MockEmptyModel)

    res = pt_inf.generate_sql("model", "prompt")
    assert "failed: PyTorch generation yielded an empty SQL sequence." in res["status"]


def test_inference_pytorch_native_empty_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test InferenceError when native generation produces 0 new tokens."""
    import torch

    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4Config

    class MockEmptyNativeModel:
        def eval(self) -> None:
            pass

        def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128) -> torch.Tensor:
            # Return same as input_ids (0 new tokens)
            return input_ids

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            return cls()

    monkeypatch.setattr(pt_inf, "AutoTokenizer", None)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM.from_pretrained", lambda *a, **k: MockEmptyNativeModel())

    tiny_cfg = Gemma4Config(vocab_size=128, hidden_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=32, intermediate_size=128)
    res = pt_inf.generate_sql("model", "prompt", backend_alias="pytorch_native", config=tiny_cfg)
    assert "failed: PyTorch native generation yielded an empty sequence." in res["status"]


def test_inference_pytorch_native_empty_decoded_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test InferenceError when native generation tokens decode to whitespace."""
    import torch

    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4Config

    class MockWhitespaceNativeModel:
        def eval(self) -> None:
            pass

        def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128) -> torch.Tensor:
            # 1 new token which is ASCII 32 (space)
            return torch.cat([input_ids, torch.tensor([[32]])], dim=-1)

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            return cls()

    monkeypatch.setattr(pt_inf, "AutoTokenizer", None)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM.from_pretrained", lambda *a, **k: MockWhitespaceNativeModel())

    tiny_cfg = Gemma4Config(vocab_size=128, hidden_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=32, intermediate_size=128)
    res = pt_inf.generate_sql("model", "prompt", backend_alias="pytorch_native", config=tiny_cfg)
    assert "failed: PyTorch native generation decoded into an empty SQL query string." in res["status"]


def test_inference_pytorch_no_sequences_scores_and_negative_logprob(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test confidence scoring when sequences_scores is absent or negative log prob."""
    import torch

    class Inputs(UserDict):
        def __init__(self) -> None:
            super().__init__({"input_ids": torch.tensor([[1, 2]])})
            self.input_ids = torch.tensor([[1, 2]])

        def to(self, _dev: object) -> Inputs:
            return self

    class MockTokenizer:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            return cls()

        def __call__(self, _prompt: str, return_tensors: str = "pt") -> object:
            return Inputs()

        def decode(self, _tokens: object, skip_special_tokens: bool = True) -> str:
            return "SELECT 1"

    # 1. Output without sequences_scores
    class MockModelNoScores:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            return cls()

        def generate(self, **kwargs: object) -> object:
            return type("Outputs", (), {"sequences": torch.tensor([[1, 2, 3]])})()

    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockTokenizer)
    monkeypatch.setattr(pt_inf, "AutoModelForCausalLM", MockModelNoScores)
    res_no_scores = pt_inf.generate_sql("model", "prompt")
    assert res_no_scores["status"] == "success"
    assert res_no_scores["confidence_score"] == pytest.approx(0.8)

    # 2. Output with negative log-prob
    class MockModelNegativeLogProb:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            return cls()

        def generate(self, **kwargs: object) -> object:
            return type("Outputs", (), {"sequences": torch.tensor([[1, 2, 3]]), "sequences_scores": [torch.tensor(-0.105)]})()

    monkeypatch.setattr(pt_inf, "AutoModelForCausalLM", MockModelNegativeLogProb)
    res_neg = pt_inf.generate_sql("model", "prompt")
    assert res_neg["status"] == "success"
    assert 0.85 <= res_neg["confidence_score"] <= 0.95


def test_inference_pytorch_multimodal_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test multimodal inference with audio_path, image_path, pixel_values, and audio_values.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    import torch

    img_file = tmp_path / "img.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n")
    aud_file = tmp_path / "aud.wav"
    aud_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")

    # 1. HF Pipeline with audio_path and image_path
    class Inputs(UserDict):
        """Mock inputs."""

        def __init__(self) -> None:
            """Initialize inputs."""
            super().__init__({"input_ids": torch.tensor([[1, 2]])})
            self.input_ids = torch.tensor([[1, 2]])

        def to(self, _dev: object) -> Inputs:
            """Transfer device."""
            return self

    class MockTokenizer:
        """Mock tokenizer."""

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            """Load from pretrained."""
            return cls()

        def __call__(self, _prompt: str, return_tensors: str = "pt") -> object:
            """Call tokenizer."""
            return Inputs()

        def decode(self, _tokens: object, skip_special_tokens: bool = True) -> str:
            """Decode tokens."""
            return "SELECT 1"

    captured_gen_kwargs: dict[str, object] = {}

    class MockHFModel:
        """Mock HuggingFace model."""

        device = "cpu"

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            """Load from pretrained."""
            return cls()

        def generate(self, **kwargs: object) -> object:
            """Generate tokens."""
            captured_gen_kwargs.update(kwargs)
            return type("Outputs", (), {"sequences": torch.tensor([[1, 2, 3]])})()

    class MockNativeModel:
        """Mock native model."""

        def eval(self) -> None:
            """Eval mode."""

        def generate(self, input_ids: object, **kwargs: object) -> object:
            """Generate tokens."""
            return torch.tensor([[1, 2, 3, 4]])

    monkeypatch.setattr(pt_inf, "AutoTokenizer", MockTokenizer)
    monkeypatch.setattr(pt_inf, "AutoModelForCausalLM", MockHFModel)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM.from_pretrained", lambda *a, **k: MockNativeModel())

    # Native with image only (covers 81->85)
    pt_inf._run_generation("model", "prompt", 1, 10, backend_alias="pytorch_native", test_mode=True, image_path=str(img_file), modality="vision")

    # Native with audio only (covers 77->81)
    pt_inf._run_generation("model", "prompt", 1, 10, backend_alias="pytorch_native", test_mode=True, audio_path=str(aud_file), modality="audio")

    # HF with image only (covers 148->152)
    pt_inf._run_generation("model", "Select users", 1, 10, backend_alias="hf", image_path=str(img_file), modality="vision")

    # HF with audio only (covers 144->148)
    pt_inf._run_generation("model", "Select users", 1, 10, backend_alias="hf", audio_path=str(aud_file), modality="audio")

    res_hf = pt_inf._run_generation(
        "model",
        "Select users",
        1,
        10,
        backend_alias="hf",
        image_path=str(img_file),
        audio_path=str(aud_file),
        modality="multimodal",
    )
    assert res_hf[0] == "SELECT 1"
    assert "pixel_values" in captured_gen_kwargs
    assert "audio_values" in captured_gen_kwargs

    # Test with pixel_values and audio_values already supplied (covers 77->81, 81->85)
    pv = torch.zeros((1, 3, 224, 224))
    av = torch.zeros((1, 1600))
    res_native = pt_inf._run_generation(
        "model",
        "prompt",
        1,
        10,
        backend_alias="pytorch_native",
        test_mode=True,
        image_path=str(img_file),
        audio_path=str(aud_file),
        pixel_values=pv,
        audio_values=av,
        modality="multimodal",
    )
    assert res_native[0] is not None
