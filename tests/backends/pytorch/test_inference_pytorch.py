"""Behavioral and unit tests for PyTorch inference."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, InferenceError
from tests.conftest import requires_torch

pytestmark = requires_torch


@pytest.fixture(autouse=True)
def _restore_inf_mod(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore inf_module globals after each test."""
    import gemma_4_sql.backends.pytorch.inference as inf_module

    monkeypatch.setattr(inf_module, "torch", inf_module.torch)
    monkeypatch.setattr(inf_module, "AutoModelForCausalLM", inf_module.AutoModelForCausalLM)
    monkeypatch.setattr(inf_module, "AutoTokenizer", inf_module.AutoTokenizer)


def _make_tiny_native_model() -> Any:
    """Create a minimal native Gemma4ForCausalLM instance for fast CPU testing."""
    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4Config, Gemma4ForCausalLM

    cfg = Gemma4Config(
        vocab_size=320,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
        layer_types=["sliding_attention"],
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
    )
    return Gemma4ForCausalLM(cfg)


def test_real_tiny_pytorch_native_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify generate_sql runs real token generation using a tiny native Gemma4ForCausalLM."""
    import gemma_4_sql.backends.pytorch.inference as inf_module
    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4ForCausalLM

    tiny_model = _make_tiny_native_model()
    monkeypatch.setattr(Gemma4ForCausalLM, "from_pretrained", classmethod(lambda _cls, *_a, **_k: tiny_model))
    monkeypatch.setattr(inf_module, "AutoTokenizer", None)

    res = inf_module.generate_sql("tiny-native", "SELECT 1", beam_width=1, max_length=2, backend_alias="pytorch_native")
    assert res["status"] == "success"
    assert isinstance(res["sql"], str)
    assert len(str(res["sql"])) > 0


def test_run_generation_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_generation native path with real torch tensors and tokenizer fallbacks."""
    import torch

    import gemma_4_sql.backends.pytorch.inference as inf_module
    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4ForCausalLM

    tiny_model = _make_tiny_native_model()
    monkeypatch.setattr(Gemma4ForCausalLM, "from_pretrained", classmethod(lambda _cls, *_a, **_k: tiny_model))

    class StubTokenizer:
        """Minimal tokenizer returning real PyTorch tensors."""

        def __init__(self, decoded: str = "SELECT 1;") -> None:
            """Initialize stub tokenizer with target decoded output."""
            self.decoded = decoded

        def __call__(self, _prompt: str, return_tensors: str = "pt") -> Any:
            """Encode prompt into real tensor input_ids."""
            _ = return_tensors
            ns = MagicMock()
            ns.input_ids = torch.tensor([[10, 20]], dtype=torch.long)
            return ns

        def decode(self, _tokens: Any, skip_special_tokens: bool = True) -> str:
            """Decode generated tokens into configured SQL string."""
            _ = skip_special_tokens
            return self.decoded

    mock_tok_cls = MagicMock()
    mock_tok_cls.from_pretrained.return_value = StubTokenizer("SELECT 1;")
    monkeypatch.setattr(inf_module, "AutoTokenizer", mock_tok_cls)

    sql, conf = inf_module._run_generation("model", "prompt", 1, 2, backend_alias="pytorch_native")
    assert sql == "SELECT 1;"
    assert conf == 0.95

    # Test empty decoded string raises InferenceError
    mock_tok_cls.from_pretrained.return_value = StubTokenizer("")
    with pytest.raises(InferenceError, match="empty SQL"):
        inf_module._run_generation("model", "prompt", 1, 2, backend_alias="pytorch_native")

    # Test empty sequence when max_new_tokens=0
    mock_tok_cls.from_pretrained.return_value = StubTokenizer("SELECT 1;")
    with pytest.raises(InferenceError, match="yielded an empty sequence"):
        inf_module._run_generation("model", "prompt", 1, 0, backend_alias="pytorch_native")

    # Test AutoTokenizer.from_pretrained raising OSError falls back to SQLTokenizer
    mock_tok_cls.from_pretrained.side_effect = OSError("missing local tokenizer")
    sql_fb, conf_fb = inf_module._run_generation("model", "SELECT 1", 1, 2, backend_alias="pytorch_native")
    assert len(sql_fb) > 0
    assert conf_fb == 0.95


def test_run_generation_native_multimodal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_generation native multimodal preprocessing with real PyTorch tensors."""
    import gemma_4_sql.backends.pytorch.inference as inf_module
    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4ForCausalLM

    tiny_model = _make_tiny_native_model()
    monkeypatch.setattr(Gemma4ForCausalLM, "from_pretrained", classmethod(lambda _cls, *_a, **_k: tiny_model))
    monkeypatch.setattr(inf_module, "AutoTokenizer", None)

    with (
        patch("gemma_4_sql.backends.common_multimodal.process_image", return_value={"pixel_values": [[0.1, 0.2]]}),
        patch("gemma_4_sql.backends.common_multimodal.process_audio", return_value={"audio_values": [[0.3, 0.4]]}),
        patch.object(tiny_model, "generate", wraps=tiny_model.generate) as spy_gen,
    ):
        sql, _conf = inf_module._run_generation(
            "model",
            "SELECT 1",
            1,
            2,
            backend_alias="pytorch_native",
            image_path="ip.png",
            audio_path="ap.wav",
        )
        assert len(sql) > 0
        spy_gen.assert_called_once()
        assert "pixel_values" in spy_gen.call_args.kwargs
        assert "audio_values" in spy_gen.call_args.kwargs

        # Image-only and audio-only branches
        inf_module._run_generation("model", "SELECT 1", 1, 2, backend_alias="pytorch_native", image_path="ip.png")
        inf_module._run_generation("model", "SELECT 1", 1, 2, backend_alias="pytorch_native", audio_path="ap.wav")


def test_run_generation_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_generation HuggingFace path using real torch.Tensor sequences and scores."""
    import torch

    import gemma_4_sql.backends.pytorch.inference as inf_module

    class StubHFOutput:
        """Container matching HuggingFace generate(return_dict_in_generate=True, output_scores=True)."""

        def __init__(self, seq: torch.Tensor, score: float) -> None:
            """Initialize with sequence tensor and scalar sequence score."""
            self.sequences = seq
            self.sequences_scores = torch.tensor([score], dtype=torch.float32)

    class StubHFModel(torch.nn.Module):
        """Minimal torch.nn.Module stub mimicking AutoModelForCausalLM."""

        def __init__(self, score: float = 0.5) -> None:
            """Initialize with a parameter tensor and target score."""
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.zeros(1))
            self.score = score

        def generate(self, **_kwargs: Any) -> StubHFOutput:
            """Return a real 2D token tensor and score."""
            return StubHFOutput(torch.tensor([[1, 2, 65, 66]], dtype=torch.long), self.score)

    class StubBatchEncoding(dict[str, Any]):
        """Minimal dict with .input_ids and .to(device) matching HF BatchEncoding."""

        def __init__(self, with_input_ids: bool = True) -> None:
            """Initialize with optional input_ids tensor."""
            super().__init__()
            if with_input_ids:
                t = torch.tensor([[1, 2]], dtype=torch.long)
                self["input_ids"] = t
                self.input_ids = t

        def to(self, _device: Any) -> StubBatchEncoding:
            """Return self on device transfer."""
            return self

    stub_model = StubHFModel(score=0.5)
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = stub_model
    monkeypatch.setattr(inf_module, "AutoModelForCausalLM", mock_auto_model)

    mock_tok = MagicMock()
    mock_tok.return_value = StubBatchEncoding(with_input_ids=True)
    mock_tok.decode.return_value = "SELECT 1;"
    mock_auto_tok = MagicMock()
    mock_auto_tok.from_pretrained.return_value = mock_tok
    monkeypatch.setattr(inf_module, "AutoTokenizer", mock_auto_tok)

    sql, conf = inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch")
    assert sql == "SELECT 1;"
    assert conf == 0.5

    # Adapter load success and fallback when peft is unavailable
    mock_peft_cls = MagicMock()
    mock_peft_cls.from_pretrained.return_value = stub_model
    with patch.object(inf_module, "optional_attr", return_value=mock_peft_cls):
        inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch", adapter_path="p")
    with patch.object(inf_module, "optional_attr", return_value=None):
        inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch", adapter_path="p")

    # Negative log-probability score branch (raw_score <= 0.0 -> exp(raw_score))
    stub_model.score = -0.5
    _sql_neg, conf_neg = inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch")
    assert 0.0 < conf_neg < 1.0

    # Empty SQL raises InferenceError
    mock_tok.decode.return_value = ""
    with pytest.raises(InferenceError, match="empty SQL"):
        inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch")

    # Fallback when inputs has no input_ids attribute and no .to() method
    mock_tok.return_value = {}
    mock_tok.decode.return_value = "promptSELECT 2;"
    sql_no_ids, _ = inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch")
    assert sql_no_ids == "SELECT 2;"


def test_run_generation_hf_multimodal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _run_generation HuggingFace multimodal path attaches real pixel and audio tensors."""
    import torch

    import gemma_4_sql.backends.pytorch.inference as inf_module

    captured_kwargs: dict[str, Any] = {}

    class StubHFModel(torch.nn.Module):
        """Minimal HF model stub recording generate kwargs."""

        def __init__(self) -> None:
            """Initialize with dummy parameter."""
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def generate(self, **kwargs: Any) -> Any:
            """Record kwargs and return tensor sequences."""
            captured_kwargs.update(kwargs)
            out = MagicMock()
            out.sequences = torch.tensor([[1, 2, 3]], dtype=torch.long)
            out.sequences_scores = torch.tensor([0.9], dtype=torch.float32)
            return out

    class StubInputs(dict[str, Any]):
        """Minimal BatchEncoding stub."""

        def __init__(self) -> None:
            """Initialize with 2-token input_ids."""
            super().__init__()
            self.input_ids = torch.tensor([[1, 2]], dtype=torch.long)
            self["input_ids"] = self.input_ids

        def to(self, _dev: Any) -> StubInputs:
            """Return self."""
            return self

    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = StubHFModel()
    monkeypatch.setattr(inf_module, "AutoModelForCausalLM", mock_auto_model)

    mock_tok = MagicMock()
    mock_tok.return_value = StubInputs()
    mock_tok.decode.return_value = "SELECT 1;"
    mock_auto_tok = MagicMock()
    mock_auto_tok.from_pretrained.return_value = mock_tok
    monkeypatch.setattr(inf_module, "AutoTokenizer", mock_auto_tok)

    with (
        patch("gemma_4_sql.backends.common_multimodal.process_image", return_value={"pixel_values": [[0.1, 0.2]]}),
        patch("gemma_4_sql.backends.common_multimodal.process_audio", return_value={"audio_values": [[0.3, 0.4]]}),
    ):
        sql, _conf = inf_module._run_generation(
            "model",
            "prompt",
            1,
            10,
            backend_alias="pytorch",
            image_path="ip.png",
            audio_path="ap.wav",
        )
        assert sql == "SELECT 1;"
        assert isinstance(captured_kwargs["pixel_values"], torch.Tensor)
        assert isinstance(captured_kwargs["audio_values"], torch.Tensor)

        # Image-only and audio-only HF branches
        inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch", image_path="ip.png")
        inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch", audio_path="ap.wav")


def test_generate_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify generate_sql end-to-end wrapper, error handling, and missing dependency check."""
    import gemma_4_sql.backends.pytorch.inference as inf_module
    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4ForCausalLM

    tiny_model = _make_tiny_native_model()
    monkeypatch.setattr(Gemma4ForCausalLM, "from_pretrained", classmethod(lambda _cls, *_a, **_k: tiny_model))
    monkeypatch.setattr(inf_module, "AutoTokenizer", None)

    res = inf_module.generate_sql("model", "SELECT 1", max_length=2, backend_alias="pytorch_native")
    assert res["status"] == "success"
    assert isinstance(res["sql"], str) and len(str(res["sql"])) > 0

    # Error handling when model loading raises RuntimeError
    def _raise_err(*_a: Any, **_k: Any) -> Any:
        """Raise RuntimeError to test failure status."""
        raise RuntimeError("boom")

    monkeypatch.setattr(Gemma4ForCausalLM, "from_pretrained", classmethod(_raise_err))
    res_err = inf_module.generate_sql("model", "SELECT 1", backend_alias="pytorch_native")
    assert "failed: boom" in str(res_err["status"])

    monkeypatch.setattr(inf_module, "torch", None)
    with pytest.raises(DependencyMissingError):
        inf_module.generate_sql("model", "prompt", backend_alias="pytorch")
