"""Behavioral and unit tests for PyTorch inference without untyped mocks."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import patch

import pytest
import torch

from gemma_4_sql.backends.pytorch.gemma4 import Gemma4Config, Gemma4ForCausalLM
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError
from tests.conftest import requires_torch

pytestmark = requires_torch


@pytest.fixture(autouse=True)
def _restore_inf_mod(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore `inf_module` globals after each test."""
    import gemma_4_sql.backends.pytorch.inference as inf_module

    monkeypatch.setattr(inf_module, "torch", inf_module.torch)
    monkeypatch.setattr(inf_module, "AutoModelForCausalLM", inf_module.AutoModelForCausalLM)
    monkeypatch.setattr(inf_module, "AutoTokenizer", inf_module.AutoTokenizer)


def _make_tiny_native_model() -> Gemma4ForCausalLM:
    """Create a minimal native `Gemma4ForCausalLM` instance for fast CPU testing.

    Returns:
        Minimal `Gemma4ForCausalLM` instance.
    """
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


def _make_from_pretrained_stub(model: Gemma4ForCausalLM) -> classmethod[Gemma4ForCausalLM, ..., Gemma4ForCausalLM]:
    """Create a typed `from_pretrained` classmethod returning `model`.

    Returns:
        Classmethod stub returning `model`.
    """

    def _from_pretrained(
        _cls: type[Gemma4ForCausalLM], *_a: str, **_k: Gemma4Config | bool | None
    ) -> Gemma4ForCausalLM:
        """Return `model`."""
        return model

    return classmethod(_from_pretrained)


def test_real_tiny_pytorch_native_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `generate_sql` runs real token generation using a tiny native `Gemma4ForCausalLM`."""
    import gemma_4_sql.backends.pytorch.inference as inf_module

    tiny_model = _make_tiny_native_model()
    monkeypatch.setattr(Gemma4ForCausalLM, "from_pretrained", _make_from_pretrained_stub(tiny_model))
    monkeypatch.setattr(inf_module, "AutoTokenizer", None)

    res = inf_module.generate_sql("tiny-native", "SELECT 1", beam_width=1, max_length=2, backend_alias="pytorch_native")
    assert res["status"] == "success"
    assert isinstance(res["sql"], str)
    assert len(str(res["sql"])) > 0


def test_run_generation_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_run_generation` native path with real torch tensors and tokenizer fallbacks."""
    import gemma_4_sql.backends.pytorch.inference as inf_module

    tiny_model = _make_tiny_native_model()
    monkeypatch.setattr(Gemma4ForCausalLM, "from_pretrained", _make_from_pretrained_stub(tiny_model))

    class StubTokInputs:
        """Minimal namespace holding `input_ids` tensor."""

        def __init__(self, ids: torch.Tensor) -> None:
            """Initialize `StubTokInputs`."""
            self.input_ids = ids

    class StubTokenizer:
        """Minimal tokenizer returning real PyTorch tensors."""

        def __init__(self, decoded: str = "SELECT 1;") -> None:
            """Initialize stub tokenizer with target decoded output."""
            self.decoded = decoded

        def __call__(self, _prompt: str, return_tensors: str = "pt") -> StubTokInputs:
            """Encode prompt into real tensor `input_ids`.

            Returns:
                Tokenized `StubTokInputs` wrapper.
            """
            _ = return_tensors
            return StubTokInputs(torch.tensor([[10, 20]], dtype=torch.long))

        def decode(self, _tokens: torch.Tensor | Sequence[int], skip_special_tokens: bool = True) -> str:
            """Decode generated tokens into configured SQL string.

            Returns:
                Decoded SQL string.
            """
            _ = skip_special_tokens
            return self.decoded

    current_tok: StubTokenizer | None = StubTokenizer("SELECT 1;")

    class _FakeTokLoader:
        """Typed `AutoTokenizer` loader double."""

        @classmethod
        def from_pretrained(cls, _name: str) -> StubTokenizer:
            """Return `current_tok` or raise `OSError` when `None`.

            Raises:
                OSError: If `current_tok` is `None`.
            """
            if current_tok is None:
                raise OSError("missing local tokenizer")
            return current_tok

    monkeypatch.setattr(inf_module, "AutoTokenizer", _FakeTokLoader)

    sql, conf = inf_module._run_generation("model", "prompt", 1, 2, backend_alias="pytorch_native")
    assert sql == "SELECT 1;"
    assert 0.0 < conf <= 1.0

    # Test empty decoded string raises InferenceError
    current_tok = StubTokenizer("")
    with pytest.raises(InferenceError, match="empty SQL"):
        inf_module._run_generation("model", "prompt", 1, 2, backend_alias="pytorch_native")

    # Test empty sequence when max_new_tokens=0
    current_tok = StubTokenizer("SELECT 1;")
    with pytest.raises(InferenceError, match="yielded an empty sequence"):
        inf_module._run_generation("model", "prompt", 1, 0, backend_alias="pytorch_native")

    # Test AutoTokenizer.from_pretrained raising OSError falls back to SQLTokenizer
    current_tok = None
    sql_fb, conf_fb = inf_module._run_generation("model", "SELECT 1", 1, 2, backend_alias="pytorch_native")
    assert len(sql_fb) > 0
    assert 0.0 < conf_fb <= 1.0


def test_run_generation_native_multimodal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_run_generation` native multimodal preprocessing with real PyTorch tensors."""
    import gemma_4_sql.backends.pytorch.inference as inf_module

    tiny_model = _make_tiny_native_model()
    monkeypatch.setattr(Gemma4ForCausalLM, "from_pretrained", _make_from_pretrained_stub(tiny_model))
    monkeypatch.setattr(inf_module, "AutoTokenizer", None)

    orig_gen = tiny_model.generate
    recorded_gen_kwargs: list[dict[str, torch.Tensor | int | None]] = []

    def _spy_generate(
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 50,
        min_new_tokens: int = 0,
        pixel_values: torch.Tensor | None = None,
        audio_values: torch.Tensor | None = None,
        **_extra: float | bool | None,
    ) -> torch.Tensor:
        """Record multimodal kwargs and forward to real `tiny_model.generate`.

        Returns:
            Generated token ID tensor.
        """
        recorded_gen_kwargs.append({"pixel_values": pixel_values, "audio_values": audio_values})
        return orig_gen(
            input_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            pixel_values=pixel_values,
            audio_values=audio_values,
        )

    with (
        patch("gemma_4_sql.backends.common_multimodal.process_image", return_value={"pixel_values": [[0.1, 0.2]]}),
        patch("gemma_4_sql.backends.common_multimodal.process_audio", return_value={"audio_values": [[0.3, 0.4]]}),
        patch.object(tiny_model, "generate", side_effect=_spy_generate),
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
        assert len(recorded_gen_kwargs) == 1
        assert isinstance(recorded_gen_kwargs[0]["pixel_values"], torch.Tensor)
        assert isinstance(recorded_gen_kwargs[0]["audio_values"], torch.Tensor)

        # Image-only and audio-only branches
        inf_module._run_generation("model", "SELECT 1", 1, 2, backend_alias="pytorch_native", image_path="ip.png")
        inf_module._run_generation("model", "SELECT 1", 1, 2, backend_alias="pytorch_native", audio_path="ap.wav")


def test_run_generation_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_run_generation` HuggingFace path using real `torch.Tensor` sequences and scores."""
    import gemma_4_sql.backends.pytorch.inference as inf_module

    class StubHFOutput:
        """Container matching HuggingFace `generate(return_dict_in_generate=True, output_scores=True)`."""

        def __init__(self, seq: torch.Tensor, score: float) -> None:
            """Initialize with sequence tensor and scalar sequence score."""
            self.sequences = seq
            self.sequences_scores = torch.tensor([score], dtype=torch.float32)

    class StubHFModel(torch.nn.Module):
        """Minimal `torch.nn.Module` stub mimicking `AutoModelForCausalLM`."""

        def __init__(self, score: float = 0.5) -> None:
            """Initialize with a parameter tensor and target score."""
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.zeros(1))
            self.score = score

        def generate(self, **_kwargs: torch.Tensor | int | bool) -> StubHFOutput:
            """Return a real 2D token tensor and score."""
            return StubHFOutput(torch.tensor([[1, 2, 65, 66]], dtype=torch.long), self.score)

    class StubBatchEncoding(dict[str, torch.Tensor]):
        """Minimal dict with `.input_ids` and `.to(device)` matching HF `BatchEncoding`."""

        def __init__(self) -> None:
            """Initialize with `input_ids` tensor."""
            super().__init__()
            t = torch.tensor([[1, 2]], dtype=torch.long)
            self["input_ids"] = t
            self.input_ids = t

        def to(self, _device: torch.device | str) -> StubBatchEncoding:
            """Return self on device transfer."""
            return self

    stub_model = StubHFModel(score=0.5)
    assert inf_module._resolve_model_device(torch.nn.Identity()) == "cpu"

    class _EmptyScoresOut:
        """HF output with empty `sequences_scores`."""

        sequences = torch.tensor([[1, 2, 65, 66]], dtype=torch.long)
        sequences_scores: tuple[()] = ()

    class _EmptyScoresModel:
        """HF model returning empty `sequences_scores`."""

        def generate(self, **_kw: torch.Tensor | int | bool) -> _EmptyScoresOut:
            """Return `_EmptyScoresOut`."""
            return _EmptyScoresOut()

    active_hf_model: StubHFModel | _EmptyScoresModel = stub_model

    class _FakeAutoModel:
        """Typed `AutoModelForCausalLM` loader double."""

        @classmethod
        def from_pretrained(cls, _name: str, **_kw: str | torch.dtype) -> StubHFModel | _EmptyScoresModel:
            """Return `active_hf_model`."""
            return active_hf_model

    tok_return_val: StubBatchEncoding | dict[str, torch.Tensor] = StubBatchEncoding()
    tok_decoded_str = "SELECT 1;"

    class _FakeHFTokenizer:
        """Typed HF tokenizer double."""

        def __call__(self, _prompt: str, return_tensors: str = "pt") -> StubBatchEncoding | dict[str, torch.Tensor]:
            """Return `tok_return_val`."""
            _ = return_tensors
            return tok_return_val

        def decode(self, _tokens: torch.Tensor | Sequence[int], skip_special_tokens: bool = True) -> str:
            """Return `tok_decoded_str`."""
            _ = skip_special_tokens
            return tok_decoded_str

    fake_tok = _FakeHFTokenizer()

    class _FakeAutoTokenizer:
        """Typed `AutoTokenizer` loader double."""

        @classmethod
        def from_pretrained(cls, _name: str) -> _FakeHFTokenizer:
            """Return `fake_tok`."""
            return fake_tok

    monkeypatch.setattr(inf_module, "AutoModelForCausalLM", _FakeAutoModel)
    monkeypatch.setattr(inf_module, "AutoTokenizer", _FakeAutoTokenizer)

    sql, conf = inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch")
    assert sql == "SELECT 1;"
    assert conf == 0.5

    active_hf_model = _EmptyScoresModel()
    _, conf_empty = inf_module._run_generation("model_empty_scores", "prompt", 1, 10, backend_alias="pytorch")
    assert conf_empty == pytest.approx(0.8)
    active_hf_model = stub_model

    # Adapter load success, cache hit on second call, and fallback when peft is unavailable
    peft_calls = 0

    class _FakePeftCls:
        """Typed `PeftModel` loader double."""

        @classmethod
        def from_pretrained(cls, _base: StubHFModel | _EmptyScoresModel, _adapter: str) -> StubHFModel:
            """Increment `peft_calls` and return `stub_model`.

            Returns:
                Pre-configured `stub_model` instance.
            """
            nonlocal peft_calls
            peft_calls += 1
            return stub_model

    with patch.object(inf_module, "import_optional_symbol", return_value=_FakePeftCls):
        inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch", adapter_path="p")
        inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch", adapter_path="p")
        assert peft_calls == 1
    with patch.object(inf_module, "import_optional_symbol", return_value=None):
        inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch", adapter_path="p_missing")

    # Negative log-probability score branch (raw_score <= 0.0 -> exp(raw_score))
    stub_model.score = -0.5
    _sql_neg, conf_neg = inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch")
    assert 0.0 < conf_neg < 1.0

    # Empty SQL raises InferenceError
    tok_decoded_str = ""
    with pytest.raises(InferenceError, match="empty SQL"):
        inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch")

    # Fallback when inputs has no input_ids attribute and no .to() method
    tok_return_val = {}
    tok_decoded_str = "promptSELECT 2;"
    sql_no_ids, _ = inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch")
    assert sql_no_ids == "SELECT 2;"


def test_run_generation_hf_multimodal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_run_generation` HuggingFace multimodal path attaches real pixel and audio tensors."""
    import gemma_4_sql.backends.pytorch.inference as inf_module

    captured_kwargs: dict[str, torch.Tensor | int | bool] = {}

    class _ScoredOutput:
        """Typed HF generation output with `.sequences` and `.sequences_scores`."""

        def __init__(self) -> None:
            """Initialize output tensors."""
            self.sequences = torch.tensor([[1, 2, 3]], dtype=torch.long)
            self.sequences_scores = torch.tensor([0.9], dtype=torch.float32)

    class StubHFModel(torch.nn.Module):
        """Minimal HF model stub recording `generate` kwargs."""

        def __init__(self) -> None:
            """Initialize with dummy parameter."""
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def generate(self, **kwargs: torch.Tensor | int | bool) -> _ScoredOutput:
            """Record kwargs and return `_ScoredOutput`.

            Returns:
                Scored generation output stub.
            """
            captured_kwargs.update(kwargs)
            return _ScoredOutput()

    class StubInputs(dict[str, torch.Tensor]):
        """Minimal `BatchEncoding` stub."""

        def __init__(self) -> None:
            """Initialize with 2-token `input_ids`."""
            super().__init__()
            self.input_ids = torch.tensor([[1, 2]], dtype=torch.long)
            self["input_ids"] = self.input_ids

        def to(self, _dev: torch.device | str) -> StubInputs:
            """Return self."""
            return self

    hf_mm_model = StubHFModel()

    class _FakeAutoModel:
        """Typed `AutoModelForCausalLM` loader double."""

        @classmethod
        def from_pretrained(cls, _name: str, **_kw: str | torch.dtype) -> StubHFModel:
            """Return `hf_mm_model`."""
            return hf_mm_model

    class _FakeTok:
        """Typed HF tokenizer double."""

        def __call__(self, _prompt: str, return_tensors: str = "pt") -> StubInputs:
            """Return `StubInputs`."""
            _ = return_tensors
            return StubInputs()

        def decode(self, _tokens: torch.Tensor | Sequence[int], skip_special_tokens: bool = True) -> str:
            """Return `'SELECT 1;'`."""
            _ = skip_special_tokens
            return "SELECT 1;"

    class _FakeAutoTok:
        """Typed `AutoTokenizer` loader double."""

        @classmethod
        def from_pretrained(cls, _name: str) -> _FakeTok:
            """Return `_FakeTok`."""
            return _FakeTok()

    monkeypatch.setattr(inf_module, "AutoModelForCausalLM", _FakeAutoModel)
    monkeypatch.setattr(inf_module, "AutoTokenizer", _FakeAutoTok)

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
    """Verify `generate_sql` end-to-end wrapper, error handling, and missing dependency check."""
    import gemma_4_sql.backends.pytorch.inference as inf_module

    tiny_model = _make_tiny_native_model()
    monkeypatch.setattr(Gemma4ForCausalLM, "from_pretrained", _make_from_pretrained_stub(tiny_model))
    monkeypatch.setattr(inf_module, "AutoTokenizer", None)

    res = inf_module.generate_sql("model", "SELECT 1", max_length=2, backend_alias="pytorch_native")
    assert res["status"] == "success"
    assert isinstance(res["sql"], str) and len(str(res["sql"])) > 0

    def _raise_static(*_a: str, **_kw: str | Gemma4Config | None) -> Gemma4ForCausalLM:
        """Raise `RuntimeError` to test failure status.

        Raises:
            RuntimeError: Always raised to test error handling.

        """
        raise RuntimeError("boom")

    monkeypatch.setattr(Gemma4ForCausalLM, "from_pretrained", staticmethod(_raise_static))
    res_err = inf_module.generate_sql("model", "SELECT 1", backend_alias="pytorch_native")
    assert "failed: boom" in str(res_err["status"])

    monkeypatch.setattr(inf_module, "torch", None)
    with pytest.raises(DependencyMissingError):
        inf_module.generate_sql("model", "prompt", backend_alias="pytorch")
