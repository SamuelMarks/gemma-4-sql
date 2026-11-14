"""Tests for PyTorch export including _MODEL_CACHE and safetensors checkpoint restoration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from gemma_4_sql.exceptions import DependencyMissingError, ExportError
from tests.conftest import requires_torch
from tests.utils.tensor_helpers import torch_load_safetensors as load_file

pytestmark = requires_torch


class _TorchMarker:
    """Non-None marker object representing `torch` module during distributed tests."""


@pytest.fixture(autouse=True)
def _restore_export_mod(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore export_module globals after each test."""
    import gemma_4_sql.backends.pytorch.export as export_module

    monkeypatch.setattr(export_module, "torch", export_module.torch)
    monkeypatch.setattr(export_module, "save_file", export_module.save_file)


def test_is_rank_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _is_rank_zero under uninitialized, rank-0, and non-rank-0 distributed states."""
    import gemma_4_sql.backends.pytorch.export as export_module

    monkeypatch.setattr(export_module, "torch", None)
    assert export_module._is_rank_zero() is True

    monkeypatch.setattr(export_module, "torch", _TorchMarker())
    with patch("torch.distributed.is_initialized", return_value=False):
        assert export_module._is_rank_zero() is True

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=0),
    ):
        assert export_module._is_rank_zero() is True

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=1),
    ):
        assert export_module._is_rank_zero() is False


def test_save_real_model_cache_checkpoint_and_export_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify _save_real_model preserves cached weights and restores checkpoint weights via from_pretrained."""
    import gemma_4_sql.backends.pytorch.export as export_module
    from gemma_4_sql.backends.common_inference import make_model_cache_key
    from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
    from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM
    from gemma_4_sql.backends.pytorch.inference import _MODEL_CACHE

    _MODEL_CACHE.clear()
    export_dir = tmp_path / "export_native"
    tiny_cfg = Gemma4Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=1,
    )

    # 1. Train/modify weights on an in-memory model, cache it, and verify export saves exact cached weights
    # (not random init!).
    trained_model = Gemma4ForCausalLM(tiny_cfg)
    with torch.no_grad():
        trained_model.embed_tokens.weight.fill_(0.375)
    _MODEL_CACHE.put_model("cached_trained_model", (trained_model, None))

    res_cached = export_module.export_model(
        "cached_trained_model",
        str(export_dir),
        backend_alias="pytorch_native",
    )
    assert res_cached["status"] == "exported_with_safetensors"
    saved_st = Path(res_cached["file_path"])
    assert saved_st.is_file()
    loaded_from_cache_export = load_file(str(saved_st))
    assert torch.allclose(
        loaded_from_cache_export["embed_tokens.weight"], torch.full_like(trained_model.embed_tokens.weight, 0.375)
    )

    # 2. Keyed cache hit branch
    _MODEL_CACHE.clear()
    keyed_id = make_model_cache_key(
        "keyed_model",
        backend_name="native",
        config=tiny_cfg,
        factories=(Gemma4ForCausalLM.from_pretrained,),
    )
    _MODEL_CACHE.put_model(keyed_id, (trained_model, None))
    res_keyed = export_module.export_model(
        "keyed_model",
        str(tmp_path / "export_keyed"),
        backend_alias="pytorch_native",
        config=tiny_cfg,
    )
    assert res_keyed["status"] == "exported_with_safetensors"
    _MODEL_CACHE.clear()

    # 3. Export from an existing checkpoint directory restores saved weights via NativeGemma4.from_pretrained!
    re_export_dir = tmp_path / "re_exported_from_ckpt"
    res_ckpt = export_module.export_model(
        str(export_dir),
        str(re_export_dir),
        backend_alias="pytorch_native",
        config=tiny_cfg,
    )
    assert res_ckpt["status"] == "exported_with_safetensors"
    loaded_from_ckpt_export = load_file(str(res_ckpt["file_path"]))
    assert torch.allclose(
        loaded_from_ckpt_export["embed_tokens.weight"], torch.full_like(trained_model.embed_tokens.weight, 0.375)
    )

    # 4. Preset cold-start initialization AND non-from_pretrained constructor fallback
    res_preset = export_module.export_model(
        "gemma-4-tiny", str(tmp_path / "preset_out"), backend_alias="pytorch_native"
    )
    assert res_preset["status"] == "exported_with_safetensors"

    def _ctor_only(_cfg: Gemma4Config) -> Gemma4ForCausalLM:
        """Construct model without `.from_pretrained` attribute.

        Returns:
            Pre-initialized `trained_model` instance.
        """
        return trained_model

    with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM", _ctor_only):
        f_ctor, stat_ctor = export_module._save_real_model(
            "gemma-4-tiny",
            str(tmp_path / "ctor_out"),
            backend_alias="pytorch_native",
            config=tiny_cfg,
        )
        assert f_ctor.name == "model.safetensors" and stat_ctor == "exported_with_safetensors"

    # 5. Test HF branch via typed fake transformers.models.gemma4 module
    raise_hf_err = False

    class _FakeHFModel:
        """Typed HF causal LM double."""

        def state_dict(self) -> dict[str, torch.Tensor]:
            """Return state dict."""
            return {"a": torch.ones((2, 2)), "lm_head.weight": torch.ones((2, 2))}

        def save_pretrained(self, save_dir: str) -> str:
            """Return `save_dir`."""
            return save_dir

    class _FakeHFGemma4Cls:
        """Typed HF `Gemma4ForCausalLM` loader double."""

        @staticmethod
        def from_pretrained(_name: str) -> _FakeHFModel:
            """Return `_FakeHFModel` or raise `ValueError`.

            Raises:
                ValueError: If `raise_hf_err` is True.
            """
            if raise_hf_err:
                raise ValueError("error")
            return _FakeHFModel()

    class _FakeHFModule:
        """Typed `transformers.models.gemma4` module double."""

        Gemma4ForCausalLM = _FakeHFGemma4Cls

    with patch.dict(sys.modules, {"transformers.models.gemma4": _FakeHFModule}):
        f_hf, stat_hf = export_module._save_real_model("m", str(export_dir), backend_alias="pytorch_hf")
        assert f_hf.name == "model.safetensors"
        assert stat_hf == "exported_with_safetensors"

        _, stat_skip = export_module._save_real_model(
            "m", str(export_dir), is_rank_zero=False, backend_alias="pytorch_hf"
        )
        assert stat_skip == "skipped_non_rank_zero"

        f_adap, _ = export_module._save_real_model(
            "m", str(export_dir), export_type="adapter", backend_alias="pytorch_hf"
        )
        assert f_adap.name == "adapter_model.safetensors"

        monkeypatch.setattr(export_module, "save_file", None)
        _, stat_nosave = export_module._save_real_model("m", str(export_dir), backend_alias="pytorch_hf")
        assert stat_nosave == "exported_with_safetensors"

        raise_hf_err = True
        with pytest.raises(ExportError, match="Failed to load model m"):
            export_module._save_real_model("m", str(export_dir), backend_alias="pytorch_hf")

    with (
        patch.dict(sys.modules, {"transformers.models.gemma4": None}),
        pytest.raises(ExportError, match=r"Gemma4ForCausalLM is not available"),
    ):
        export_module._save_real_model("m", str(export_dir), backend_alias="pytorch_hf")

    with pytest.raises(DependencyMissingError):
        export_module.export_model("m", "export")

    monkeypatch.setattr(export_module, "torch", None)
    with pytest.raises(DependencyMissingError):
        export_module.export_model("m", "export")


def test_is_rank_zero_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _is_rank_zero returns True when torch.distributed.is_initialized raises RuntimeError."""
    import gemma_4_sql.backends.pytorch.export as pexp

    monkeypatch.setattr(pexp, "torch", _TorchMarker())
    with patch("torch.distributed.is_initialized", side_effect=RuntimeError("mocked error")):
        assert pexp._is_rank_zero() is True


def test_is_rank_zero_get_rank_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _is_rank_zero returns True when torch.distributed.get_rank raises RuntimeError."""
    import torch.distributed as dist

    import gemma_4_sql.backends.pytorch.export as pexp

    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def _raise_rank() -> int:
        """Raise RuntimeError.

        Raises:
            RuntimeError: Always raised to simulate distributed rank failure.
        """
        raise RuntimeError("mocked error")

    monkeypatch.setattr(dist, "get_rank", _raise_rank)
    assert pexp._is_rank_zero() is True
