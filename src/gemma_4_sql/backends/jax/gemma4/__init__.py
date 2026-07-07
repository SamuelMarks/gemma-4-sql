"""Core functionality for the __init__ module."""

from __future__ import annotations

from ...lazy_loader import catch_optional_imports

with catch_optional_imports():
    from .cache import Cache, LayerCache, init_cache
    from .modeling import Gemma4ForCausalLM, Gemma4Model, forward
    from .modeling import ModelConfig as Gemma4Config
    from .params import create_gemma4_from_pretrained

__all__ = ["Cache", "Gemma4Config", "Gemma4ForCausalLM", "Gemma4Model", "LayerCache", "create_gemma4_from_pretrained", "forward", "init_cache"]
