"""Module docstring."""

import builtins
import importlib

import gemma_4_sql.backends.maxtext.peft as mod


def test_maxtext_peft_import_error():
    """Docstring for test_maxtext_peft_import_error."""
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "jax" or name == "numpy":
            raise ImportError("mock")
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        importlib.reload(mod)
        assert mod.jax is None
    finally:
        builtins.__import__ = orig_import
        importlib.reload(mod)


def test_maxtext_peft_import_error_gemma():
    """Docstring for test_maxtext_peft_import_error_gemma."""
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if "maxtext.models.gemma4" in name:
            raise ImportError("mock")
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        importlib.reload(mod)
        assert mod.Gemma4Model is None
    finally:
        builtins.__import__ = orig_import
        importlib.reload(mod)


def test_maxtext_peft_inject_branches():
    """Docstring for test_maxtext_peft_inject_branches."""
    import jax.numpy as jnp

    import gemma_4_sql.backends.maxtext.peft as mod

    # 118->116 ? Just run _inject with something that hits the else branch
    params = {"a": {"b": jnp.zeros((2, 2))}, "c": jnp.zeros((2,))}
    mod.transform_params_to_lora(params, ["nonexistent"])


def test_merge_lora_weights_branches():
    """Docstring for test_merge_lora_weights_branches."""
    import jax.numpy as jnp

    import gemma_4_sql.backends.maxtext.peft as mod

    assert mod.merge_lora_weights("not_a_dict") == "not_a_dict"

    params = {"a": {"kernel": jnp.zeros((2, 2)), "lora_a": jnp.zeros((2, 2)), "lora_b": jnp.zeros((2, 2)), "lora_scale": 1.0, "extra_key": jnp.zeros((1,))}}
    mod.merge_lora_weights(params)
