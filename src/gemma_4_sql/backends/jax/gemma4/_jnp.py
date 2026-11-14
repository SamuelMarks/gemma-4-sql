"""Typed `jax.numpy` accessor for `gemma4` JAX modules and JAX test suites."""

from __future__ import annotations

from gemma_4_sql.backends.jax._protocols import TypedJaxNumpyProtocol
from gemma_4_sql.backends.lazy_loader import optional_import

_loaded_jnp = optional_import("jax.numpy", TypedJaxNumpyProtocol)
assert _loaded_jnp is not None
jnp: TypedJaxNumpyProtocol = _loaded_jnp

__all__ = ["jnp"]
