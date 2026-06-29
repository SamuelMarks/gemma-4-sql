"""Module docstring."""

import gemma_4_sql.backends.maxtext.quantize as q_maxtext


def test_maxtext_quantize_present() -> object:  # type: ignore[return]
    """Initialize function test_maxtext_quantize_present."""
    original_jnp = getattr(q_maxtext, "jnp", None)
    try:

        class MockJNP:
            """Initialize class MockJNP."""

        q_maxtext.jnp = MockJNP()  # type: ignore[attr-defined]
        res = q_maxtext.quantize_model("dummy", "int8")
        if not res["status"] == "quantized_int8":
            raise AssertionError
        res2 = q_maxtext.quantize_model("dummy", "awq")
        if not res2["status"] == "quantized_awq":
            raise AssertionError
    finally:
        q_maxtext.jnp = original_jnp  # type: ignore[attr-defined]
