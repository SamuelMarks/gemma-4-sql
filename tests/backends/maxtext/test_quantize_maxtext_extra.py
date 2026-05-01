import gemma_4_sql.backends.maxtext.quantize as q_maxtext


def test_maxtext_quantize_present():
    original_jnp = getattr(q_maxtext, "jnp", None)
    try:

        class MockJNP:
            pass

        q_maxtext.jnp = MockJNP()
        res = q_maxtext.quantize_model("dummy", "int8")
        assert res["status"] == "quantized_int8"

        res2 = q_maxtext.quantize_model("dummy", "awq")
        assert res2["status"] == "quantized_awq"
    finally:
        q_maxtext.jnp = original_jnp
