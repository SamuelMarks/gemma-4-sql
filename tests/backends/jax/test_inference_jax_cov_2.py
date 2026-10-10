"""Test file."""


def test_inference_restore_nnx_has_no_update_false_again_6(monkeypatch):
    """Test function."""
    """Test function."""
    from unittest.mock import MagicMock

    import gemma_4_sql.backends.jax.inference as inf

    monkeypatch.setattr(inf, "jax", MagicMock())
    monkeypatch.setattr(inf, "jnp", MagicMock())

    class MockModel:
        """Docstring for MockModel."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

    monkeypatch.setattr(inf, "Gemma4ForCausalLM", MockModel)
    monkeypatch.setattr(inf, "Gemma4Config", MagicMock())

    # Force nnx to be None
    monkeypatch.setattr(inf, "nnx", None)

    import pathlib

    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)

    import sys

    mock_ocp = MagicMock()
    mock_ocp.PyTreeCheckpointer.return_value.restore.return_value = "restored"

    if "orbax.checkpoint" in sys.modules:
        del sys.modules["orbax.checkpoint"]
    if "orbax" in sys.modules:
        del sys.modules["orbax"]

    sys.modules["orbax.checkpoint"] = mock_ocp
    sys.modules["orbax"] = MagicMock()

    mock_tokenizer_cls = MagicMock()
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1]
    mock_tok.vocab_size = 10
    mock_tokenizer_cls.return_value = mock_tok
    monkeypatch.setattr(inf, "SQLTokenizer", mock_tokenizer_cls)

    def mock_jax_beam_search(*args, **kwargs):
        """Docstring for mock_jax_beam_search."""
        out = MagicMock()
        out.tolist.return_value = [1]
        out.__len__ = lambda self: 1
        return ([out], 0.0)

    monkeypatch.setattr(inf, "jax_beam_search", mock_jax_beam_search)

    # We must use a unique model name
    inf.generate_sql("model_test_no_update_final_6", "prompt")
