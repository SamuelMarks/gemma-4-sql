"""Test file."""

from unittest.mock import MagicMock


def test_inference_compute_step_probs(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.jax.inference as inf

    mock_jnp = MagicMock()
    mock_jnp.concatenate.return_value = MagicMock()
    mock_jnp.argsort.return_value = [1, 0]

    class MockOutput:
        """Docstring for MockOutput."""

        def __init__(self, s):
            """Docstring for __init__."""
            self.shape = s
            self.item = lambda: 0.1

        def __getitem__(self, idx):
            """Docstring for __getitem__."""
            return self

    mock_jnp.zeros.return_value = MockOutput((2,))
    mock_jnp.log.return_value = MockOutput((2,))

    mock_jax = MagicMock()
    monkeypatch.setattr(inf, "jax", mock_jax)
    monkeypatch.setattr(inf, "jnp", mock_jnp)

    res = inf._compute_step_probs([MockOutput((1, 5)), MockOutput((1, 5))], 2)
    assert res is not None


def test_inference_restore_typeerror(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.jax.inference as inf

    monkeypatch.setattr(inf, "jax", MagicMock())
    monkeypatch.setattr(inf, "jnp", MagicMock())

    class MockModel:
        """Docstring for MockModel."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

    monkeypatch.setattr(inf, "Gemma4ForCausalLM", MockModel)
    monkeypatch.setattr(inf, "Gemma4Config", MagicMock())

    class MockNNX:
        """Docstring for MockNNX."""

        def Rngs(self, *args):
            """Docstring for Rngs."""
            return

    monkeypatch.setattr(inf, "nnx", MockNNX())

    inf._MODEL_CACHE.pop("fail_typeerror", None)

    import pathlib

    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)

    import sys

    mock_ocp = MagicMock()
    mock_ocp.PyTreeCheckpointer.return_value.restore.side_effect = TypeError("type error during restore")
    sys.modules["orbax.checkpoint"] = mock_ocp
    mock_orbax = MagicMock()
    mock_orbax.checkpoint = mock_ocp
    sys.modules["orbax"] = mock_orbax

    mock_tokenizer_cls = MagicMock()
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1]
    mock_tok.vocab_size = 10
    mock_tok.decode.return_value = "decoded"
    mock_tokenizer_cls.return_value = mock_tok
    monkeypatch.setattr(inf, "SQLTokenizer", mock_tokenizer_cls)

    def mock_jax_beam_search(*args, **kwargs):
        """Docstring for mock_jax_beam_search."""
        out = MagicMock()
        out.tolist.return_value = [1]
        out.__len__ = lambda self: 1
        return ([out], 0.0)

    monkeypatch.setattr(inf, "jax_beam_search", mock_jax_beam_search)

    res = inf.generate_sql("fail_typeerror", "prompt")
    assert res["status"] == "success"


def test_inference_restore_other_error(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.jax.inference as inf

    monkeypatch.setattr(inf, "jax", MagicMock())
    monkeypatch.setattr(inf, "jnp", MagicMock())

    class MockModel:
        """Docstring for MockModel."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

    monkeypatch.setattr(inf, "Gemma4ForCausalLM", MockModel)
    monkeypatch.setattr(inf, "Gemma4Config", MagicMock())

    class MockNNX:
        """Docstring for MockNNX."""

        def Rngs(self, *args):
            """Docstring for Rngs."""
            return

    monkeypatch.setattr(inf, "nnx", MockNNX())

    inf._MODEL_CACHE.pop("fail_oserror", None)

    import pathlib

    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)

    import sys

    mock_ocp = MagicMock()
    mock_ocp.PyTreeCheckpointer.return_value.restore.side_effect = OSError("os error during restore")
    sys.modules["orbax.checkpoint"] = mock_ocp
    mock_orbax = MagicMock()
    mock_orbax.checkpoint = mock_ocp
    sys.modules["orbax"] = mock_orbax

    mock_tokenizer_cls = MagicMock()
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1]
    mock_tok.vocab_size = 10
    mock_tok.decode.return_value = "decoded"
    mock_tokenizer_cls.return_value = mock_tok
    monkeypatch.setattr(inf, "SQLTokenizer", mock_tokenizer_cls)

    def mock_jax_beam_search(*args, **kwargs):
        """Docstring for mock_jax_beam_search."""
        out = MagicMock()
        out.tolist.return_value = [1]
        out.__len__ = lambda self: 1
        return ([out], 0.0)

    monkeypatch.setattr(inf, "jax_beam_search", mock_jax_beam_search)

    res = inf.generate_sql("fail_oserror", "prompt")
    assert res["status"] == "success"


def test_generate_sql_restore_runtime_error(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.jax.inference as inf

    monkeypatch.setattr(inf, "jax", MagicMock())
    monkeypatch.setattr(inf, "jnp", MagicMock())

    class MockModel:
        """Docstring for MockModel."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

    monkeypatch.setattr(inf, "Gemma4ForCausalLM", MockModel)
    monkeypatch.setattr(inf, "Gemma4Config", MagicMock())

    class MockNNX:
        """Docstring for MockNNX."""

        def Rngs(self, *args):
            """Docstring for Rngs."""
            return

    monkeypatch.setattr(inf, "nnx", MockNNX())

    inf._MODEL_CACHE.pop("fail_runtimeerror", None)

    import pathlib

    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)

    import sys

    mock_ocp = MagicMock()
    mock_ocp.PyTreeCheckpointer.return_value.restore.side_effect = RuntimeError("runtime error during restore")
    sys.modules["orbax.checkpoint"] = mock_ocp
    mock_orbax = MagicMock()
    mock_orbax.checkpoint = mock_ocp
    sys.modules["orbax"] = mock_orbax

    mock_tokenizer_cls = MagicMock()
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1]
    mock_tok.vocab_size = 10
    mock_tok.decode.return_value = "decoded"
    mock_tokenizer_cls.return_value = mock_tok
    monkeypatch.setattr(inf, "SQLTokenizer", mock_tokenizer_cls)

    def mock_jax_beam_search(*args, **kwargs):
        """Docstring for mock_jax_beam_search."""
        out = MagicMock()
        out.tolist.return_value = [1]
        out.__len__ = lambda self: 1
        return ([out], 0.0)

    monkeypatch.setattr(inf, "jax_beam_search", mock_jax_beam_search)

    res = inf.generate_sql("fail_runtimeerror", "prompt")
    assert res["status"] == "success"


def test_generate_sql_restore_key_error(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.jax.inference as inf

    monkeypatch.setattr(inf, "jax", MagicMock())
    monkeypatch.setattr(inf, "jnp", MagicMock())

    class MockModel:
        """Docstring for MockModel."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

    monkeypatch.setattr(inf, "Gemma4ForCausalLM", MockModel)
    monkeypatch.setattr(inf, "Gemma4Config", MagicMock())

    class MockNNX:
        """Docstring for MockNNX."""

        def Rngs(self, *args):
            """Docstring for Rngs."""
            return

    monkeypatch.setattr(inf, "nnx", MockNNX())

    inf._MODEL_CACHE.pop("fail_keyerror", None)

    import pathlib

    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)

    import sys

    mock_ocp = MagicMock()
    mock_ocp.PyTreeCheckpointer.return_value.restore.side_effect = KeyError("key error during restore")
    sys.modules["orbax.checkpoint"] = mock_ocp
    mock_orbax = MagicMock()
    mock_orbax.checkpoint = mock_ocp
    sys.modules["orbax"] = mock_orbax

    mock_tokenizer_cls = MagicMock()
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1]
    mock_tok.vocab_size = 10
    mock_tok.decode.return_value = "decoded"
    mock_tokenizer_cls.return_value = mock_tok
    monkeypatch.setattr(inf, "SQLTokenizer", mock_tokenizer_cls)

    def mock_jax_beam_search(*args, **kwargs):
        """Docstring for mock_jax_beam_search."""
        out = MagicMock()
        out.tolist.return_value = [1]
        out.__len__ = lambda self: 1
        return ([out], 0.0)

    monkeypatch.setattr(inf, "jax_beam_search", mock_jax_beam_search)

    res = inf.generate_sql("fail_keyerror", "prompt")
    assert res["status"] == "success"


def test_generate_sql_restore_attribute_error(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.jax.inference as inf

    monkeypatch.setattr(inf, "jax", MagicMock())
    monkeypatch.setattr(inf, "jnp", MagicMock())

    class MockModel:
        """Docstring for MockModel."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

    monkeypatch.setattr(inf, "Gemma4ForCausalLM", MockModel)
    monkeypatch.setattr(inf, "Gemma4Config", MagicMock())

    class MockNNX:
        """Docstring for MockNNX."""

        def Rngs(self, *args):
            """Docstring for Rngs."""
            return

    monkeypatch.setattr(inf, "nnx", MockNNX())

    inf._MODEL_CACHE.pop("fail_attributeerror", None)

    import pathlib

    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)

    import sys

    mock_ocp = MagicMock()
    mock_ocp.PyTreeCheckpointer.return_value.restore.side_effect = AttributeError("attribute error during restore")
    sys.modules["orbax.checkpoint"] = mock_ocp
    mock_orbax = MagicMock()
    mock_orbax.checkpoint = mock_ocp
    sys.modules["orbax"] = mock_orbax

    mock_tokenizer_cls = MagicMock()
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1]
    mock_tok.vocab_size = 10
    mock_tok.decode.return_value = "decoded"
    mock_tokenizer_cls.return_value = mock_tok
    monkeypatch.setattr(inf, "SQLTokenizer", mock_tokenizer_cls)

    def mock_jax_beam_search(*args, **kwargs):
        """Docstring for mock_jax_beam_search."""
        out = MagicMock()
        out.tolist.return_value = [1]
        out.__len__ = lambda self: 1
        return ([out], 0.0)

    monkeypatch.setattr(inf, "jax_beam_search", mock_jax_beam_search)

    res = inf.generate_sql("fail_attributeerror", "prompt")
    assert res["status"] == "success"


def test_inference_restore_all_exceptions(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.jax.inference as inf

    monkeypatch.setattr(inf, "jax", MagicMock())
    monkeypatch.setattr(inf, "jnp", MagicMock())

    class MockModel:
        """Docstring for MockModel."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

    monkeypatch.setattr(inf, "Gemma4ForCausalLM", MockModel)
    monkeypatch.setattr(inf, "Gemma4Config", MagicMock())

    class MockNNX:
        """Docstring for MockNNX."""

        def Rngs(self, *args):
            """Docstring for Rngs."""
            return

    monkeypatch.setattr(inf, "nnx", MockNNX())

    inf._MODEL_CACHE.pop("fail_all", None)

    import pathlib

    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)

    import sys

    mock_ocp = MagicMock()

    # We will trigger the except block via an AttributeError on PyTreeCheckpointer
    def raise_err():
        """Docstring for raise_err."""
        raise AttributeError("Simulated attribute err")

    mock_ocp.PyTreeCheckpointer = raise_err
    sys.modules["orbax.checkpoint"] = mock_ocp
    mock_orbax = MagicMock()
    mock_orbax.checkpoint = mock_ocp
    sys.modules["orbax"] = mock_orbax

    mock_tokenizer_cls = MagicMock()
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1]
    mock_tok.vocab_size = 10
    mock_tok.decode.return_value = "decoded"
    mock_tokenizer_cls.return_value = mock_tok
    monkeypatch.setattr(inf, "SQLTokenizer", mock_tokenizer_cls)

    def mock_jax_beam_search(*args, **kwargs):
        """Docstring for mock_jax_beam_search."""
        out = MagicMock()
        out.tolist.return_value = [1]
        out.__len__ = lambda self: 1
        return ([out], 0.0)

    monkeypatch.setattr(inf, "jax_beam_search", mock_jax_beam_search)

    res = inf.generate_sql("fail_all", "prompt")
    assert res["status"] == "success"


def test_inference_restore_all_exceptions2(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.jax.inference as inf

    monkeypatch.setattr(inf, "jax", MagicMock())
    monkeypatch.setattr(inf, "jnp", MagicMock())

    class MockModel:
        """Docstring for MockModel."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

    monkeypatch.setattr(inf, "Gemma4ForCausalLM", MockModel)
    monkeypatch.setattr(inf, "Gemma4Config", MagicMock())

    class MockNNX:
        """Docstring for MockNNX."""

        def Rngs(self, *args):
            """Docstring for Rngs."""
            return

    monkeypatch.setattr(inf, "nnx", MockNNX())

    inf._MODEL_CACHE.pop("fail_all2", None)

    import pathlib

    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)

    import sys

    mock_ocp = MagicMock()

    # We will trigger the except block via an AttributeError on PyTreeCheckpointer
    def raise_err():
        """Docstring for raise_err."""
        raise AttributeError("Simulated attribute err")

    mock_ocp.PyTreeCheckpointer = raise_err
    sys.modules["orbax.checkpoint"] = mock_ocp
    mock_orbax = MagicMock()
    mock_orbax.checkpoint = mock_ocp
    sys.modules["orbax"] = mock_orbax

    mock_tokenizer_cls = MagicMock()
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1]
    mock_tok.vocab_size = 10
    mock_tok.decode.return_value = "decoded"
    mock_tokenizer_cls.return_value = mock_tok
    monkeypatch.setattr(inf, "SQLTokenizer", mock_tokenizer_cls)

    def mock_jax_beam_search(*args, **kwargs):
        """Docstring for mock_jax_beam_search."""
        out = MagicMock()
        out.tolist.return_value = [1]
        out.__len__ = lambda self: 1
        return ([out], 0.0)

    monkeypatch.setattr(inf, "jax_beam_search", mock_jax_beam_search)

    res = inf.generate_sql("fail_all2", "prompt")
    assert res["status"] == "success"


def test_inference_restore_all_exceptions3(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.jax.inference as inf

    monkeypatch.setattr(inf, "jax", MagicMock())
    monkeypatch.setattr(inf, "jnp", MagicMock())

    class MockModel:
        """Docstring for MockModel."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

    monkeypatch.setattr(inf, "Gemma4ForCausalLM", MockModel)
    monkeypatch.setattr(inf, "Gemma4Config", MagicMock())

    class MockNNX:
        """Docstring for MockNNX."""

        def Rngs(self, *args):
            """Docstring for Rngs."""
            return

    monkeypatch.setattr(inf, "nnx", MockNNX())

    inf._MODEL_CACHE.pop("fail_all3", None)

    import pathlib

    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)

    import sys

    mock_ocp = MagicMock()

    def raise_err(*args, **kwargs):
        """Docstring for raise_err."""
        raise ValueError("Simulated ValueError")

    mock_ocp.PyTreeCheckpointer.return_value.restore.side_effect = raise_err
    sys.modules["orbax.checkpoint"] = mock_ocp
    mock_orbax = MagicMock()
    mock_orbax.checkpoint = mock_ocp
    sys.modules["orbax"] = mock_orbax

    mock_tokenizer_cls = MagicMock()
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1]
    mock_tok.vocab_size = 10
    mock_tok.decode.return_value = "decoded"
    mock_tokenizer_cls.return_value = mock_tok
    monkeypatch.setattr(inf, "SQLTokenizer", mock_tokenizer_cls)

    def mock_jax_beam_search(*args, **kwargs):
        """Docstring for mock_jax_beam_search."""
        out = MagicMock()
        out.tolist.return_value = [1]
        out.__len__ = lambda self: 1
        return ([out], 0.0)

    monkeypatch.setattr(inf, "jax_beam_search", mock_jax_beam_search)

    res = inf.generate_sql("fail_all3", "prompt")
    assert res["status"] == "success"


def test_generate_sql_restore_typeerror_actually(monkeypatch):
    """Test function."""
    """Test function."""
    """Test function."""
    import gemma_4_sql.backends.jax.inference as inf

    monkeypatch.setattr(inf, "jax", MagicMock())
    monkeypatch.setattr(inf, "jnp", MagicMock())

    class MockModel:
        """Docstring for MockModel."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

    monkeypatch.setattr(inf, "Gemma4ForCausalLM", MockModel)
    monkeypatch.setattr(inf, "Gemma4Config", MagicMock())

    class MockNNX:
        """Docstring for MockNNX."""

        def Rngs(self, *args):
            """Docstring for Rngs."""
            return

    monkeypatch.setattr(inf, "nnx", MockNNX())

    inf._MODEL_CACHE.pop("fail_actual_typeerror", None)

    import pathlib

    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)

    import sys

    mock_ocp = MagicMock()
    mock_ocp.PyTreeCheckpointer.return_value.restore.side_effect = TypeError("type error")
    sys.modules["orbax.checkpoint"] = mock_ocp
    mock_orbax = MagicMock()
    mock_orbax.checkpoint = mock_ocp
    sys.modules["orbax"] = mock_orbax

    mock_tokenizer_cls = MagicMock()
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1]
    mock_tok.vocab_size = 10
    mock_tok.decode.return_value = "decoded"
    mock_tokenizer_cls.return_value = mock_tok
    monkeypatch.setattr(inf, "SQLTokenizer", mock_tokenizer_cls)

    def mock_jax_beam_search(*args, **kwargs):
        """Docstring for mock_jax_beam_search."""
        out = MagicMock()
        out.tolist.return_value = [1]
        out.__len__ = lambda self: 1
        return ([out], 0.0)

    monkeypatch.setattr(inf, "jax_beam_search", mock_jax_beam_search)

    res = inf.generate_sql("fail_actual_typeerror", "prompt")
    assert res["status"] == "success"
