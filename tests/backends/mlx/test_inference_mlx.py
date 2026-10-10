"""Tests for mlx inference."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, InferenceError


def test_mlx_inference_imports():
    """Test mlx inference imports fallback."""

    with patch.dict(sys.modules, {"mlx": None, "mlx.core": None, "mlx_lm": None}):
        if "gemma_4_sql.backends.mlx.inference" in sys.modules:
            del sys.modules["gemma_4_sql.backends.mlx.inference"]
        import gemma_4_sql.backends.mlx.inference as inf_module

        assert inf_module.mx is None
        assert inf_module.load is None
        assert inf_module.generate is None


def test_compute_confidence_score():
    """Test compute_confidence_score."""
    import gemma_4_sql.backends.mlx.inference as inf_module

    assert inf_module.compute_confidence_score([], 0) == 0.0

    # max capped
    assert inf_module.compute_confidence_score([0.1, 0.2], 2) == 1.0

    # valid
    assert 0.0 < inf_module.compute_confidence_score([-1.0, -2.0], 2) < 1.0

    # float
    assert 0.0 < inf_module.compute_confidence_score(-3.0, 2) < 1.0


def test_mlx_beam_search_no_model():
    """Test mlx_beam_search."""
    import gemma_4_sql.backends.mlx.inference as inf_module

    with pytest.raises(InferenceError, match="Valid model instance is required"):
        inf_module.mlx_beam_search(None, None, "prompt")


def test_mlx_beam_search_success():
    """Test mlx_beam_search."""
    import gemma_4_sql.backends.mlx.inference as inf_module

    mock_mx = MagicMock()
    inf_module.mx = mock_mx

    mock_model = MagicMock()

    # Next logits structure
    mock_next_logits = MagicMock()
    mock_next_logits.shape = (1, 1, 100)

    mock_mx.logsumexp.return_value = 0
    # Provide next_logits - logsumexp result
    mock_log_probs = MagicMock()
    mock_next_logits.__sub__.return_value = mock_log_probs

    # Return indices
    mock_mx.argsort.return_value = MagicMock()
    # Let's say top-k returns token IDs [5, 6, 2]
    # We are returning a list for .tolist()
    mock_mx.argsort.return_value.__getitem__.return_value.tolist.return_value = [5, 2]  # 2 is EOS

    mock_log_probs.__getitem__.side_effect = lambda idx: MagicMock(item=lambda: -0.1) if idx == 5 else MagicMock(item=lambda: -0.2)

    # out.ndim == 3 -> out[0, -1, :]
    mock_out = MagicMock()
    mock_out.ndim = 3
    mock_out.__getitem__.return_value = mock_next_logits
    mock_model.return_value = mock_out

    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3]
    mock_tok.eos_token_id = 2
    mock_tok.decode.return_value = "SELECT 1;"

    sql, conf = inf_module.mlx_beam_search(
        model=mock_model,
        tokenizer=mock_tok,
        prompt="prompt",
        beam_width=2,
        max_length=5,
        eos_token_id=None,
    )

    assert sql == "SELECT 1;"
    assert conf > 0.0

    # test ndim 2
    class MockOut2D:
        """Docstring for MockOut2D."""

        ndim = 2

        def __getitem__(self, item):
            """Docstring for __getitem__."""
            return mock_next_logits

    mock_model.return_value = MockOut2D()
    inf_module.mlx_beam_search(mock_model, mock_tok, "prompt", max_length=1)

    # test list
    mock_model.return_value = [mock_next_logits]
    inf_module.mlx_beam_search(mock_model, mock_tok, "prompt", max_length=1)

    # test tuple
    mock_model.return_value = (mock_next_logits,)
    inf_module.mlx_beam_search(mock_model, mock_tok, "prompt", max_length=1)

    # test other
    mock_model.return_value = mock_next_logits
    inf_module.mlx_beam_search(mock_model, mock_tok, "prompt", max_length=1)


def test_mlx_beam_search_no_mx():
    """Test mlx_beam_search no mx."""
    import gemma_4_sql.backends.mlx.inference as inf_module

    inf_module.mx = None

    mock_model = MagicMock()
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3]
    mock_tok.eos_token_id = 2

    # Returns out
    mock_model.return_value = [1, 2]

    # It will hit the branch where next_logits doesn't have shape
    with pytest.raises(InferenceError, match="yielded an empty sequence|yielded only an EOS"):
        inf_module.mlx_beam_search(mock_model, mock_tok, "prompt", max_length=1)


def test_mlx_beam_search_fallbacks():
    """Test mlx_beam_search missing tokenizer tools."""
    import gemma_4_sql.backends.mlx.inference as inf_module

    inf_module.mx = MagicMock()

    mock_model = MagicMock()
    mock_tok = MagicMock(spec=[])  # No encode/decode/eos_token_id

    mock_out = MagicMock()
    mock_out.ndim = 2
    mock_next_logits = MagicMock()
    mock_next_logits.shape = (1, 10)
    mock_out.__getitem__.return_value = mock_next_logits
    mock_model.return_value = mock_out

    mock_next_logits.__sub__.return_value.__getitem__.return_value.item.return_value = -0.1
    inf_module.mx.argsort.return_value.__getitem__.return_value.tolist.return_value = [5]  # Not EOS (EOS is 1)

    with patch("gemma_4_sql.tokenization.SQLTokenizer") as mock_sql_tok:
        mock_sql_tok.return_value.encode.return_value = [5, 6]
        mock_sql_tok.return_value.decode.return_value = "sql"

        sql, conf = inf_module.mlx_beam_search(mock_model, mock_tok, "prompt", max_length=1)
        assert sql == "sql"


def test_mlx_beam_search_empty_results():
    """Test mlx_beam_search empty results."""
    import gemma_4_sql.backends.mlx.inference as inf_module

    inf_module.mx = MagicMock()
    mock_model = MagicMock()

    mock_out = MagicMock()
    mock_out.ndim = 2
    mock_next_logits = MagicMock()
    mock_next_logits.shape = (1, 10)
    mock_out.__getitem__.return_value = mock_next_logits
    mock_model.return_value = mock_out

    mock_next_logits.__sub__.return_value.__getitem__.return_value = -0.1
    inf_module.mx.argsort.return_value.__getitem__.return_value.tolist.return_value = [1]

    mock_tok = MagicMock()
    mock_tok.encode.return_value = []
    mock_tok.eos_token_id = 1

    with pytest.raises(InferenceError, match="yielded only an EOS"):
        inf_module.mlx_beam_search(mock_model, mock_tok, "prompt", max_length=1)

    inf_module.mx.argsort.return_value.__getitem__.return_value.tolist.return_value = [2]
    mock_tok.decode.return_value = ""
    with pytest.raises(InferenceError, match="empty SQL query string"):
        inf_module.mlx_beam_search(mock_model, mock_tok, "prompt", max_length=1)


def test_generate_sql():
    """Test generate_sql."""
    import gemma_4_sql.backends.mlx.inference as inf_module

    inf_module.load = MagicMock()

    mock_model = MagicMock()
    mock_tok = MagicMock()
    inf_module.load.__call__ = MagicMock(return_value=(mock_model, mock_tok))

    with patch.object(inf_module, "mlx_beam_search") as mock_beam:
        mock_beam.return_value = ("SELECT 1", 0.9)

        res = inf_module.generate_sql("model", "prompt", eos_token_id=2)
        assert res["sql"] == "SELECT 1"
        assert res["status"] == "success"

        # Test error
        mock_beam.side_effect = RuntimeError("error")
        res2 = inf_module.generate_sql("model", "prompt")
        assert res2["status"] == "failed: error"
        assert res2["sql"] == ""

    inf_module.load = None
    with pytest.raises(DependencyMissingError):
        inf_module.generate_sql("model", "prompt")


def test_mlx_inference_successful_imports():
    """Test mlx inference successful imports."""
    import sys
    from unittest.mock import MagicMock

    mock_mlx_lm = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "mlx_lm": mock_mlx_lm,
        },
    ):
        if "gemma_4_sql.backends.mlx.inference" in sys.modules:
            del sys.modules["gemma_4_sql.backends.mlx.inference"]
        import gemma_4_sql.backends.mlx.inference as inf_module

        assert inf_module.load is not None
        assert inf_module.generate is not None


def test_mlx_beam_search_eos_and_empty():
    """Docstring for test_mlx_beam_search_eos_and_empty."""
    from unittest.mock import MagicMock

    import gemma_4_sql.backends.mlx.inference as inf_module
    from gemma_4_sql.exceptions import InferenceError

    inf_module.mx = MagicMock()
    mock_model = MagicMock()
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1]

    # Test eos_token_id is not None
    # We will trigger the loop but let it just run 1 iter
    # To hit the missing line 158, max_length=0

    with pytest.raises(InferenceError, match="yielded an empty sequence"):
        inf_module.mlx_beam_search(
            mock_model,
            mock_tok,
            "prompt",
            max_length=0,
            eos_token_id=5,
        )
