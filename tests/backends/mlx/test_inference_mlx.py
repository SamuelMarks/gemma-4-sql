import math
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.mlx.inference import (
    compute_confidence_score,
    generate_sql,
    mlx_beam_search,
)
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError


def test_compute_confidence_score():
    assert compute_confidence_score([], 0) == 0.0
    assert compute_confidence_score([-0.5, -0.5], 2) == pytest.approx(math.exp(-0.5))
    assert compute_confidence_score(-1.0, 2) == pytest.approx(math.exp(-0.5))
    assert compute_confidence_score([10.0], 1) == 1.0


def test_mlx_beam_search_no_model():
    with pytest.raises(InferenceError, match="Valid model instance is required"):
        mlx_beam_search(None, None, "test")


def test_mlx_beam_search_with_mx():
    mock_model = MagicMock()
    mock_mx = MagicMock()
    mock_mx.array.return_value = [1, 2, 3]
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.decode.return_value = " SELECT * "
    mock_tokenizer.eos_token_id = 99

    mock_out = MagicMock()
    mock_out.ndim = 3

    mock_next_logits = MagicMock()
    mock_next_logits.shape = [1, 100]

    mock_out.__getitem__.return_value = mock_next_logits
    mock_model.return_value = mock_out

    mock_log_probs = MagicMock()
    mock_mx.logsumexp.return_value = 0.1
    mock_next_logits.__sub__.return_value = mock_log_probs

    mock_top_k = MagicMock()
    mock_top_k.tolist.return_value = [4, 99]
    mock_mx.argsort.return_value.__getitem__.return_value = mock_top_k

    mock_lp = MagicMock()
    mock_lp.item.return_value = -0.1
    mock_log_probs.__getitem__.return_value = mock_lp

    with patch("gemma_4_sql.backends.mlx.inference.mx", mock_mx):
        sql, _conf = mlx_beam_search(mock_model, mock_tokenizer, "test", beam_width=2, max_length=1)
        assert sql == "SELECT *"


def test_mlx_beam_search_no_mx():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = MagicMock(tolist=lambda: [1, 2, 3])
    mock_tokenizer.decode.return_value = "SELECT *"
    mock_tokenizer.eos_token_id = None

    mock_model.return_value = [MagicMock()]

    with patch("gemma_4_sql.backends.mlx.inference.mx", None), pytest.raises(InferenceError, match="yielded only an EOS token"):
        mlx_beam_search(mock_model, mock_tokenizer, "test", beam_width=2, max_length=2, eos_token_id=99)


def test_mlx_beam_search_ndim_2():
    mock_model = MagicMock()
    mock_mx = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = MagicMock(tolist=lambda: [1, 2, 3])
    mock_tokenizer.decode.return_value = "SELECT *"
    mock_tokenizer.eos_token_id = None

    mock_out = MagicMock()
    mock_out.ndim = 2
    mock_next_logits = MagicMock()
    del mock_next_logits.shape  # ensure hasattr(next_logits, "shape") is False
    mock_out.__getitem__.return_value = mock_next_logits
    mock_model.return_value = mock_out

    with patch("gemma_4_sql.backends.mlx.inference.mx", mock_mx), pytest.raises(InferenceError, match="yielded only an EOS token"):
        mlx_beam_search(mock_model, mock_tokenizer, "test", beam_width=2, max_length=1)


def test_mlx_beam_search_empty_output():
    mock_model = MagicMock()
    mock_mx = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1]
    mock_tokenizer.decode.return_value = "  "
    mock_tokenizer.eos_token_id = 99

    mock_out = MagicMock()
    mock_out.ndim = 0
    mock_next_logits = MagicMock()
    mock_next_logits.shape = [1, 100]
    mock_out = mock_next_logits
    mock_model.return_value = mock_out

    mock_log_probs = MagicMock()
    mock_mx.logsumexp.return_value = 0.1
    mock_next_logits.__sub__.return_value = mock_log_probs
    mock_top_k = MagicMock()
    mock_top_k.tolist.return_value = [4, 99]
    mock_mx.argsort.return_value.__getitem__.return_value = mock_top_k
    mock_log_probs.__getitem__.return_value = -0.1

    with patch("gemma_4_sql.backends.mlx.inference.mx", mock_mx), pytest.raises(InferenceError, match="decoded into an empty SQL query string"):
        mlx_beam_search(mock_model, mock_tokenizer, "test", max_length=1)


def test_mlx_beam_search_only_eos():
    mock_model = MagicMock()
    mock_model.return_value = [MagicMock()]
    with patch("gemma_4_sql.tokenization.SQLTokenizer") as mock_tok_cls:
        mock_tok = mock_tok_cls.return_value
        mock_tok.encode.return_value = []
        mock_tok.decode.return_value = ""
        with patch("gemma_4_sql.backends.mlx.inference.mx", None), pytest.raises(InferenceError, match="yielded only an EOS token"):
            mlx_beam_search(mock_model, None, "test", eos_token_id=1, max_length=1)


def test_generate_sql_missing_deps():
    with patch("gemma_4_sql.backends.mlx.inference.load", None), pytest.raises(DependencyMissingError):
        generate_sql("model", "test")


def test_generate_sql_success():
    mock_load = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_load.return_value = (mock_model, mock_tokenizer)

    with patch("gemma_4_sql.backends.mlx.inference.load", mock_load), patch("gemma_4_sql.backends.mlx.inference.mlx_beam_search", return_value=("SELECT *", 0.9)):
        res = generate_sql("model", "test", eos_token_id=99)
        assert res["sql"] == "SELECT *"
        assert res["confidence_score"] == 0.9


def test_generate_sql_list_load():
    mock_load = MagicMock()
    mock_load.return_value = [MagicMock(), MagicMock()]

    with patch("gemma_4_sql.backends.mlx.inference.load", mock_load), patch("gemma_4_sql.backends.mlx.inference.mlx_beam_search", return_value=("SELECT *", 0.9)):
        res = generate_sql("model", "test")
        assert res["sql"] == "SELECT *"


def test_generate_sql_exception():
    mock_load = MagicMock()
    mock_load.side_effect = RuntimeError("Failed")
    with patch("gemma_4_sql.backends.mlx.inference.load", mock_load):
        res = generate_sql("model", "test")
        assert res["sql"] == ""
        assert res["confidence_score"] == 0.0
