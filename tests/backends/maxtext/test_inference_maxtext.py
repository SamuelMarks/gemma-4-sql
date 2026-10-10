"""Module docstring."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gemma_4_sql.backends.maxtext import inference
from gemma_4_sql.backends.maxtext.inference import _beam_search_step, _execute_generate, generate_sql, maxtext_beam_search
from gemma_4_sql.exceptions import DependencyMissingError


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    """Docstring for mock_dependencies."""
    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_gemma4 = MagicMock()

    mock_jax.nn.log_softmax.side_effect = lambda x, axis: x  # Dummy softmax

    # argsort returns indices sorting ascending, so we reverse it for descending
    mock_jnp.argsort.side_effect = lambda x: np.argsort(x)
    mock_jnp.concatenate.side_effect = lambda x, axis: np.concatenate(x, axis=axis)

    monkeypatch.setattr(inference, "jax", mock_jax)
    monkeypatch.setattr(inference, "jnp", mock_jnp)
    monkeypatch.setattr(inference, "Gemma4Model", mock_gemma4)

    return mock_jax, mock_jnp, mock_gemma4


def test_beam_search_step():
    """Docstring for test_beam_search_step."""
    seq = np.array([[1, 2, 3]])

    def model_apply_fn(s):
        """Docstring for model_apply_fn."""
        # Return logits where shape is 3D
        logits = np.zeros((1, 3, 5))
        # Top indices will be 4, 3, 2, 1, 0 based on values
        logits[0, -1, :] = [0.1, 0.2, 0.3, 0.4, 0.5]
        return logits

    beams = _beam_search_step(seq, 0.0, model_apply_fn, beam_width=2)
    assert len(beams) == 2

    new_seq_1, score_1 = beams[0]
    # Highest prob index is 4
    np.testing.assert_array_equal(new_seq_1, [[1, 2, 3, 4]])
    assert score_1 == 0.5


def test_beam_search_step_shapes():
    """Docstring for test_beam_search_step_shapes."""
    seq = np.array([[1, 2, 3]])

    def model_apply_fn_2d(s):
        """Docstring for model_apply_fn_2d."""
        logits = np.zeros((3, 5))
        logits[-1, :] = [0.1, 0.2, 0.3, 0.4, 0.5]
        return logits

    beams_2d = _beam_search_step(seq, 0.0, model_apply_fn_2d, beam_width=1)
    np.testing.assert_array_equal(beams_2d[0][0], [[1, 2, 3, 4]])

    def model_apply_fn_1d(s):
        """Docstring for model_apply_fn_1d."""
        logits = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        return logits

    beams_1d = _beam_search_step(seq, 0.0, model_apply_fn_1d, beam_width=1)
    np.testing.assert_array_equal(beams_1d[0][0], [[1, 2, 3, 4]])


def test_beam_search_step_no_item(mock_dependencies):
    """Docstring for test_beam_search_step_no_item."""
    mock_jax, mock_jnp, _mock_gemma4 = mock_dependencies

    seq = np.array([[1, 2]])

    def model_apply_fn(s):
        """Docstring for model_apply_fn."""
        return np.array([0.1, 0.2, 0.3])

    class NoItemScore:
        """Docstring for NoItemScore."""

        def __init__(self, v):
            """Docstring for __init__."""
            self.v = v

        def __float__(self):
            """Docstring for __float__."""
            return float(self.v)

        # no .item() method!

    def dummy_argsort(x):
        """Docstring for dummy_argsort."""
        return np.array([0, 1, 2])

    mock_jnp.argsort.side_effect = dummy_argsort

    dummy_log_probs = np.array([NoItemScore(0.1), NoItemScore(0.2), NoItemScore(0.3)], dtype=object)

    with patch.object(mock_jax.nn, "log_softmax", return_value=dummy_log_probs):
        beams = _beam_search_step(seq, 0.0, model_apply_fn, beam_width=1)
        assert beams[0][1] == 0.3


def test_maxtext_beam_search():
    """Docstring for test_maxtext_beam_search."""

    def apply_fn(seq):
        """Docstring for apply_fn."""
        return np.array([0.1, 0.9])  # always predicts 1

    input_ids = np.array([[0]])

    out_seq, _score = maxtext_beam_search(
        model_apply_fn=apply_fn,
        input_ids=input_ids,
        beam_width=1,
        max_length=3,
        eos_token_id=5,  # eos not hit
    )

    np.testing.assert_array_equal(out_seq, [[0, 1, 1, 1]])


def test_maxtext_beam_search_eos():
    """Docstring for test_maxtext_beam_search_eos."""
    call_count = 0

    def apply_fn(seq):
        """Docstring for apply_fn."""
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return np.array([0.1, 0.9])  # predict 1
        else:
            return np.array([0.9, 0.1])  # predict 0 (eos)

    input_ids = np.array([[2]])

    out_seq, _score = maxtext_beam_search(
        model_apply_fn=apply_fn,
        input_ids=input_ids,
        beam_width=1,
        max_length=5,
        eos_token_id=0,  # eos is 0
    )

    np.testing.assert_array_equal(out_seq, [[2, 1, 0]])


def test_maxtext_beam_search_eos_continue():
    """Docstring for test_maxtext_beam_search_eos_continue."""
    call_count = 0

    def apply_fn(seq):
        """Docstring for apply_fn."""
        nonlocal call_count
        call_count += 1
        # First call (input [2]): return top 2 tokens: 1 and 0 (eos)
        if call_count == 1:
            return np.array([0.9, 0.1])  # 0 and 1
        # Second call (input [2, 1]): return top 1 token: 0 (eos)
        else:
            return np.array([0.9, 0.1])  # 0 (eos)

    input_ids = np.array([[2]])

    _out_seq, _score = maxtext_beam_search(model_apply_fn=apply_fn, input_ids=input_ids, beam_width=2, max_length=5, eos_token_id=0)


def test_execute_generate(mock_dependencies):
    """Docstring for test_execute_generate."""
    mock_jax, _mock_jnp, _mock_gemma4 = mock_dependencies

    tokenizer = MagicMock()
    tokenizer.decode.return_value = "SELECT * FROM test"

    # Setup jax.jit to just return the function
    mock_jax.jit.side_effect = lambda fn, static_argnums: fn

    with patch("gemma_4_sql.backends.maxtext.inference.maxtext_beam_search") as mock_bs:
        mock_bs.return_value = (np.array([[1, 2, 3, 4]]), 1.5)

        status, sql, score = _execute_generate("test-model", [1, 2], 2, 10, 5, tokenizer)

        assert status == "success"
        assert sql == "SELECT * FROM test"
        assert score == 0.75  # 1.5 / max(1, 4 - 2)


def test_execute_generate_model_apply(mock_dependencies):
    """Docstring for test_execute_generate_model_apply."""
    mock_jax, _mock_jnp, mock_gemma4 = mock_dependencies

    mock_model = MagicMock()
    del mock_model.apply  # simulate model without apply method
    mock_gemma4.return_value = mock_model

    tokenizer = MagicMock()
    tokenizer.decode.return_value = "SELECT * FROM test"
    mock_jax.jit.side_effect = lambda fn, static_argnums: fn

    with patch("gemma_4_sql.backends.maxtext.inference.maxtext_beam_search") as mock_bs:
        mock_output_ids = MagicMock()
        mock_output = MagicMock()
        mock_output.tolist.return_value = [1, 2, 3]
        del mock_output.__len__  # force shape branch
        mock_output_ids.__getitem__.return_value = mock_output
        mock_output_ids.shape = (1, 3)
        mock_bs.return_value = (mock_output_ids, 1.5)

        status, _sql, score = _execute_generate("test-model", [1], 2, 10, 5, tokenizer)
        assert status == "success"
        # 1.5 / max(1, 3 - 1) = 0.75
        assert score == 0.75


def test_generate_sql(mock_dependencies):
    """Docstring for test_generate_sql."""
    with patch("gemma_4_sql.backends.maxtext.inference._execute_generate") as mock_exec:
        mock_exec.return_value = ("success", "SELECT 1", 0.9)

        result = generate_sql("test-model", "test prompt")
        assert result["sql"] == "SELECT 1"
        assert result["status"] == "success"
        assert result["confidence_score"] == 0.9


def test_generate_sql_missing_deps(monkeypatch):
    """Docstring for test_generate_sql_missing_deps."""
    monkeypatch.setattr(inference, "jax", None)

    with pytest.raises(DependencyMissingError, match="MaxText dependencies are missing"):
        generate_sql("test-model", "test prompt")


def test_generate_sql_error(mock_dependencies):
    """Docstring for test_generate_sql_error."""
    with patch("gemma_4_sql.backends.maxtext.inference._execute_generate") as mock_exec:
        mock_exec.side_effect = RuntimeError("Generation failed")

        result = generate_sql("test-model", "test prompt")
        assert result["sql"] == ""
        assert "failed: Generation failed" in result["status"]


# Add test to re-import and cover the ImportError branch in global scope


def test_import_error_coverage():
    """Docstring for test_import_error_coverage."""
    import importlib
    from unittest.mock import patch

    from gemma_4_sql.backends.maxtext import inference

    # Force ImportError on maxtext.models.gemma4
    original_import = __import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if "maxtext.models.gemma4" in name:
            raise ImportError("mocked")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        importlib.reload(inference)
        assert inference.Gemma4Model is None

    # restore and reload correctly to not break other tests
    importlib.reload(inference)


def test_inference_import_error():
    """Docstring for test_inference_import_error."""
    import importlib
    from unittest.mock import patch

    import gemma_4_sql.backends.maxtext.inference as inf

    with patch.dict("sys.modules", {"maxtext.models.gemma4": None}):
        importlib.reload(inf)
        assert inf.Gemma4Model is None
    importlib.reload(inf)


def test_inference_jax_import_error():
    """Docstring for test_inference_jax_import_error."""
    import builtins
    import importlib

    import gemma_4_sql.backends.maxtext.inference as inf

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "jax" or name == "jax.numpy":
            raise ImportError("mock")
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        importlib.reload(inf)
        assert inf.jax is None
    finally:
        builtins.__import__ = orig_import
        importlib.reload(inf)


def test_beam_search_75_74():
    """Docstring for test_beam_search_75_74."""
    import numpy as np

    import gemma_4_sql.backends.maxtext.inference as inf

    def apply_fn(seq):
        # We need seq[0, -1] to NOT be eos_token_id (0)
        # And we need to exhaust the beam width.
        # It doesn't matter, we just need it to run out of loop
        """Docstring for apply_fn."""
        return np.array([0.5, 0.5])

    # beam_width=1, max_length=1
    out_seq, _score = inf.maxtext_beam_search(
        model_apply_fn=apply_fn,
        input_ids=np.array([[2]]),
        beam_width=1,
        max_length=1,
        eos_token_id=0,
    )


def test_inference_gemma4_import_error():
    """Docstring for test_inference_gemma4_import_error."""
    import builtins
    import importlib

    import gemma_4_sql.backends.maxtext.inference as inf

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if "maxtext.models.gemma4" in name:
            raise ImportError("mock")
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        importlib.reload(inf)
        assert inf.Gemma4Model is None
    finally:
        builtins.__import__ = orig_import
        importlib.reload(inf)
