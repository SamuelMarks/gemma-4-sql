"""Integration tests for real JAX and MaxText beam search algorithms using genuine JAX arrays."""

from __future__ import annotations

import jax.numpy as jnp

from gemma_4_sql.backends.jax.inference import _beam_search_step as jax_step
from gemma_4_sql.backends.jax.inference import jax_beam_search
from gemma_4_sql.backends.maxtext.inference import _beam_search_step as maxtext_step
from gemma_4_sql.backends.maxtext.inference import maxtext_beam_search


def test_real_jax_beam_search_step() -> None:
    """Test _beam_search_step with real JAX arrays.

    Returns:
        None.
    """
    vocab_size = 10
    seq = jnp.array([[1, 2, 3]])
    score = -0.5

    def mock_apply(input_seq: jnp.ndarray, _pos: jnp.ndarray) -> jnp.ndarray:
        """Mock apply function returning 3D logits.

        Args:
            input_seq: Input sequence array.
            _pos: Position IDs array.

        Returns:
            Logits tensor.
        """
        b_sz, s_len = input_seq.shape
        logits = jnp.zeros((b_sz, s_len, vocab_size))
        logits = logits.at[0, -1, 7].set(10.0)
        logits = logits.at[0, -1, 4].set(8.0)
        logits = logits.at[0, -1, 1].set(6.0)
        return logits

    new_beams = jax_step(seq, score, mock_apply, beam_width=2)
    assert len(new_beams) == 2
    top_beam_seq, top_beam_score = new_beams[0]
    assert top_beam_seq.shape == (1, 4)
    assert int(top_beam_seq[0, -1]) == 7
    assert top_beam_score < score


def test_real_jax_beam_search_expansion_and_eos() -> None:
    """Test jax_beam_search beam expansion and early termination on EOS.

    Returns:
        None.
    """
    vocab_size = 8
    eos_id = 2
    initial_seq = jnp.array([[1]])

    def mock_apply(input_seq: jnp.ndarray, _pos: jnp.ndarray) -> jnp.ndarray:
        """Mock apply function returning deterministic logits.

        Args:
            input_seq: Input sequence array.
            _pos: Position IDs array.

        Returns:
            Logits tensor.
        """
        b_sz, s_len = input_seq.shape
        logits = jnp.zeros((b_sz, s_len, vocab_size))
        if s_len >= 3:
            logits = logits.at[0, -1, eos_id].set(20.0)
        else:
            logits = logits.at[0, -1, 5].set(10.0)
            logits = logits.at[0, -1, 6].set(8.0)
        return logits

    best_seq, best_score = jax_beam_search(
        model_apply_fn=mock_apply,
        input_ids=initial_seq,
        beam_width=2,
        max_length=10,
        eos_token_id=eos_id,
    )

    assert isinstance(best_seq, jnp.ndarray)
    assert int(best_seq[0, -1]) == eos_id
    assert best_seq.shape[1] == 4
    assert best_score < 0.0


def test_real_maxtext_beam_search_step() -> None:
    """Test maxtext _beam_search_step with real JAX arrays.

    Returns:
        None.
    """
    vocab_size = 12
    seq = jnp.array([[3, 4]])
    score = 0.0

    def mock_apply(input_seq: jnp.ndarray) -> jnp.ndarray:
        """Mock apply function returning 3D logits.

        Args:
            input_seq: Input sequence array.

        Returns:
            Logits tensor.
        """
        b_sz, s_len = input_seq.shape
        logits = jnp.zeros((b_sz, s_len, vocab_size))
        logits = logits.at[0, -1, 9].set(15.0)
        logits = logits.at[0, -1, 8].set(10.0)
        return logits

    new_beams = maxtext_step(seq, score, mock_apply, beam_width=2)
    assert len(new_beams) == 2
    top_seq, _top_score = new_beams[0]
    assert top_seq.shape == (1, 3)
    assert int(top_seq[0, -1]) == 9


def test_real_maxtext_beam_search_expansion_and_eos() -> None:
    """Test maxtext_beam_search expansion, score accumulation, and EOS termination.

    Returns:
        None.
    """
    vocab_size = 10
    eos_id = 0
    initial_seq = jnp.array([[5]])

    def mock_apply(input_seq: jnp.ndarray) -> jnp.ndarray:
        """Mock apply function returning 2D or 3D logits.

        Args:
            input_seq: Input sequence array.

        Returns:
            Logits tensor.
        """
        b_sz, s_len = input_seq.shape
        logits = jnp.zeros((b_sz, s_len, vocab_size))
        if s_len >= 2:
            logits = logits.at[0, -1, eos_id].set(50.0)
        else:
            logits = logits.at[0, -1, 3].set(20.0)
            logits = logits.at[0, -1, 4].set(15.0)
        return logits

    best_seq, best_score = maxtext_beam_search(
        model_apply_fn=mock_apply,
        input_ids=initial_seq,
        beam_width=2,
        max_length=5,
        eos_token_id=eos_id,
    )

    assert isinstance(best_seq, jnp.ndarray)
    assert int(best_seq[0, -1]) == eos_id
    assert best_seq.shape[1] == 3
    assert best_score < 0.0


def test_jax_and_maxtext_beam_search_2d_and_1d_logits() -> None:
    """Test _beam_search_step with 2D and 1D logits for both JAX and MaxText.

    Returns:
        None.
    """
    seq = jnp.array([[1, 2]])
    beams_2d = jax_step(seq, 0.0, lambda s, p: jnp.zeros((2, 10)), beam_width=2)
    assert len(beams_2d) == 2

    beams_3d = jax_step(seq, 0.0, lambda s, p: jnp.zeros((3, 10)), beam_width=2)
    assert len(beams_3d) == 2

    beams_1d = jax_step(seq, 0.0, lambda s, p: jnp.zeros(10), beam_width=2)
    assert len(beams_1d) == 2

    max_2d = maxtext_step(seq, 0.0, lambda s: jnp.zeros((2, 10)), beam_width=2)
    assert len(max_2d) == 2

    max_3d = maxtext_step(seq, 0.0, lambda s: jnp.zeros((3, 10)), beam_width=2)
    assert len(max_3d) == 2

    max_1d = maxtext_step(seq, 0.0, lambda s: jnp.zeros(10), beam_width=2)
    assert len(max_1d) == 2
