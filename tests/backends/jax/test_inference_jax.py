import sys
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.jax.inference import (
    _MODEL_CACHE,
    _beam_search_step,
    _compute_step_probs,
    generate_sql,
    jax_beam_search,
)
from gemma_4_sql.exceptions import DependencyMissingError


@pytest.fixture
def mock_jax_deps(monkeypatch):
    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_nnx = MagicMock()
    mock_gemma_config = MagicMock()
    mock_gemma_model = MagicMock()
    mock_ocp = MagicMock()

    # jax
    mock_jax.nn.log_softmax = MagicMock(side_effect=lambda x, axis: x)
    mock_jax.jit = MagicMock(side_effect=lambda fn, **kwargs: fn)

    # jnp
    def mock_argsort(x):
        return MagicMock()

    mock_jnp.argsort = MagicMock(side_effect=mock_argsort)
    mock_jnp.arange = MagicMock(return_value=MagicMock())
    mock_jnp.concatenate = MagicMock(return_value=MagicMock())
    mock_jnp.array = MagicMock(side_effect=lambda x, dtype=None: x)

    # Mock orbax checkpoint
    sys.modules["orbax"] = MagicMock()
    sys.modules["orbax.checkpoint"] = mock_ocp

    monkeypatch.setattr("gemma_4_sql.backends.jax.inference.jax", mock_jax)
    monkeypatch.setattr("gemma_4_sql.backends.jax.inference.jnp", mock_jnp)
    monkeypatch.setattr("gemma_4_sql.backends.jax.inference.nnx", mock_nnx)
    monkeypatch.setattr("gemma_4_sql.backends.jax.inference.Gemma4Config", mock_gemma_config)
    monkeypatch.setattr("gemma_4_sql.backends.jax.inference.Gemma4ForCausalLM", mock_gemma_model)

    # We also need to clear _MODEL_CACHE
    import gemma_4_sql.backends.jax.inference as inf_module

    inf_module._MODEL_CACHE.clear()

    return {
        "jax": mock_jax,
        "jnp": mock_jnp,
        "nnx": mock_nnx,
        "gemma_config": mock_gemma_config,
        "gemma_model": mock_gemma_model,
        "ocp": mock_ocp,
    }


def test_compute_step_probs():
    # 3D
    logits = MagicMock()
    logits.shape = (1, 10, 100)
    last_logits = MagicMock()
    logits.__getitem__.return_value = last_logits

    mock_jax = MagicMock()
    mock_jax.nn.log_softmax.return_value = MagicMock()

    mock_jnp = MagicMock()
    mock_indices = MagicMock()
    mock_jnp.argsort.return_value = MagicMock(__getitem__=MagicMock(return_value=mock_indices))

    with patch("gemma_4_sql.backends.jax.inference.jax", mock_jax), patch("gemma_4_sql.backends.jax.inference.jnp", mock_jnp):
        top_idx, top_probs = _compute_step_probs(logits, 2)
        assert logits.__getitem__.call_args[0][0] == (0, -1, slice(None, None, None))

    # 2D
    logits = MagicMock()
    logits.shape = (10, 100)
    with patch("gemma_4_sql.backends.jax.inference.jax", mock_jax), patch("gemma_4_sql.backends.jax.inference.jnp", mock_jnp):
        top_idx, top_probs = _compute_step_probs(logits, 2)
        assert logits.__getitem__.call_args[0][0] == (-1, slice(None, None, None))

    # 1D
    logits = MagicMock()
    logits.shape = (100,)
    with patch("gemma_4_sql.backends.jax.inference.jax", mock_jax), patch("gemma_4_sql.backends.jax.inference.jnp", mock_jnp):
        _top_idx, _top_probs = _compute_step_probs(logits, 2)
        assert not logits.__getitem__.called  # because len is 1, wait, len is 1 but we don't mock len


def test_compute_step_probs_1d():
    # 1D
    class ShapeMock:
        def __init__(self, shape):
            self.shape = shape

        def __getitem__(self, idx):
            return self

        def __len__(self):
            return len(self.shape)

    logits = ShapeMock((100,))

    mock_jax = MagicMock()
    mock_jax.nn.log_softmax.return_value = MagicMock()

    mock_jnp = MagicMock()
    mock_indices = MagicMock()
    mock_jnp.argsort.return_value = MagicMock(__getitem__=MagicMock(return_value=mock_indices))

    with patch("gemma_4_sql.backends.jax.inference.jax", mock_jax), patch("gemma_4_sql.backends.jax.inference.jnp", mock_jnp):
        _top_idx, _top_probs = _compute_step_probs(logits, 2)
        # Should just pass logits directly


def test_beam_search_step(mock_jax_deps):
    seq = MagicMock()
    seq.shape = (1, 5)
    model_apply_fn = MagicMock(return_value=MagicMock())

    top_indices = [MagicMock(), MagicMock()]
    top_indices[0].reshape.return_value = "token1"
    top_indices[1].reshape.return_value = "token2"

    top_probs = [0.5, 0.4]  # has .item() mocked by being float? Let's use MagicMock with item()
    prob1 = MagicMock()
    prob1.item.return_value = 0.5
    prob2 = MagicMock()
    # test branch where item is not available
    del prob2.item
    prob2.__float__ = MagicMock(return_value=0.4)

    top_probs = [prob1, prob2]

    # We need to mock _compute_step_probs because jax.jit is mocked to return the function itself
    with patch("gemma_4_sql.backends.jax.inference._compute_step_probs", return_value=(top_indices, top_probs)):
        beams = _beam_search_step(seq, 1.0, model_apply_fn, 2)

    assert len(beams) == 2
    assert beams[0][1] == 1.5
    assert beams[1][1] == 1.4


def test_beam_search_step_no_jit():
    # Test branch where jax is None or has no jit
    seq = MagicMock()
    seq.shape = (1, 5)
    model_apply_fn = MagicMock(return_value=MagicMock())

    top_indices = [MagicMock()]
    top_indices[0].reshape.return_value = "token1"
    top_probs = [0.5]

    mock_jax = MagicMock()
    del mock_jax.jit

    with patch("gemma_4_sql.backends.jax.inference.jax", mock_jax), patch("gemma_4_sql.backends.jax.inference.jnp", MagicMock()), patch("gemma_4_sql.backends.jax.inference._compute_step_probs", return_value=(top_indices, top_probs)):
        beams = _beam_search_step(seq, 1.0, model_apply_fn, 1)
        assert len(beams) == 1
        assert beams[0][1] == 1.5


def test_jax_beam_search():
    model_apply_fn = MagicMock()

    # Mock sequence tensors
    seq_init = MagicMock()
    seq_init.__getitem__.return_value = 0  # Not eos

    seq_eos = MagicMock()
    seq_eos.__getitem__.return_value = 99  # eos

    seq_not_eos = MagicMock()
    seq_not_eos.__getitem__.return_value = 1

    # First step expands to seq_eos and seq_not_eos
    def mock_step(seq, score, fn, bw):
        if seq == seq_init:
            return [(seq_not_eos, 0.9), (seq_eos, 0.8)]
        if seq == seq_not_eos:
            return [(seq_eos, 1.5)]
        return []

    with patch("gemma_4_sql.backends.jax.inference._beam_search_step", side_effect=mock_step):
        res_seq, res_score = jax_beam_search(model_apply_fn, seq_init, beam_width=2, max_length=3, eos_token_id=99)

    # First iter:
    # seq_init -> seq_not_eos (0.9), seq_eos (0.8)
    # Beams: (seq_not_eos, 0.9), (seq_eos, 0.8)
    # Second iter:
    # seq_not_eos -> seq_eos (1.5)
    # seq_eos is eos, so kept: (seq_eos, 0.8)
    # New beams sorted: (seq_eos, 1.5), (seq_eos, 0.8)
    # All are eos, break
    assert res_seq == seq_eos
    assert res_score == 1.5


def test_generate_sql_missing_deps(monkeypatch):
    monkeypatch.setattr("gemma_4_sql.backends.jax.inference.jax", None)

    with pytest.raises(DependencyMissingError, match="JAX inference dependencies are missing."):
        generate_sql("model", "prompt")


def test_generate_sql_success(mock_jax_deps):
    mock_model = mock_jax_deps["gemma_model"].return_value
    mock_model.__call__ = MagicMock(return_value="logits")

    # Mock beam search
    output_seq = MagicMock()
    output_seq.tolist.return_value = [1, 2, 3]
    output_seq.__len__.return_value = 3
    mock_jax_beam_search = MagicMock(return_value=([output_seq], 3.0))

    # Mock Path to simulate checkpoint loading
    mock_path = MagicMock()
    mock_path.exists.return_value = True

    with patch("gemma_4_sql.backends.jax.inference.jax_beam_search", mock_jax_beam_search), patch("pathlib.Path", return_value=mock_path):
        res = generate_sql("my_model", "my prompt")

        assert res["status"] == "success"
        assert res["backend"] == "jax"
        assert "my_model" in _MODEL_CACHE

        # Test caching and model_forward
        # Second call should use cache
        mock_model_forward = mock_jax_beam_search.call_args[0][0]

        generate_sql("my_model", "my prompt")
    assert mock_jax_deps["gemma_model"].call_count == 1  # Only called once

    # Call model forward
    mock_model_forward("seq", "pos")
    mock_model.assert_called_with("seq", "pos")

    # Type error branch in model_forward
    mock_model.side_effect = [TypeError(""), "logits"]
    mock_model_forward("seq", "pos")


def test_generate_sql_multimodal(mock_jax_deps):
    mock_model = mock_jax_deps["gemma_model"].return_value
    mock_model.__call__ = MagicMock(return_value="logits")

    output_seq = MagicMock()
    output_seq.tolist.return_value = [1, 2, 3]
    del output_seq.__len__

    output_ids = MagicMock()
    output_ids.__getitem__.return_value = output_seq
    output_ids.shape = (1, 3)

    mock_jax_beam_search = MagicMock(return_value=(output_ids, 3.0))

    mock_path = MagicMock()
    mock_path.exists.return_value = False

    mock_format = MagicMock(return_value={"prompt": "multimodal prompt"})
    mock_process_image = MagicMock(return_value={"pixel_values": [0.1, 0.2]})
    mock_process_audio = MagicMock(return_value={"audio_values": [0.3, 0.4]})

    with (
        patch("gemma_4_sql.backends.jax.inference.jax_beam_search", mock_jax_beam_search),
        patch("pathlib.Path", return_value=mock_path),
        patch("gemma_4_sql.backends.common_multimodal.format_multimodal_prompt", mock_format),
        patch("gemma_4_sql.backends.common_multimodal.process_image", mock_process_image),
        patch("gemma_4_sql.backends.common_multimodal.process_audio", mock_process_audio),
    ):
        res = generate_sql("my_model", "my prompt", image_path="img.png", audio_path="aud.wav")

    assert res["status"] == "success"

    mock_model_forward = mock_jax_beam_search.call_args[0][0]
    mock_model_forward("seq", "pos")
    # assert kwargs passed to model
    _args, kwargs = mock_model.call_args
    assert "pixel_values" in kwargs
    assert "audio_values" in kwargs


def test_generate_sql_checkpoint_error(mock_jax_deps):
    mock_path = MagicMock()
    mock_path.exists.return_value = True

    mock_ocp = mock_jax_deps["ocp"]
    mock_ocp.PyTreeCheckpointer.return_value.restore.side_effect = RuntimeError("Checkpoint err")

    mock_jax_beam_search = MagicMock(return_value=([MagicMock()], 3.0))

    with patch("gemma_4_sql.backends.jax.inference.jax_beam_search", mock_jax_beam_search), patch("pathlib.Path", return_value=mock_path):
        res = generate_sql("my_model", "my prompt")

    assert res["status"] == "success"


def test_generate_sql_nnx_none(mock_jax_deps):
    mock_jax_deps["nnx"] = None
    import gemma_4_sql.backends.jax.inference as inf_module

    inf_module.nnx = None

    mock_path = MagicMock()
    mock_path.exists.return_value = False

    mock_jax_beam_search = MagicMock(return_value=([MagicMock()], 3.0))

    with patch("gemma_4_sql.backends.jax.inference.jax_beam_search", mock_jax_beam_search), patch("pathlib.Path", return_value=mock_path):
        res = generate_sql("my_model_nnx_none", "my prompt")

    assert res["status"] == "success"
    inf_module.nnx = mock_jax_deps["nnx"]  # restore


def test_generate_sql_multimodal_with_values(mock_jax_deps):
    # test branch where image_path is None but pixel_values is provided
    # meaning format_multimodal_prompt is called with has_image=True
    # and process_image is NOT called
    mock_jax_beam_search = MagicMock(return_value=([MagicMock()], 3.0))
    mock_format = MagicMock(return_value={"prompt": "multimodal prompt"})

    with patch("gemma_4_sql.backends.jax.inference.jax_beam_search", mock_jax_beam_search), patch("gemma_4_sql.backends.common_multimodal.format_multimodal_prompt", mock_format):
        generate_sql("model", "prompt", image_path="fake", pixel_values="pixels", audio_values="audio")

    mock_format.assert_called_with("prompt", has_image=True, has_audio=True)


def test_jax_beam_search_max_length():
    model_apply_fn = MagicMock()

    seq_not_eos = MagicMock()
    seq_not_eos.__getitem__.return_value = 1

    # Always return itself with lower score to avoid infinite loops, but max_length breaks it
    def mock_step(seq, score, fn, bw):
        return [(seq_not_eos, score - 0.1)]

    with patch("gemma_4_sql.backends.jax.inference._beam_search_step", side_effect=mock_step):
        res_seq, res_score = jax_beam_search(model_apply_fn, seq_not_eos, beam_width=1, max_length=2, eos_token_id=99)

    assert res_seq == seq_not_eos
    assert res_score == -0.2


def test_generate_sql_checkpoint_no_update(mock_jax_deps):
    mock_path = MagicMock()
    mock_path.exists.return_value = True

    mock_ocp = mock_jax_deps["ocp"]
    # return restored as None
    mock_ocp.PyTreeCheckpointer.return_value.restore.return_value = None

    mock_jax_beam_search = MagicMock(return_value=([MagicMock()], 3.0))

    with patch("gemma_4_sql.backends.jax.inference.jax_beam_search", mock_jax_beam_search), patch("pathlib.Path", return_value=mock_path):
        res = generate_sql("my_model_none", "my prompt")

    assert res["status"] == "success"
