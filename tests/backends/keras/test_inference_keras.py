"""Module docstring."""

from unittest.mock import MagicMock

import pytest


def test_extract_flat_scores():
    """Docstring for test_extract_flat_scores."""
    import gemma_4_sql.backends.keras.inference as inf

    # list pop(0) in BFS manner
    # [[1.0, 2], 3.0] -> pop [1.0, 2], stack=[3.0, 1.0, 2] -> pop 3.0 -> flat=[3.0], stack=[1.0, 2]
    # -> pop 1.0 -> flat=[3.0, 1.0], stack=[2] -> pop 2 -> flat=[3.0, 1.0, 2.0]
    res = inf._extract_flat_scores([[1.0, 2], 3.0])
    assert res == [3.0, 1.0, 2.0]


def test_compute_keras_confidence():
    """Docstring for test_compute_keras_confidence."""
    import math

    import gemma_4_sql.backends.keras.inference as inf

    assert inf.compute_keras_confidence(None, 0) == 0.0

    mock_np = MagicMock()
    mock_np.numpy.return_value = [0.1, 0.2]
    mock_np.tolist.return_value = [0.1, 0.2]

    assert inf.compute_keras_confidence(mock_np, 2) == pytest.approx(0.15)

    # negative log probs
    assert inf.compute_keras_confidence([-1.0, -2.0], 2) == pytest.approx(math.exp(-1.5))

    # scalar positive
    assert inf.compute_keras_confidence(0.5, 2) == 0.5
    # scalar negative
    assert inf.compute_keras_confidence(-1.0, 2) == pytest.approx(math.exp(-0.5))

    # empty list
    assert inf.compute_keras_confidence([], 2) == 0.5

    # fallback
    assert inf.compute_keras_confidence(None, 10) > 0.0


def test_configure_beam_sampler(monkeypatch):
    """Docstring for test_configure_beam_sampler."""
    import gemma_4_sql.backends.keras.inference as inf

    mock_model = MagicMock()
    # successful import
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "keras_nlp":
            mock_nlp = MagicMock()
            mock_sampler_cls = MagicMock(return_value="sampler_obj")
            mock_nlp.samplers.BeamSampler = mock_sampler_cls
            return mock_nlp
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    # with compile
    inf.configure_beam_sampler(mock_model, 2)
    mock_model.compile.assert_called_with(sampler="sampler_obj")

    # with sampler attribute
    mock_model2 = MagicMock(spec=["sampler"])
    inf.configure_beam_sampler(mock_model2, 2)
    assert mock_model2.sampler == "sampler_obj"

    # import error
    def mock_import_err(name, *args, **kwargs):
        """Docstring for mock_import_err."""
        if name == "keras_nlp":
            raise ImportError("sim")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import_err)

    res = inf.configure_beam_sampler(MagicMock(), 2)
    assert res is None


def test_generate_sql(monkeypatch):
    """Docstring for test_generate_sql."""
    import gemma_4_sql.backends.keras.inference as inf
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(inf, "keras", None)
    with pytest.raises(DependencyMissingError):
        inf.generate_sql("m", "p")

    monkeypatch.setattr(inf, "keras", MagicMock())
    monkeypatch.setattr(inf, "tf", MagicMock())
    monkeypatch.setattr(inf, "configure_beam_sampler", MagicMock())

    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "keras_nlp.models":
            mock_models = MagicMock()
            mock_cls = MagicMock()
            mock_model = MagicMock()
            mock_model.generate.return_value = "prompt SELECT 1"
            mock_cls.from_preset.return_value = mock_model
            mock_models.GemmaCausalLM = mock_cls
            return mock_models
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    res = inf.generate_sql("m", "prompt ")
    assert res["status"] == "success"
    assert res["sql"] == "SELECT 1"


def test_generate_sql_outputs(monkeypatch):
    """Docstring for test_generate_sql_outputs."""
    import gemma_4_sql.backends.keras.inference as inf

    monkeypatch.setattr(inf, "keras", MagicMock())
    monkeypatch.setattr(inf, "tf", MagicMock())
    monkeypatch.setattr(inf, "configure_beam_sampler", MagicMock())

    def setup_mock(ret_val):
        """Docstring for setup_mock."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            """Docstring for mock_import."""
            if name == "keras_nlp.models":
                mock_models = MagicMock()
                mock_cls = MagicMock()
                mock_model = MagicMock()
                mock_model.generate.return_value = ret_val
                mock_cls.from_preset.return_value = mock_model
                mock_models.GemmaCausalLM = mock_cls
                return mock_models
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

    # dict output
    setup_mock({"text": "prompt SEL", "scores": 0.5})
    res = inf.generate_sql("m", "prompt ")
    assert res["status"] == "success"

    # tuple output
    setup_mock(("prompt SEL", 0.5))
    res2 = inf.generate_sql("m", "prompt ")
    assert res2["status"] == "success"

    # empty output
    setup_mock("prompt ")
    res3 = inf.generate_sql("m", "prompt ")
    assert "failed" in res3["status"]


def test_extract_flat_scores_missing_branch():
    """Docstring for test_extract_flat_scores_missing_branch."""
    import gemma_4_sql.backends.keras.inference as inf

    res = inf._extract_flat_scores([[1.0, 2], "skip_me", None, 3.0])
    assert res == [3.0, 1.0, 2.0]


def test_configure_beam_sampler_branches(monkeypatch):
    """Docstring for test_configure_beam_sampler_branches."""
    from unittest.mock import MagicMock

    import gemma_4_sql.backends.keras.inference as inf

    mock_model = MagicMock()
    del mock_model.compile
    del mock_model.sampler

    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "keras_nlp":
            mock_nlp = MagicMock()
            mock_nlp.samplers = MagicMock()
            mock_nlp.samplers.BeamSampler = None
            return mock_nlp
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    inf.configure_beam_sampler(mock_model, 2)

    def mock_import_2(name, *args, **kwargs):
        """Docstring for mock_import_2."""
        if name == "keras_nlp":
            mock_nlp = MagicMock()
            mock_nlp.samplers.BeamSampler = MagicMock(return_value="sampler_obj")
            return mock_nlp
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import_2)

    inf.configure_beam_sampler(mock_model, 2)


def test_generate_sql_outputs_else_block(monkeypatch):
    """Docstring for test_generate_sql_outputs_else_block."""
    from unittest.mock import MagicMock

    import gemma_4_sql.backends.keras.inference as inf

    monkeypatch.setattr(inf, "keras", MagicMock())
    monkeypatch.setattr(inf, "tf", MagicMock())
    monkeypatch.setattr(inf, "configure_beam_sampler", MagicMock())

    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "keras_nlp.models":
            mock_models = MagicMock()
            mock_cls = MagicMock()
            mock_model = MagicMock()
            mock_model.generate.return_value = 12345
            mock_cls.from_preset.return_value = mock_model
            mock_models.GemmaCausalLM = mock_cls
            return mock_models
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    res = inf.generate_sql("m", "")
    assert res["status"] == "success"
    assert "12345" in res["sql"]


def test_module_load_import_error():
    """Docstring for test_module_load_import_error."""
    import gemma_4_sql.backends.keras.inference as q

    with open(q.__file__) as f:
        code = f.read()

    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name in ("keras", "tensorflow"):
            raise ImportError("simulated missing import")
        return orig_import(name, *args, **kwargs)

    namespace = {"__name__": "mock_inference", "__builtins__": dict(builtins.__dict__)}
    namespace["__builtins__"]["__import__"] = mock_import

    exec(code, namespace)  # noqa: S102

    assert namespace.get("keras") is None
    assert namespace.get("tf") is None


def test_module_load_import_error_coverage(monkeypatch):
    """Docstring for test_module_load_import_error_coverage."""
    import importlib
    import sys

    import gemma_4_sql.backends.keras.inference as inf

    monkeypatch.setitem(sys.modules, "keras", None)
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    importlib.reload(inf)
    assert inf.keras is None
    assert inf.tf is None
    monkeypatch.undo()
    importlib.reload(inf)
