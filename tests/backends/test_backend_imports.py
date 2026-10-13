import sys


def test_keras_etl_imports(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setitem(sys.modules, "datasets", MagicMock())
    monkeypatch.setitem(sys.modules, "grain.python", MagicMock())
    monkeypatch.setitem(sys.modules, "grain", MagicMock())

    # Reload the module to hit the try block
    import importlib

    import gemma_4_sql.backends.keras.etl as ketl

    importlib.reload(ketl)
    assert ketl.datasets is not None


def test_maxtext_etl_imports(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setitem(sys.modules, "datasets", MagicMock())
    monkeypatch.setitem(sys.modules, "grain.python", MagicMock())
    monkeypatch.setitem(sys.modules, "grain", MagicMock())

    # Reload the module to hit the try block
    import importlib

    import gemma_4_sql.backends.maxtext.etl as metl

    importlib.reload(metl)
    assert metl.datasets is not None


def test_mlx_etl_imports(monkeypatch):
    import sys
    from unittest.mock import MagicMock

    monkeypatch.setitem(sys.modules, "datasets", MagicMock())

    import importlib

    import gemma_4_sql.backends.mlx.etl as mlx_etl

    importlib.reload(mlx_etl)
    assert mlx_etl.datasets is not None


def test_jax_inference_imports(monkeypatch):
    import sys
    from unittest.mock import MagicMock

    monkeypatch.setitem(sys.modules, "flax.nnx", MagicMock())
    monkeypatch.setitem(sys.modules, "flax", MagicMock())
    import importlib

    import gemma_4_sql.backends.jax.inference as jinf

    importlib.reload(jinf)
    assert jinf.nnx is not None
    import sys
    from unittest.mock import MagicMock

    monkeypatch.setitem(sys.modules, "flax.nnx", MagicMock())
    monkeypatch.setitem(sys.modules, "flax", MagicMock())

    import importlib

    import gemma_4_sql.backends.jax.inference as jinf

    importlib.reload(jinf)
    assert jinf.nnx is not None
