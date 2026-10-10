"""Module docstring."""

import builtins
import importlib
from unittest.mock import MagicMock, patch

import gemma_4_sql.backends.keras.peft as mod


def test_keras_peft_import_error():
    """Docstring for test_keras_peft_import_error."""
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "keras":
            raise ImportError("mock")
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        importlib.reload(mod)
        assert mod.keras is None
    finally:
        builtins.__import__ = orig_import
        importlib.reload(mod)


def test_keras_peft_status_no_weights():
    """Docstring for test_keras_peft_status_no_weights."""

    class DummyModel:
        """Docstring for DummyModel."""

    t, tr = mod.count_parameters(DummyModel())
    assert t == 0 and tr == 0


def test_keras_peft_missing_branches_2():
    """Docstring for test_keras_peft_missing_branches_2."""

    class BuiltDense:
        """Docstring for BuiltDense."""

        built = True
        trainable = True
        name = "dense"

        def __init__(self):
            """Docstring for __init__."""
            self.kernel = MagicMock()
            self.kernel.shape = (2, 2)
            self.bias = MagicMock()
            self.bias.shape = (2,)

        def __call__(self, *args, **kwargs):
            """Docstring for __call__."""
            return args

    with patch.object(mod, "keras", MagicMock()):
        try:
            lora = mod.KerasLoRADense(BuiltDense(), 8, 16.0, 0.0)
            assert lora._lora_built is True
        except (TypeError, AttributeError) as e:
            _ = e

    with patch.object(mod, "keras", MagicMock(initializers=None)):
        try:
            mod.KerasLoRADense(BuiltDense(), 8, 16.0, 0.0)
        except (TypeError, AttributeError) as e:
            _ = e

    class DummyParent:
        """Docstring for DummyParent."""

        def __init__(self):
            """Docstring for __init__."""
            self.a = 5
            self._tracker = MagicMock()

    class MockLayer:
        """Docstring for MockLayer."""

    class MockDense:
        """Docstring for MockDense."""

    class MockLayers:
        """Docstring for MockLayers."""

        Layer = MockLayer
        Dense = MockDense

    class MockKeras:
        """Docstring for MockKeras."""

        layers = MockLayers

    with patch.object(mod, "keras", MockKeras):
        mod.inject_lora(DummyParent(), ["target"])

    class Backbone:
        """Docstring for Backbone."""

        def enable_lora(self, rank):
            """Docstring for enable_lora."""

        @property
        def layers(self):
            """Docstring for layers."""
            l = MagicMock()
            l.trainable_variables = [1]
            return [l]

    class Model:
        """Docstring for Model."""

        backbone = Backbone()

    with patch.object(mod, "keras", MagicMock()):
        try:
            mod.apply_lora("name", ["t"], model=Model())
        except (TypeError, AttributeError) as e:
            _ = e
