"""Module docstring."""

import gemma_4_sql.backends.keras.peft as pt


def test_peft_properties():
    """Docstring for test_peft_properties."""

    class DummyDense:
        """Docstring for DummyDense."""

        bias = "bias"

    layer = pt.KerasLoRADense(dense=DummyDense(), r=4)
    assert layer.bias == "bias"


def test_peft_count_params(monkeypatch):
    """Docstring for test_peft_count_params."""

    class DummyModel:
        """Docstring for DummyModel."""

        weights = (1,)
        trainable_weights = (1,)

    monkeypatch.setattr(pt.ops, "size", lambda w: 100)
    t, f = pt.count_parameters(DummyModel())
    assert t == 100
    assert f == 100


def test_peft_count_params_not_trainable(monkeypatch):
    """Docstring for test_peft_count_params_not_trainable."""

    class DummyModel:
        """Docstring for DummyModel."""

        weights = (1,)
        trainable_weights = ()

    monkeypatch.setattr(pt.ops, "size", lambda w: 100)
    t, f = pt.count_parameters(DummyModel())
    assert t == 100
    assert f == 0


def test_peft_list_modifier():
    """Docstring for test_peft_list_modifier."""
    import keras

    class DummyModel:
        """Docstring for DummyModel."""

        def __init__(self):
            """Docstring for __init__."""
            self.layers = [keras.layers.Dense(64, name="target")]

    m = DummyModel()
    m.layers[0].build((None, 10))
    pt.inject_lora(m, ["target"])


def test_peft_apply_lora_save_path(monkeypatch):
    """Docstring for test_peft_apply_lora_save_path."""
    import sys

    import gemma_4_sql.backends.keras.peft as pt

    class MockModel:
        """Docstring for MockModel."""

        def save(self, path):
            """Docstring for save."""

        def save_weights(self, path):
            """Docstring for save_weights."""

    class MockGemma:
        """Docstring for MockGemma."""

        @classmethod
        def from_preset(cls, *args, **kwargs):
            """Docstring for from_preset."""
            return MockModel()

    monkeypatch.setitem(sys.modules, "keras_nlp.models", type("models", (), {"GemmaCausalLM": MockGemma}))
    monkeypatch.setattr(pt, "inject_lora", lambda model, *a, **k: (model, 1))
    monkeypatch.setattr(pt, "count_parameters", lambda model: (100, 10))
    res = pt.apply_lora("dummy", ["q"], output_dir="dummy/path")
    assert res["status"] == "completed"


def test_peft_apply_kwargs():
    """Docstring for test_peft_apply_kwargs."""

    class DummyModel:
        """Docstring for DummyModel."""

    pt.apply_lora("dummy", ["q"], model=DummyModel())


def test_peft_apply_merge():
    """Docstring for test_peft_apply_merge."""

    class DummyModel:
        """Docstring for DummyModel."""

    pt.apply_lora("dummy", ["q"], merge=True)


def test_peft_apply_native_lora():
    """Docstring for test_peft_apply_native_lora."""

    class Backbone:
        """Docstring for Backbone."""

        def enable_lora(self, rank):
            """Docstring for enable_lora."""

        layers = ()

    class DummyModel:
        """Docstring for DummyModel."""

        backbone = Backbone()

    pt.apply_lora("dummy", ["q"], model=DummyModel())
