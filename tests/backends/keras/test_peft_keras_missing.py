import gemma_4_sql.backends.keras.peft as pt


def test_peft_properties():
    class DummyDense:
        bias = "bias"

    layer = pt.KerasLoRADense(dense=DummyDense(), r=4)
    assert layer.bias == "bias"


def test_peft_count_params(monkeypatch):
    class DummyModel:
        weights = (1,)
        trainable_weights = (1,)

    monkeypatch.setattr(pt.ops, "size", lambda w: 100)
    t, f = pt.count_parameters(DummyModel())
    assert t == 100
    assert f == 100


def test_peft_count_params_not_trainable(monkeypatch):
    class DummyModel:
        weights = (1,)
        trainable_weights = ()

    monkeypatch.setattr(pt.ops, "size", lambda w: 100)
    t, f = pt.count_parameters(DummyModel())
    assert t == 100
    assert f == 0


def test_peft_list_modifier():
    import keras

    class DummyModel:
        def __init__(self):
            self.layers = [keras.layers.Dense(64, name="target")]

    m = DummyModel()
    m.layers[0].build((None, 10))
    pt.inject_lora(m, ["target"])


def test_peft_apply_lora_save_path(monkeypatch):
    import sys

    import gemma_4_sql.backends.keras.peft as pt

    class MockModel:
        def save(self, path):
            pass

        def save_weights(self, path):
            pass

    class MockGemma:
        @classmethod
        def from_preset(cls, *args, **kwargs):
            return MockModel()

    monkeypatch.setitem(sys.modules, "keras_nlp.models", type("models", (), {"GemmaCausalLM": MockGemma}))
    monkeypatch.setattr(pt, "inject_lora", lambda model, *a, **k: (model, 1))
    monkeypatch.setattr(pt, "count_parameters", lambda model: (100, 10))
    res = pt.apply_lora("dummy", ["q"], output_dir="dummy/path")
    assert res["status"] == "completed"


def test_peft_apply_kwargs():
    class DummyModel:
        pass

    pt.apply_lora("dummy", ["q"], model=DummyModel())


def test_peft_apply_merge():
    class DummyModel:
        pass

    pt.apply_lora("dummy", ["q"], merge=True)


def test_peft_apply_native_lora():
    class Backbone:
        def enable_lora(self, rank):
            pass

        layers = ()

    class DummyModel:
        backbone = Backbone()

    pt.apply_lora("dummy", ["q"], model=DummyModel())
