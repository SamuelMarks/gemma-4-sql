"""Test Keras DPO logic."""

import pytest

import gemma_4_sql.backends.keras.dpo as keras_dpo
from gemma_4_sql.type_hints import DPOConfig, TrainerState


class MockTf:
    """Provide class docstring."""

    def expand_dims(self, a: object, axis: int) -> object:
        """Execute function."""
        return a

    def gather(self, params: object, indices: object, batch_dims: int) -> object:
        """Execute function."""
        return params

    def squeeze(self, a: object, axis: int) -> object:
        """Execute function."""
        return a

    def cast(self, a: object, dtype: object) -> object:
        """Execute function."""
        return a

    def reduce_sum(self, a: object, axis: int) -> object:
        """Execute function."""
        return a

    def zeros(self, *_a: object, **_k: object) -> object:
        """Execute function."""
        return 0

    class nn:
        """Class docstring."""

        @staticmethod
        def log_softmax(logits: object, axis: int) -> object:
            """Execute function."""
            return logits

    class math:
        """Class docstring."""

        @staticmethod
        def log_sigmoid(x: object) -> object:
            """Test function."""
            return x

    class GradientTape:
        """Class docstring."""

        def __enter__(self):
            """Test function."""
            return self

        def __exit__(self, *a):
            """Test function."""

        def gradient(self, *a, **k):
            """Test function."""
            return []

    def function(self, fn: object) -> object:
        """Test function."""
        return fn

    int32 = "int32"
    float32 = "float32"


def test_compute_logps_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test compute logps in keras."""
    monkeypatch.setattr(keras_dpo, "tf", MockTf())

    class MockModel:
        """Class docstring."""

        def __call__(self, inputs: object) -> object:
            """Test function."""
            return inputs

    inputs = [1, 2]
    labels = [1, 2]
    model = MockModel()
    res = keras_dpo._compute_logps(model, inputs, labels)
    assert res is not None


def test_dpo_loss_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function."""
    res = keras_dpo.dpo_loss(0, 0, 0, 0)
    assert res is not None
    monkeypatch.setattr(keras_dpo, "tf", MockTf())
    res = keras_dpo.dpo_loss(0, 0, 0, 0)
    assert res is not None


def test_get_train_step_fn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function."""
    monkeypatch.setattr(keras_dpo, "tf", MockTf())

    class MockModel:
        """Class docstring."""

        def __call__(self, inputs: object) -> object:
            """Test function."""
            return inputs

    class MockOpt:
        """Class docstring."""

        def apply_gradients(self, *a, **k) -> None:
            """Test function."""

    fn = keras_dpo._get_train_step_fn(MockModel(), MockModel(), MockOpt(), 0.1)
    batch = {"chosen_inputs": 1, "chosen_labels": 1, "rejected_inputs": 1, "rejected_labels": 1}
    res = fn(batch)
    assert res is not None


def test_run_training_epochs() -> None:
    """Test function."""

    class MockLoss:
        """Class docstring."""

        def numpy(self):
            """Test function."""
            return 1.0

    state = TrainerState(dataloader=[1, 2], epochs=2, train_step=lambda *a: MockLoss())
    res = keras_dpo._run_training_epochs(state)
    assert res == 1.0


def test_run_dpo_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function."""
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(keras_dpo, "tf", None)
    with pytest.raises(DependencyMissingError, match="Keras DPO dependencies are missing."):
        keras_dpo.run_dpo(DPOConfig(model_name="m", dataset="d"))


def test_run_dpo_real_no_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function."""

    class MockKeras:
        """Class docstring."""

        def Input(self, *a, **k):
            """Test function."""
            return 1

        class layers:
            """Class docstring."""

            def Embedding(self, *a, **k):
                """Test function."""
                return lambda x: x

            def Dense(self, *a, **k):
                """Test function."""
                return lambda x: x

        def Model(self, *a, **k):
            """Test function."""

            class _Model:
                """Class docstring."""

                def __call__(self, *a, **k):
                    """Test function."""
                    return 0

            return _Model()

        class optimizers:
            """Class docstring."""

            @staticmethod
            def AdamW(*a: object, **k: object) -> object:
                """Test function."""

                class MockOpt:
                    """Class docstring."""

                    def apply_gradients(self, *a: object, **k: object) -> None:
                        """Test function."""

                return MockOpt()

    monkeypatch.setattr(keras_dpo, "keras", MockKeras())
    monkeypatch.setattr(keras_dpo, "tf", MockTf())
    monkeypatch.setattr(keras_dpo, "build_dataloader", lambda *a, **k: {})
    res = keras_dpo.run_dpo(DPOConfig(model_name="m", dataset="d"))
    if "failed" not in res["status"]:
        raise AssertionError


def test_run_dpo_real_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function."""

    class MockKeras:
        """Class docstring."""

        def Input(self, *a, **k):
            """Test function."""
            return 1

        class layers:
            """Class docstring."""

            def Embedding(self, *a, **k):
                """Test function."""
                return lambda x: x

            def Dense(self, *a, **k):
                """Test function."""
                return lambda x: x

        def Model(self, *a, **k):
            """Test function."""

            class _Model:
                """Class docstring."""

                def __call__(self, *a, **k):
                    """Test function."""
                    return 0

            return _Model()

        class optimizers:
            """Class docstring."""

            @staticmethod
            def AdamW(*a: object, **k: object) -> object:
                """Test function."""

                class MockOpt:
                    """Class docstring."""

                    def apply_gradients(self, *a: object, **k: object) -> None:
                        """Test function."""

                return MockOpt()

    monkeypatch.setattr(keras_dpo, "keras", MockKeras())
    monkeypatch.setattr(keras_dpo, "tf", MockTf())
    import builtins
    import sys

    class MockKerasLM:
        @classmethod
        def from_preset(cls, preset):
            class _Model:
                def __call__(self, *a, **k):
                    return 0

            return _Model()

    monkeypatch.setitem(sys.modules, "keras_nlp", type("MockKerasNLP", (), {}))
    monkeypatch.setitem(sys.modules, "keras_nlp.models", type("MockModels", (), {"GemmaCausalLM": MockKerasLM}))

    orig_import = builtins.__import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        """Docstring."""
        if name == "keras_nlp.models" and kwargs.get("fromlist") and "GemmaCausalLM" in kwargs["fromlist"]:
            return sys.modules["keras_nlp.models"]
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    monkeypatch.setattr(keras_dpo, "build_dataloader", lambda *a, **k: {"loader": [{"chosen_inputs": 1, "chosen_labels": 1, "rejected_inputs": 1, "rejected_labels": 1}]})
    res = keras_dpo.run_dpo(DPOConfig(model_name="m", dataset="d"))
    if "completed" not in res["status"]:
        raise AssertionError


def test_run_dpo_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function."""
    monkeypatch.setattr(keras_dpo, "keras", object())
    monkeypatch.setattr(keras_dpo, "tf", MockTf())

    def raise_err(*a, **k):
        """Test function."""
        raise ValueError("err")

    monkeypatch.setattr(keras_dpo, "_execute_dpo", raise_err)
    res = keras_dpo.run_dpo(DPOConfig(model_name="m", dataset="d"))
    if "failed" not in res["status"]:
        raise AssertionError


def test_run_dpo_invalid_dataloader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function."""
    monkeypatch.setattr(keras_dpo, "build_dataloader", lambda *a, **k: {"loader": None})

    class MockKerasLM:
        @classmethod
        def from_preset(cls, preset):
            return lambda x: 0

    import sys

    monkeypatch.setitem(sys.modules, "keras_nlp", type("MockKerasNLP", (), {}))
    monkeypatch.setitem(sys.modules, "keras_nlp.models", type("MockModels", (), {"GemmaCausalLM": MockKerasLM}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "keras_nlp.models" and kwargs.get("fromlist") and "GemmaCausalLM" in kwargs["fromlist"]:
            return sys.modules["keras_nlp.models"]
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    monkeypatch.setattr(keras_dpo, "keras", type("MockKeras", (), {"optimizers": type("Opts", (), {"AdamW": lambda **kw: None})}))
    monkeypatch.setattr(keras_dpo, "tf", MockTf())

    res = keras_dpo.run_dpo(DPOConfig(model_name="m", dataset="d"))
    assert "failed" in res["status"]
    assert "Invalid dataloader" in res["status"]
