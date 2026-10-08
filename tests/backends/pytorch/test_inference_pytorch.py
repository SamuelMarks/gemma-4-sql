"""Real PyTorch inference tests."""

import pytest
import torch

from gemma_4_sql.backends.pytorch.inference import generate_sql
from gemma_4_sql.exceptions import DependencyMissingError


def test_generate_sql_success() -> None:
    """Docstring for test_generate_sql_success."""
    # Just test that the pipeline can run using pytorch_native mock config via kwargs
    # We will pass backend_alias="pytorch_native" to use the internal model
    from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config

    config = Gemma4Config(
        vocab_size=128,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        head_dim=8,
    )
    res = generate_sql(
        model_name="dummy_model",
        prompt="SELECT *",
        beam_width=1,
        max_length=2,
        backend_alias="pytorch_native",
        config=config,
    )
    assert res["status"] == "success"
    assert "sql" in res


def test_generate_sql_multimodal() -> None:
    """Docstring for test_generate_sql_multimodal."""
    from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config

    config = Gemma4Config(
        vocab_size=128,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        head_dim=8,
    )
    res = generate_sql(
        model_name="dummy_model",
        prompt="SELECT *",
        beam_width=1,
        max_length=2,
        backend_alias="pytorch_native",
        config=config,
        pixel_values=torch.randn(1, 3, 224, 224),
        audio_values=torch.randn(1, 16000),
    )
    assert res["status"] == "success"


def test_generate_sql_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Docstring for test_generate_sql_hf."""

    class MockModel:
        """Docstring for MockModel."""

        @classmethod
        def from_pretrained(cls, *a, **k):
            """Docstring for from_pretrained."""
            return cls()

        def to(self, *a, **k):
            """Docstring for to."""
            return self

        def eval(self):
            """Docstring for eval."""

        def generate(self, **kwargs):
            """Docstring for generate."""

            class Outputs:
                """Docstring for Outputs."""

                sequences = torch.tensor([[1, 2, 3]])
                sequences_scores = torch.tensor([-0.5])

            return Outputs()

    class MockTokenizer:
        """Docstring for MockTokenizer."""

        @classmethod
        def from_pretrained(cls, *a, **k):
            """Docstring for from_pretrained."""
            return cls()

        def __call__(self, text, return_tensors):
            """Docstring for __call__."""

            class Inputs(dict):
                """Docstring for Inputs."""

                def __init__(self):
                    """Docstring for __init__."""
                    super().__init__({"input_ids": torch.tensor([[1]])})
                    self.input_ids = self["input_ids"]

                def to(self, *a):
                    """Docstring for to."""
                    return self

            return Inputs()

        def decode(self, *a, **k):
            """Docstring for decode."""
            return "SELECT * FROM t"

    import gemma_4_sql.backends.pytorch.inference as inf

    monkeypatch.setattr(inf, "AutoModelForCausalLM", MockModel)
    monkeypatch.setattr(inf, "AutoTokenizer", MockTokenizer)

    res = generate_sql(
        model_name="hf_model",
        prompt="SELECT *",
        beam_width=1,
        max_length=2,
        backend_alias="pytorch",
        pixel_values=torch.randn(1, 3, 224, 224),
        audio_values=torch.randn(1, 16000),
    )
    assert res["status"] == "success"


def test_generate_sql_hf_empty_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Docstring for test_generate_sql_hf_empty_sequence."""

    class MockModel:
        """Docstring for MockModel."""

        @classmethod
        def from_pretrained(cls, *a, **k):
            """Docstring for from_pretrained."""
            return cls()

        def to(self, *a, **k):
            """Docstring for to."""
            return self

        def eval(self):
            """Docstring for eval."""

        def generate(self, **kwargs):
            """Docstring for generate."""

            class Outputs:
                """Docstring for Outputs."""

                sequences = torch.tensor([[1]])  # Only prompt, no generation
                sequences_scores = torch.tensor([0.9])

            return Outputs()

    class MockTokenizer:
        """Docstring for MockTokenizer."""

        @classmethod
        def from_pretrained(cls, *a, **k):
            """Docstring for from_pretrained."""
            return cls()

        def __call__(self, text, return_tensors):
            """Docstring for __call__."""

            class Inputs(dict):
                """Docstring for Inputs."""

                def __init__(self):
                    """Docstring for __init__."""
                    super().__init__({"input_ids": torch.tensor([[1]])})
                    self.input_ids = self["input_ids"]

                def to(self, *a):
                    """Docstring for to."""
                    return self

            return Inputs()

        def decode(self, *a, **k):
            """Docstring for decode."""
            return ""  # empty sql

    import gemma_4_sql.backends.pytorch.inference as inf

    monkeypatch.setattr(inf, "AutoModelForCausalLM", MockModel)
    monkeypatch.setattr(inf, "AutoTokenizer", MockTokenizer)

    res = generate_sql(
        model_name="hf_model",
        prompt="SELECT *",
        beam_width=1,
        max_length=2,
        backend_alias="pytorch",
    )
    assert "failed" in res["status"]


def test_generate_sql_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Docstring for test_generate_sql_missing_deps."""
    import gemma_4_sql.backends.pytorch.inference as inf

    monkeypatch.setattr(inf, "torch", None)
    with pytest.raises(DependencyMissingError):
        generate_sql("model", "prompt")


def test_inference_pytorch_no_attrs(monkeypatch, tmp_path):
    """Docstring for test_inference_pytorch_no_attrs."""
    from gemma_4_sql.backends.pytorch.inference import generate_sql

    class DummyModelNoAttrs:
        """Docstring for DummyModelNoAttrs."""

        def generate(self, *args, **kwargs):
            """Docstring for generate."""
            import torch

            class MockOutput:
                """Docstring for MockOutput."""

                sequences = torch.zeros((1, 1, 10))

            return MockOutput()

    class MockAutoModelForCausalLM:
        """Docstring for MockAutoModelForCausalLM."""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            """Docstring for from_pretrained."""
            return DummyModelNoAttrs()

    class MockTokenizer:
        """Docstring for MockTokenizer."""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            """Docstring for from_pretrained."""

            class Tok:
                """Docstring for Tok."""

                def __call__(self, *a, **k):
                    """Docstring for __call__."""
                    import torch

                    return {"input_ids": torch.tensor([[1]])}

                def decode(self, *a, **k):
                    """Docstring for decode."""
                    return "SELECT 1"

            return Tok()

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoModelForCausalLM", MockAutoModelForCausalLM)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoTokenizer", MockTokenizer)

    res = generate_sql("model", "query", beam_width=1, max_length=1)
    assert res["status"] == "success"


def test_inference_pytorch_dict_output(monkeypatch, tmp_path):
    """Docstring for test_inference_pytorch_dict_output."""
    from gemma_4_sql.backends.pytorch.inference import generate_sql

    class DummyModelDictOutput:
        """Docstring for DummyModelDictOutput."""

        def eval(self):
            """Docstring for eval."""

        def generate(self, *args, **kwargs):
            """Docstring for generate."""
            import torch

            class MockOutput:
                """Docstring for MockOutput."""

                sequences = torch.ones((1, 10))

            return MockOutput()

    class MockAutoModelForCausalLM:
        """Docstring for MockAutoModelForCausalLM."""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            """Docstring for from_pretrained."""
            return DummyModelDictOutput()

    class MockTokenizer:
        """Docstring for MockTokenizer."""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            """Docstring for from_pretrained."""

            class Tok:
                """Docstring for Tok."""

                def __call__(self, *a, **k):
                    """Docstring for __call__."""
                    import torch

                    return {"input_ids": torch.tensor([[1]])}

                def decode(self, *a, **k):
                    """Docstring for decode."""
                    return "SELECT 1"

            return Tok()

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoModelForCausalLM", MockAutoModelForCausalLM)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoTokenizer", MockTokenizer)

    res = generate_sql("model", "query", beam_width=1, max_length=1)
    assert res["status"] == "success"


def test_inference_pytorch_tokenizer_err(monkeypatch, tmp_path):
    """Docstring for test_inference_pytorch_tokenizer_err."""
    from gemma_4_sql.backends.pytorch.inference import generate_sql

    class DummyModelDictOutput:
        """Docstring for DummyModelDictOutput."""

        def eval(self):
            """Docstring for eval."""

        def generate(self, *args, **kwargs):
            """Docstring for generate."""
            import torch

            class MockOutput:
                """Docstring for MockOutput."""

                sequences = torch.ones((1, 10))

            return MockOutput()

    class MockAutoModelForCausalLM:
        """Docstring for MockAutoModelForCausalLM."""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            """Docstring for from_pretrained."""
            return DummyModelDictOutput()

    class MockTokenizer:
        """Docstring for MockTokenizer."""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            """Docstring for from_pretrained."""
            raise OSError("err")

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoModelForCausalLM", MockAutoModelForCausalLM)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoTokenizer", MockTokenizer)

    res = generate_sql("model", "query", beam_width=1, max_length=1)
    assert "failed" in res["status"]


def test_inference_pytorch_empty_gen(monkeypatch, tmp_path):
    """Docstring for test_inference_pytorch_empty_gen."""
    from gemma_4_sql.backends.pytorch.inference import generate_sql

    class DummyModelEmpty:
        """Docstring for DummyModelEmpty."""

        def eval(self):
            """Docstring for eval."""

        def generate(self, *args, **kwargs):
            """Docstring for generate."""
            import torch

            class MockOutput:
                """Docstring for MockOutput."""

                sequences = torch.tensor([[1]])

            return MockOutput()

    class MockAutoModelForCausalLM:
        """Docstring for MockAutoModelForCausalLM."""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            """Docstring for from_pretrained."""
            return DummyModelEmpty()

    class MockTokenizer:
        """Docstring for MockTokenizer."""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            """Docstring for from_pretrained."""

            class Tok:
                """Docstring for Tok."""

                def __call__(self, *a, **k):
                    """Docstring for __call__."""
                    import torch

                    return {"input_ids": torch.tensor([[1]])}

                def decode(self, *a, **k):
                    """Docstring for decode."""
                    return ""

            return Tok()

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoModelForCausalLM", MockAutoModelForCausalLM)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoTokenizer", MockTokenizer)

    res = generate_sql("model", "query", beam_width=1, max_length=1)
    assert "failed" in str(res["status"])


def test_inference_pytorch_empty_sql(monkeypatch, tmp_path):
    """Docstring for test_inference_pytorch_empty_sql."""
    from gemma_4_sql.backends.pytorch.inference import generate_sql

    class DummyModelEmpty:
        """Docstring for DummyModelEmpty."""

        def eval(self):
            """Docstring for eval."""

        def generate(self, *args, **kwargs):
            """Docstring for generate."""
            import torch

            class MockOutput:
                """Docstring for MockOutput."""

                sequences = torch.tensor([[1, 2]])

            return MockOutput()

    class MockAutoModelForCausalLM:
        """Docstring for MockAutoModelForCausalLM."""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            """Docstring for from_pretrained."""
            return DummyModelEmpty()

    class MockTokenizer:
        """Docstring for MockTokenizer."""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            """Docstring for from_pretrained."""

            class Tok:
                """Docstring for Tok."""

                def __call__(self, *a, **k):
                    """Docstring for __call__."""
                    import torch

                    return {"input_ids": torch.tensor([[1]])}

                def decode(self, *a, **k):
                    """Docstring for decode."""
                    return ""

            return Tok()

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoModelForCausalLM", MockAutoModelForCausalLM)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoTokenizer", MockTokenizer)

    res = generate_sql("model", "query", beam_width=1, max_length=1)
    assert "failed" in str(res["status"])


def test_inference_pytorch_multimodal_edge_cases(monkeypatch, tmp_path):
    """Docstring for test_inference_pytorch_multimodal_edge_cases."""
    from gemma_4_sql.backends.pytorch.inference import generate_sql

    class DummyModel:
        """Docstring for DummyModel."""

        device = "cpu"

        def eval(self):
            """Docstring for eval."""

        def generate(self, *args, **kwargs):
            """Docstring for generate."""
            import torch

            class MockOutput:
                """Docstring for MockOutput."""

                sequences = torch.tensor([[1, 2]])

            return MockOutput()

    class MockAutoModelForCausalLM:
        """Docstring for MockAutoModelForCausalLM."""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            """Docstring for from_pretrained."""
            return DummyModel()

    class MockTokenizer:
        """Docstring for MockTokenizer."""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            """Docstring for from_pretrained."""

            class Tok:
                """Docstring for Tok."""

                def __call__(self, *a, **k):
                    """Docstring for __call__."""
                    import torch

                    return {"input_ids": torch.tensor([[1]])}

                def decode(self, *a, **k):
                    """Docstring for decode."""
                    return "SELECT 1"

            return Tok()

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoModelForCausalLM", MockAutoModelForCausalLM)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoTokenizer", MockTokenizer)

    dummy_img = tmp_path / "img.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    dummy_audio = tmp_path / "aud.wav"
    dummy_audio.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    res1 = generate_sql("model", "query", beam_width=1, max_length=1, image_path=str(dummy_img))
    res2 = generate_sql("model", "query", beam_width=1, max_length=1, audio_path=str(dummy_audio))
    res3 = generate_sql("model", "query", beam_width=1, max_length=1, adapter_path="dummy/path")
    assert res1["status"] == "success"
    assert res2["status"] == "success"
    assert res3["status"] == "success"


def test_inference_pytorch_pixel_in_inputs(monkeypatch, tmp_path):
    """Docstring for test_inference_pytorch_pixel_in_inputs."""
    from gemma_4_sql.backends.pytorch.inference import generate_sql

    class DummyModel:
        """Docstring for DummyModel."""

        device = "cpu"

        def eval(self):
            """Docstring for eval."""

        def generate(self, *args, **kwargs):
            """Docstring for generate."""
            import torch

            class MockOutput:
                """Docstring for MockOutput."""

                sequences = torch.tensor([[1, 2]])

            return MockOutput()

    class MockAutoModelForCausalLM:
        """Docstring for MockAutoModelForCausalLM."""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            """Docstring for from_pretrained."""
            return DummyModel()

    class MockTokenizer:
        """Docstring for MockTokenizer."""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            """Docstring for from_pretrained."""

            class Tok:
                """Docstring for Tok."""

                def __call__(self, *a, **k):
                    """Docstring for __call__."""
                    import torch

                    return {"input_ids": torch.tensor([[1]]), "pixel_values": 1}

                def decode(self, *a, **k):
                    """Docstring for decode."""
                    return "SELECT 1"

            return Tok()

    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoModelForCausalLM", MockAutoModelForCausalLM)
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.inference.AutoTokenizer", MockTokenizer)

    res = generate_sql("model", "query", beam_width=1, max_length=1)
    assert res["status"] == "success"
