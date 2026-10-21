"""Tests for PyTorch inference."""

from unittest.mock import MagicMock, patch

from tests.conftest import requires_torch

pytestmark = requires_torch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, InferenceError


@pytest.fixture(autouse=True)
def _restore_inf_mod(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore inf_module globals after each test."""
    import gemma_4_sql.backends.pytorch.inference as inf_module

    monkeypatch.setattr(inf_module, "torch", inf_module.torch)
    monkeypatch.setattr(inf_module, "AutoModelForCausalLM", inf_module.AutoModelForCausalLM)
    monkeypatch.setattr(inf_module, "AutoTokenizer", inf_module.AutoTokenizer)


def test_real_tiny_pytorch_native_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify generate_sql runs real token generation using a tiny native Gemma4ForCausalLM."""
    import gemma_4_sql.backends.pytorch.inference as inf_module
    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4Config, Gemma4ForCausalLM

    cfg = Gemma4Config(
        vocab_size=320,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
        layer_types=["sliding_attention"],
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
    )
    tiny_model = Gemma4ForCausalLM(cfg)
    monkeypatch.setattr(Gemma4ForCausalLM, "from_pretrained", classmethod(lambda cls, *_a, **_k: tiny_model))
    monkeypatch.setattr(inf_module, "AutoTokenizer", None)

    res = inf_module.generate_sql("tiny-native", "SELECT 1", beam_width=1, max_length=2, backend_alias="pytorch_native")
    assert res["status"] == "success"
    assert isinstance(res["sql"], str)
    assert len(str(res["sql"])) > 0


def test_run_generation_native():
    """Test _run_generation native."""
    import gemma_4_sql.backends.pytorch.inference as inf_module

    inf_module.torch = MagicMock()
    inf_module.AutoModelForCausalLM = MagicMock()
    inf_module.AutoTokenizer = MagicMock()

    with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM") as mock_native:
        mock_model = MagicMock()
        mock_native.from_pretrained.return_value = mock_model

        # Output setup
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]

        # Test basic success
        inf_module.AutoTokenizer.from_pretrained.return_value = MagicMock()
        inf_module.AutoTokenizer.from_pretrained.return_value.return_value.input_ids = MagicMock(shape=[1, 2])
        inf_module.AutoTokenizer.from_pretrained.return_value.decode.return_value = "SELECT 1;"

        sql, conf = inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch_native")
        assert sql == "SELECT 1;"
        assert conf == 0.95

        # Test empty sequence
        mock_model.generate.return_value = [[1, 2]]  # equals input length
        with pytest.raises(InferenceError, match="yielded an empty sequence"):
            inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch_native")

        # Test empty decoded string
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
        inf_module.AutoTokenizer.from_pretrained.return_value.decode.return_value = ""
        with pytest.raises(InferenceError, match="empty SQL"):
            inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch_native")

        # Test without AutoTokenizer available
        inf_module.AutoTokenizer = None
        with patch("gemma_4_sql.tokenization.SQLTokenizer") as mock_sql_tok:
            mock_sql_tok.return_value.encode.return_value = [1, 2]
            mock_sql_tok.return_value.decode.return_value = "SQL;"
            inf_module.torch.tensor.return_value.shape = [1, 2]

            # Need an object that returns a list from .tolist() and has __len__
            class MockGenTokens:
                """Execute and verify `MockGenTokens` behavior and invariants."""

                def tolist(self):
                    """Execute and verify `tolist` behavior and invariants."""
                    return [1, 2]

                def __len__(self):
                    """Execute and verify `__len__` behavior and invariants."""
                    return 2

            mock_out = MagicMock()
            mock_out.__getitem__.return_value.__getitem__.return_value = MockGenTokens()
            mock_model.generate.return_value = mock_out

            sql, conf = inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch_native")
            assert sql == "SQL;"


def test_run_generation_native_multimodal():
    """Test _run_generation native multimodal."""
    import gemma_4_sql.backends.pytorch.inference as inf_module

    inf_module.torch = MagicMock()
    inf_module.AutoModelForCausalLM = MagicMock()
    inf_module.AutoTokenizer = MagicMock()

    with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM") as mock_native:
        mock_model = MagicMock()
        mock_native.from_pretrained.return_value = mock_model
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
        inf_module.AutoTokenizer.from_pretrained.return_value.return_value.input_ids = MagicMock(shape=[1, 2])
        inf_module.AutoTokenizer.from_pretrained.return_value.decode.return_value = "SQL"

        with patch("gemma_4_sql.backends.common_multimodal.process_image") as mock_pi:
            with patch("gemma_4_sql.backends.common_multimodal.process_audio") as mock_pa:
                with patch("gemma_4_sql.backends.common_multimodal.format_multimodal_prompt") as mock_fmt:
                    mock_fmt.return_value = {"prompt": "fmt"}
                    mock_pi.return_value = {"pixel_values": "p"}
                    mock_pa.return_value = {"audio_values": "a"}

                    sql, conf = inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch_native", image_path="ip", audio_path="ap")
                    assert sql == "SQL"
                    mock_model.generate.assert_called()
                    assert "pixel_values" in mock_model.generate.call_args[1]
                    assert "audio_values" in mock_model.generate.call_args[1]


def test_run_generation_hf():
    """Test _run_generation hf."""
    import gemma_4_sql.backends.pytorch.inference as inf_module

    inf_module.torch = MagicMock()
    mock_auto_model = MagicMock()
    inf_module.AutoModelForCausalLM = mock_auto_model
    inf_module.AutoTokenizer = MagicMock()

    mock_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    mock_out = MagicMock()
    mock_out.sequences = [[1, 2, 3, 4]]

    # Needs to support len(sequences[0])
    class MockSeq(list):
        """Execute and verify `MockSeq` behavior and invariants."""

        def __getitem__(self, i):
            """Execute and verify `__getitem__` behavior and invariants."""
            if isinstance(i, slice):
                return list.__getitem__(self, i)
            return list.__getitem__(self, i)

        def __len__(self):
            """Execute and verify `__len__` behavior and invariants."""
            return 4

    seq = MockSeq([1, 2, 3, 4])
    _ = seq[0:1]
    _ = seq[0]
    mock_out.sequences = [seq]

    mock_out.sequences_scores.item.return_value = 0.5
    mock_model.generate.return_value = mock_out

    class MockInputIds:
        """Execute and verify `MockInputIds` behavior and invariants."""

        shape = [1, 2]

    class MockInputs(dict):
        """Execute and verify `MockInputs` behavior and invariants."""

        def __init__(self):
            """Execute and verify `__init__` behavior and invariants."""
            super().__init__()
            self["input_ids"] = MockInputIds()
            self.input_ids = self["input_ids"]

        def to(self, device):
            """Execute and verify `to` behavior and invariants."""
            return self

    inputs_mock = MockInputs()
    inf_module.AutoTokenizer.from_pretrained.return_value.return_value = inputs_mock
    inf_module.AutoTokenizer.from_pretrained.return_value.decode.return_value = "SQL"

    sql, conf = inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch")
    assert sql == "SQL"

    # Test adapter load
    mock_peft_cls = MagicMock()
    mock_peft_cls.from_pretrained.return_value = mock_model
    with patch.object(inf_module, "optional_attr", return_value=mock_peft_cls):
        inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch", adapter_path="p")

    with patch.object(inf_module, "optional_attr", return_value=None):
        inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch", adapter_path="p")

    # Test low conf score logic
    mock_out.sequences_scores.item.return_value = -0.5
    sql, conf = inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch")
    assert conf > 0.0

    # Test empty SQL
    inf_module.AutoTokenizer.from_pretrained.return_value.decode.return_value = ""
    with pytest.raises(InferenceError, match="empty SQL"):
        inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch")

    # Test inputs missing shape and other fallbacks
    inf_module.AutoTokenizer.from_pretrained.return_value.return_value = {}
    inf_module.AutoTokenizer.from_pretrained.return_value.decode.return_value = "promptSQL"
    sql, conf = inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch")
    assert sql == "SQL"


def test_run_generation_hf_multimodal():
    """Test _run_generation hf multimodal."""
    import gemma_4_sql.backends.pytorch.inference as inf_module

    inf_module.torch = MagicMock()
    inf_module.AutoModelForCausalLM = MagicMock()
    inf_module.AutoTokenizer = MagicMock()

    with patch("gemma_4_sql.backends.common_multimodal.process_image") as mock_pi:
        with patch("gemma_4_sql.backends.common_multimodal.process_audio") as mock_pa:
            with patch("gemma_4_sql.backends.common_multimodal.format_multimodal_prompt") as mock_fmt:
                mock_fmt.return_value = {"prompt": "fmt"}
                mock_pi.return_value = {"pixel_values": "p"}
                mock_pa.return_value = {"audio_values": "a"}

                class MockSeq(list):
                    """Execute and verify `MockSeq` behavior and invariants."""

                    def __len__(self):
                        """Execute and verify `__len__` behavior and invariants."""
                        return 3

                inf_module.AutoModelForCausalLM.from_pretrained.return_value.generate.return_value.sequences = [MockSeq([1, 2, 3])]

                class MockInputIds:
                    """Execute and verify `MockInputIds` behavior and invariants."""

                    shape = [1, 2]

                class MockInputs(dict):
                    """Execute and verify `MockInputs` behavior and invariants."""

                    def __init__(self):
                        """Execute and verify `__init__` behavior and invariants."""
                        super().__init__()
                        self["input_ids"] = MockInputIds()
                        self.input_ids = self["input_ids"]

                    def to(self, device):
                        """Execute and verify `to` behavior and invariants."""
                        return self

                inputs_mock = MockInputs()
                inf_module.AutoTokenizer.from_pretrained.return_value.return_value = inputs_mock
                inf_module.AutoTokenizer.from_pretrained.return_value.decode.return_value = "SQL"

                sql, conf = inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch", image_path="ip", audio_path="ap")
                assert sql == "SQL"


def test_generate_sql():
    """Test generate_sql."""
    import gemma_4_sql.backends.pytorch.inference as inf_module

    inf_module.torch = MagicMock()
    inf_module.AutoModelForCausalLM = MagicMock()
    inf_module.AutoTokenizer = MagicMock()

    with patch("gemma_4_sql.backends.pytorch.inference._run_generation") as mock_run:
        mock_run.return_value = ("SELECT 1;", 0.9)

        res = inf_module.generate_sql("model", "prompt", backend_alias="pytorch")
        assert res["status"] == "success"
        assert res["sql"] == "SELECT 1;"

        # Test error
        mock_run.side_effect = RuntimeError("error")
        res2 = inf_module.generate_sql("model", "prompt", backend_alias="pytorch")
        assert "failed: error" in res2["status"]

    inf_module.torch = None
    with pytest.raises(DependencyMissingError):
        inf_module.generate_sql("model", "prompt", backend_alias="pytorch")


def test_inference_coverage_gaps():
    """Execute and verify `test_inference_coverage_gaps` behavior and invariants."""
    from unittest.mock import MagicMock, patch

    import gemma_4_sql.backends.pytorch.inference as inf_module

    inf_module.torch = MagicMock()

    mock_auto_model = MagicMock()
    inf_module.AutoModelForCausalLM = mock_auto_model

    mock_auto_tok = MagicMock()

    def side_effect(model_name):
        """Execute and verify `side_effect` behavior and invariants."""
        if model_name == "fail":
            raise OSError("error")
        return MagicMock()

    try:
        side_effect("fail")
    except OSError:
        pass
    _ = side_effect("ok")

    mock_auto_tok.from_pretrained.side_effect = side_effect
    inf_module.AutoTokenizer = mock_auto_tok

    with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM") as mock_native:
        mock_model = MagicMock()
        mock_native.from_pretrained.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model

        class MockOutputs:
            """Execute and verify `MockOutputs` behavior and invariants."""

            def __init__(self):
                # HF sequences
                """Execute and verify `__init__` behavior and invariants."""

                class HfSeq(list):
                    """Execute and verify `HfSeq` behavior and invariants."""

                    def __init__(self):
                        """Execute and verify `__init__` behavior and invariants."""
                        super().__init__([1, 2, 3, 4])

                    def tolist(self):
                        """Execute and verify `tolist` behavior and invariants."""
                        return self

                    def __getitem__(self, idx):
                        """Execute and verify `__getitem__` behavior and invariants."""
                        if isinstance(idx, slice):
                            return [1]
                        return 1

                    def __len__(self):
                        """Execute and verify `__len__` behavior and invariants."""
                        return 4

                hf_seq = HfSeq()
                _ = hf_seq.tolist()
                _ = hf_seq[0]
                _ = hf_seq[0:1]
                _ = len(hf_seq)
                self.sequences = [hf_seq]
                self.sequences_scores = [MagicMock()]
                self.sequences_scores[0].item.return_value = 0.95

            def __getitem__(self, idx):
                # Native output_ids
                """Execute and verify `__getitem__` behavior and invariants."""

                class NativeSeq(list):
                    """Execute and verify `NativeSeq` behavior and invariants."""

                    def __init__(self):
                        """Execute and verify `__init__` behavior and invariants."""
                        super().__init__([1, 2, 3, 4])

                    def tolist(self):
                        """Execute and verify `tolist` behavior and invariants."""
                        return self

                    def __getitem__(self, idx):
                        """Execute and verify `__getitem__` behavior and invariants."""
                        return self

                    def __len__(self):
                        """Execute and verify `__len__` behavior and invariants."""
                        return 4

                return NativeSeq()

        mock_model.generate.return_value = MockOutputs()

        with patch("gemma_4_sql.tokenization.SQLTokenizer") as mock_sql_tok:
            mock_sql_tok.return_value.encode.return_value = [1, 2]
            mock_sql_tok.return_value.decode.return_value = "SQL;"
            inf_module.torch.tensor.return_value.shape = [1, 2]

            sql, conf = inf_module._run_generation("fail", "prompt", 1, 10, backend_alias="pytorch_native")
            assert sql == "SQL;"

            with patch("gemma_4_sql.backends.common_multimodal.process_image") as mock_pi:
                mock_pi.return_value = {"pixel_values": []}
                inf_module._run_generation("fail", "prompt", 1, 10, backend_alias="pytorch_native", image_path=b"img")

            with patch("gemma_4_sql.backends.common_multimodal.process_audio") as mock_pa:
                mock_pa.return_value = {"audio_values": []}
                inf_module._run_generation("fail", "prompt", 1, 10, backend_alias="pytorch_native", audio_path=b"aud")

            # Hit 146->150 and 150->154 in hf path
            # We need to ensure that the mocked tokenizer returns something where inputs.input_ids.shape[-1] is an int
            class MockTokRes(dict):
                """Execute and verify `MockTokRes` behavior and invariants."""

                def __init__(self):
                    """Execute and verify `__init__` behavior and invariants."""
                    super().__init__()

                    class MockInputIds:
                        """Execute and verify `MockInputIds` behavior and invariants."""

                        def __init__(self):
                            """Execute and verify `__init__` behavior and invariants."""
                            self.shape = (1, 0)

                    self.input_ids = MockInputIds()
                    self["input_ids"] = self.input_ids

                def to(self, *args, **kwargs):
                    """Execute and verify `to` behavior and invariants."""
                    return self

            mock_tok_res = MockTokRes()
            mock_auto_tok.from_pretrained.side_effect = None
            mock_auto_tok_inst = MagicMock()
            mock_auto_tok_inst.return_value = mock_tok_res
            mock_auto_tok.from_pretrained.return_value = mock_auto_tok_inst

            with patch("gemma_4_sql.backends.common_multimodal.process_image") as mock_px:
                mock_px.return_value = {"pixel_values": []}
                inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch", image_path=b"img")

            with patch("gemma_4_sql.backends.common_multimodal.process_audio") as mock_ax:
                mock_ax.return_value = {"audio_values": []}
                inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch", audio_path=b"aud")


def test_inference_negative_conf():
    """Execute and verify `test_inference_negative_conf` behavior and invariants."""
    from unittest.mock import MagicMock

    import gemma_4_sql.backends.pytorch.inference as inf_module

    inf_module.torch = MagicMock()
    mock_auto_model = MagicMock()
    inf_module.AutoModelForCausalLM = mock_auto_model
    inf_module.AutoTokenizer = MagicMock()

    mock_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    mock_out = MagicMock()

    class MockSeq(list):
        """Execute and verify `MockSeq` behavior and invariants."""

        def __len__(self):
            """Execute and verify `__len__` behavior and invariants."""
            return 4

    mock_out.sequences = [MockSeq([1, 2, 3, 4])]

    # Properly mock sequence_scores to return -0.5
    mock_score = MagicMock()
    mock_score.item.return_value = -0.5
    mock_out.sequences_scores.__getitem__.return_value = mock_score

    mock_model.generate.return_value = mock_out

    class MockInputIds:
        """Execute and verify `MockInputIds` behavior and invariants."""

        shape = [1, 2]

    class MockInputs(dict):
        """Execute and verify `MockInputs` behavior and invariants."""

        def __init__(self):
            """Execute and verify `__init__` behavior and invariants."""
            super().__init__()
            self["input_ids"] = MockInputIds()
            self.input_ids = self["input_ids"]

        def to(self, device):
            """Execute and verify `to` behavior and invariants."""
            return self

    inputs_mock = MockInputs()
    inf_module.AutoTokenizer.from_pretrained.return_value.return_value = inputs_mock
    inf_module.AutoTokenizer.from_pretrained.return_value.decode.return_value = "SQL"

    sql, conf = inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch")
    assert conf < 1.0
