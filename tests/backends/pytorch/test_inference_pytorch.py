"""Tests for PyTorch inference."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, InferenceError


def test_pytorch_inference_imports():
    """Test pytorch inference imports fallback."""
    import importlib

    with patch.dict(sys.modules, {"torch": None, "transformers": None}):
        import gemma_4_sql.backends.pytorch.inference as inf_module

        importlib.reload(inf_module)
        assert inf_module.torch is None
        assert inf_module.AutoModelForCausalLM is None
        assert inf_module.AutoTokenizer is None
    importlib.reload(inf_module)


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
                """Docstring for MockGenTokens."""

                def tolist(self):
                    """Docstring for tolist."""
                    return [1, 2]

                def __len__(self):
                    """Docstring for __len__."""
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
        """Docstring for MockSeq."""

        def __getitem__(self, i):
            """Docstring for __getitem__."""
            if isinstance(i, slice):
                return list.__getitem__(self, i)
            return list.__getitem__(self, i)

        def __len__(self):
            """Docstring for __len__."""
            return 4

    seq = MockSeq([1, 2, 3, 4])
    mock_out.sequences = [seq]

    mock_out.sequences_scores.item.return_value = 0.5
    mock_model.generate.return_value = mock_out

    class MockInputIds:
        """Docstring for MockInputIds."""

        shape = [1, 2]

    class MockInputs(dict):
        """Docstring for MockInputs."""

        def __init__(self):
            """Docstring for __init__."""
            super().__init__()
            self["input_ids"] = MockInputIds()
            self.input_ids = self["input_ids"]

        def to(self, device):
            """Docstring for to."""
            return self

    inputs_mock = MockInputs()
    inf_module.AutoTokenizer.from_pretrained.return_value.return_value = inputs_mock
    inf_module.AutoTokenizer.from_pretrained.return_value.decode.return_value = "SQL"

    sql, conf = inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch")
    assert sql == "SQL"

    # Test adapter load
    with patch("builtins.__import__") as mock_import:
        mock_peft = MagicMock()
        mock_peft.PeftModel.from_pretrained.return_value = mock_model
        mock_import.return_value = mock_peft

        inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch", adapter_path="p")

        # Test PEFT import failure
        mock_import.side_effect = ImportError("error")
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
                    """Docstring for MockSeq."""

                    def __len__(self):
                        """Docstring for __len__."""
                        return 3

                inf_module.AutoModelForCausalLM.from_pretrained.return_value.generate.return_value.sequences = [MockSeq([1, 2, 3])]

                class MockInputIds:
                    """Docstring for MockInputIds."""

                    shape = [1, 2]

                class MockInputs(dict):
                    """Docstring for MockInputs."""

                    def __init__(self):
                        """Docstring for __init__."""
                        super().__init__()
                        self["input_ids"] = MockInputIds()
                        self.input_ids = self["input_ids"]

                    def to(self, device):
                        """Docstring for to."""
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
    """Docstring for test_inference_coverage_gaps."""
    from unittest.mock import MagicMock, patch

    import gemma_4_sql.backends.pytorch.inference as inf_module

    inf_module.torch = MagicMock()

    mock_auto_model = MagicMock()
    inf_module.AutoModelForCausalLM = mock_auto_model

    mock_auto_tok = MagicMock()

    def side_effect(model_name):
        """Docstring for side_effect."""
        if model_name == "fail":
            raise OSError("error")
        return MagicMock()

    mock_auto_tok.from_pretrained.side_effect = side_effect
    inf_module.AutoTokenizer = mock_auto_tok

    with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM") as mock_native:
        mock_model = MagicMock()
        mock_native.from_pretrained.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model

        class MockOutputs:
            """Docstring for MockOutputs."""

            def __init__(self):
                # HF sequences
                """Docstring for __init__."""

                class HfSeq(list):
                    """Docstring for HfSeq."""

                    def __init__(self):
                        """Docstring for __init__."""
                        super().__init__([1, 2, 3, 4])

                    def tolist(self):
                        """Docstring for tolist."""
                        return self

                    def __getitem__(self, idx):
                        """Docstring for __getitem__."""
                        if isinstance(idx, slice):
                            return [1]
                        return 1

                    def __len__(self):
                        """Docstring for __len__."""
                        return 4

                self.sequences = [HfSeq()]
                self.sequences_scores = [MagicMock()]
                self.sequences_scores[0].item.return_value = 0.95

            def __getitem__(self, idx):
                # Native output_ids
                """Docstring for __getitem__."""

                class NativeSeq(list):
                    """Docstring for NativeSeq."""

                    def __init__(self):
                        """Docstring for __init__."""
                        super().__init__([1, 2, 3, 4])

                    def tolist(self):
                        """Docstring for tolist."""
                        return self

                    def __getitem__(self, idx):
                        """Docstring for __getitem__."""
                        return self

                    def __len__(self):
                        """Docstring for __len__."""
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
                """Docstring for MockTokRes."""

                def __init__(self):
                    """Docstring for __init__."""
                    super().__init__()

                    class MockInputIds:
                        """Docstring for MockInputIds."""

                        def __init__(self):
                            """Docstring for __init__."""
                            self.shape = (1, 0)

                    self.input_ids = MockInputIds()
                    self["input_ids"] = self.input_ids

                def to(self, *args, **kwargs):
                    """Docstring for to."""
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
    """Docstring for test_inference_negative_conf."""
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
        """Docstring for MockSeq."""

        def __len__(self):
            """Docstring for __len__."""
            return 4

    mock_out.sequences = [MockSeq([1, 2, 3, 4])]

    # Properly mock sequence_scores to return -0.5
    mock_score = MagicMock()
    mock_score.item.return_value = -0.5
    mock_out.sequences_scores.__getitem__.return_value = mock_score

    mock_model.generate.return_value = mock_out

    class MockInputIds:
        """Docstring for MockInputIds."""

        shape = [1, 2]

    class MockInputs(dict):
        """Docstring for MockInputs."""

        def __init__(self):
            """Docstring for __init__."""
            super().__init__()
            self["input_ids"] = MockInputIds()
            self.input_ids = self["input_ids"]

        def to(self, device):
            """Docstring for to."""
            return self

    inputs_mock = MockInputs()
    inf_module.AutoTokenizer.from_pretrained.return_value.return_value = inputs_mock
    inf_module.AutoTokenizer.from_pretrained.return_value.decode.return_value = "SQL"

    sql, conf = inf_module._run_generation("model", "prompt", 1, 10, backend_alias="pytorch")
    assert conf < 1.0
