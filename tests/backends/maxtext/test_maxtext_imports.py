"""Module docstring."""

import importlib
import sys
from unittest.mock import MagicMock


def test_maxtext_imports_success():
    """Docstring for test_maxtext_imports_success."""
    sys.modules["jax"] = MagicMock()
    sys.modules["jax.numpy"] = MagicMock()
    sys.modules["maxtext"] = MagicMock()
    sys.modules["maxtext.models"] = MagicMock()
    sys.modules["maxtext.models.gemma4"] = MagicMock()
    sys.modules["maxtext.quantization"] = MagicMock()
    sys.modules["maxtext.peft"] = MagicMock()
    sys.modules["optax"] = MagicMock()
    sys.modules["orbax"] = MagicMock()
    sys.modules["orbax.checkpoint"] = MagicMock()

    import gemma_4_sql.backends.maxtext as m_init

    importlib.reload(m_init)

    import gemma_4_sql.backends.maxtext.benchmark as m_bench

    importlib.reload(m_bench)

    import gemma_4_sql.backends.maxtext.dpo as m_dpo

    importlib.reload(m_dpo)

    import gemma_4_sql.backends.maxtext.export as m_export

    importlib.reload(m_export)

    import gemma_4_sql.backends.maxtext.inference as m_inf

    importlib.reload(m_inf)

    import gemma_4_sql.backends.maxtext.peft as m_peft

    importlib.reload(m_peft)

    import gemma_4_sql.backends.maxtext.quantize as m_quant

    importlib.reload(m_quant)


def test_maxtext_imports_missing_force():
    """Docstring for test_maxtext_imports_missing_force."""
    if "maxtext" in sys.modules:
        del sys.modules["maxtext"]
    if "jax" in sys.modules:
        del sys.modules["jax"]
    import gemma_4_sql.backends.maxtext as m_init

    importlib.reload(m_init)

    import gemma_4_sql.backends.maxtext.benchmark as m_bench

    importlib.reload(m_bench)

    import gemma_4_sql.backends.maxtext.dpo as m_dpo

    importlib.reload(m_dpo)

    import gemma_4_sql.backends.maxtext.export as m_export

    importlib.reload(m_export)

    import gemma_4_sql.backends.maxtext.inference as m_inf

    importlib.reload(m_inf)

    import gemma_4_sql.backends.maxtext.peft as m_peft

    importlib.reload(m_peft)

    import gemma_4_sql.backends.maxtext.quantize as m_quant

    importlib.reload(m_quant)
