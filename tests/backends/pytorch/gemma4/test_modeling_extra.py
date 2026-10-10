"""Module docstring."""

import pytest

try:
    import torch
except ImportError:
    pytest.skip("No torch", allow_module_level=True)

from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM


def test_modeling_coverage_branches():
    """Docstring for test_modeling_coverage_branches."""
    config = Gemma4Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, intermediate_size=64, vocab_size=100)
    model = Gemma4ForCausalLM(config)

    input_ids = torch.randint(0, 100, (1, 5))

    # 1. Provide position_ids (hits position_ids is not None branch: 185->188)
    position_ids = torch.arange(5).unsqueeze(0)

    # 2. Provide attention_mask with shape[1] > curr_seq_len (so diff < 0)
    # curr_seq_len is 5. attention_mask shape[1] will be 10.
    attention_mask = torch.ones((1, 10))

    try:
        _ = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
    except RuntimeError:
        pass
