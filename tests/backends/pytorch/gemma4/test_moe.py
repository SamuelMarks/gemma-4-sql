"""Tests for Mixture of Experts layer."""

import torch

from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
from gemma_4_sql.backends.pytorch.gemma4.moe import Gemma4MoE, calculate_load_balancing_loss


def test_moe_layer():
    """Test Gemma4MoE layer."""
    config = Gemma4Config(hidden_size=256, intermediate_size=512, num_experts=4, num_experts_per_tok=2, router_jitter_noise=0.1)
    moe = Gemma4MoE(config)
    x = torch.randn(2, 10, 256)

    # Test training mode (jitter enabled)
    moe.train()
    out, router_logits = moe(x)
    assert out.shape == (2, 10, 256)
    assert router_logits.shape == (20, 4)

    # Test eval mode (jitter disabled)
    moe.eval()
    out_eval, _ = moe(x)
    assert out_eval.shape == (2, 10, 256)

    # Test loss computation
    loss = calculate_load_balancing_loss(router_logits, num_experts=4, top_k=2)
    assert loss.item() >= 0.0

    # Test empty routing (force router to only pick expert 0)
    moe.router.gate.weight = torch.nn.Parameter(torch.zeros_like(moe.router.gate.weight))
    moe.router.gate.weight.data[0, :] = 100.0  # expert 0 gets all
    out_empty, _ = moe(x)
    assert out_empty.shape == (2, 10, 256)
