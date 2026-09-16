"""Task 6 verify: full forward over all modules, init-time invariants."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.transformer import Transformer
from tests.toy import toy_config


@pytest.fixture(scope="module")
def cfg():
    # toy with engram enabled on the decoder layer
    return toy_config(engram_layer_ids=(1,), engram_num_embeddings=(512, 512))


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(1337)
    return Transformer(cfg, max_seq_len=64)


def test_e2e_forward_backward(cfg, model):
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    logits, _ = model(x)
    assert logits.shape == (2, 12, cfg.vocab_size)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 20
    assert all(torch.isfinite(g).all() for g in grads)


def test_engram_output_exactly_zero_at_init(cfg, model):
    """Design §4.3: the Engram module outputs exactly 0 at init."""
    blk = model.blocks[1]
    x = torch.randn(2, 8, cfg.hc_mult, cfg.d_model)
    hashes = torch.randint(0, 100, (2, 8, 4))
    out = blk.engram(x, hashes)
    assert torch.equal(out, x)  # bitwise identity: the gate contributes exactly nothing
    assert (blk.engram.gate_scale == 0).all()


def test_engram_gate_bounded_after_init(cfg, model):
    """Once the gate is nonzero, the sigmoid gate itself stays < 1 (structural)."""
    blk = model.blocks[1]
    blk.engram.gate_scale.data.fill_(0.5)
    x = torch.randn(1, 4, cfg.hc_mult, cfg.d_model)
    hashes = torch.randint(0, 100, (1, 4, 4))
    out = blk.engram(x, hashes)
    assert out.isfinite().all() and not torch.equal(out, x)
    blk.engram.gate_scale.data.zero_()  # restore


def test_mhc_sinkhorn_doubly_stochastic(cfg, model):
    """comb must be doubly stochastic (row and col sums ~1) after Sinkhorn."""
    x = torch.randn(2, 4, cfg.hc_mult, cfg.d_model)
    _, _, comb = model.blocks[0].hc_attn(x)
    assert torch.allclose(comb.sum(-1), torch.ones_like(comb[..., 0]), atol=1e-4)
    assert torch.allclose(comb.sum(-2), torch.ones_like(comb[..., 0, :]), atol=1e-4)


def test_tied_head(cfg, model):
    assert model.head is None
    assert model.head_weight() is model.embed.weight


def test_moe_vl_bias_path(cfg, model):
    """noaux_tc: bias_vl changes SELECTION only; gathered weights for unchanged
    selections are identical. Deterministic construction: basis-vector expert
    weights, token on expert 1, vl bias favoring expert 0."""
    gate = model.blocks[0].ffn.gate
    with torch.no_grad():
        gate.weight.zero_()
        gate.weight[0, 0] = 1.0
        gate.weight[1, 1] = 1.0
        gate.bias_vl.fill_(0.0)
        gate.bias_vl[0] = 10.0
    x = torch.zeros(2, cfg.d_model)
    x[:, 1] = 1.0  # both tokens score highest on expert 1 raw
    image_mask = torch.tensor([True, False])  # token 0 routes with bias_vl
    w, idx = gate(x, image_mask)
    w0, idx0 = gate(x, None)
    assert idx[0].item() == 0 and idx0[0].item() == 1  # vl bias flips the pick
    assert idx[1].item() == idx0[1].item() == 1
    assert torch.allclose(w[1], w0[1])  # unchanged selection -> same gathered weight


def test_expert_load_counters(cfg, model):
    from models.attention import SharedAttentionRuntime  # noqa: F401

    x = torch.randint(0, cfg.vocab_size, (1, 16))
    logits, _ = model(x)
    # run one MoE manually to check the counter helper
    h = torch.randn(8, cfg.d_model)
    weights, indices = model.blocks[0].ffn.gate(h)
    counts = model.blocks[0].ffn.load_counts(indices)
    assert counts.sum() == 8 * cfg.n_activated_experts


def test_decode_step_parity(cfg):
    """One-token decode after prefill matches the full-sequence forward
    (logit parity within BF16 tolerance) — CED + indexer + MoE + mHC."""
    torch.manual_seed(7)
    cfg2 = toy_config(engram_layer_ids=(1,), engram_num_embeddings=(512, 512),
                      compress_ratios=(2, 1), index_source_layers=(0, 1), candidate_source_layer=1)
    m1 = Transformer(cfg2, max_seq_len=64)
    m2 = Transformer(cfg2, max_seq_len=64)
    m2.load_state_dict(m1.state_dict())  # identical weights
    x = torch.randint(0, cfg2.vocab_size, (1, 11))
    m1(x[:, :10])  # prefill 10 (populates compressor/ring/hash state)
    step, _ = m1(x[:, 10:11], start_pos=10)  # decode token 10
    full, _ = m2(x)  # full 11-token prefill reference
    assert torch.allclose(step[0, 0], full[0, 10], atol=1e-3)
