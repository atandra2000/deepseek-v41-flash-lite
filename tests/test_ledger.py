"""Task 2 verify: ledger numbers, config maps, and structural invariants."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.config import LiteConfig, load_config
from models.csa2 import layer_mode, mode_table
from models.transformer import Transformer


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(1337)
    return Transformer(cfg)


def test_config_loads_and_sha_embeds(cfg):
    assert len(cfg.sha256) == 64
    assert cfg.name == "lite-v3"


def test_ledger_bands(cfg, model):
    total = sum(p.numel() for p in model.parameters())
    assert abs(total - 1.178268637e9) < 1e6  # pinned exact figure; re-run ledger.py on drift
    assert (1 - 0.05) * 1.16e9 <= total <= (1 + 0.05) * 1.16e9

    # active (text path, excl. embedding/vision/dspark/engram-tables) == re-derived contract figure
    active = 0
    for blk in model.blocks:
        for m in (blk.attn, blk.compressor, blk.ced_projection, blk.indexer, blk.ffn.gate,
                  blk.hc_attn, blk.hc_ffn, blk.attn_norm, blk.ffn_norm):
            if m is not None:
                active += sum(p.numel() for p in m.parameters())
        active += sum(p.numel() for p in blk.ffn.shared_experts.parameters())
        active += sum(p.numel() for p in blk.ffn.experts.parameters()) * cfg.n_activated_experts / cfg.n_routed_experts
    active += sum(p.numel() for p in model.norm.parameters())
    assert abs(active - 218_528_976) < 1  # re-derived figure (incl. decoder CED projections)
    assert (1 - 0.05) * 218.5e6 <= active <= (1 + 0.05) * 218.5e6


def test_producer_and_mode_maps(cfg):
    from models.ced import ProducerMap

    pm = ProducerMap(cfg)
    assert pm.producer_of(0) is None and pm.producer_of(1) is None
    assert pm.producer_of(2) == 2 and pm.producer_of(5) == 2  # 2-7 read source 2
    assert pm.producer_of(8) == 8 and pm.producer_of(23) == 8  # 8-23 read source 8
    modes = {r["layer"]: r["mode"] for r in mode_table(cfg)}
    assert modes[2] == modes[8] == "reindex"  # encoder index sources, no pool yet
    assert modes[12] == "full"  # candidate source
    assert modes[16] == modes[20] == "reindex"  # pool users
    assert modes[13] == modes[21] == "reuse"
    assert all(modes[l] == "reuse" for l in range(2, 24) if l not in (2, 8, 12, 16, 20))


def test_layer_mode_contract(cfg):
    # upstream head pattern: backbone layers 0-1 are SWA-only; every other
    # backbone layer compresses (m=2 or m=1); DSpark layers are swa.
    modes = {r["layer"]: r["mode"] for r in mode_table(cfg)}
    assert modes[0] == modes[1] == "swa"
    for l in range(2, cfg.n_layers):
        assert modes[l] != "swa", f"backbone layer {l} must compress"
    for l in range(cfg.n_layers, len(cfg.compress_ratios)):
        assert layer_mode(cfg, l) == "swa"


def test_pool_capacity_at_stage2(cfg):
    # stage-2 16K: encoder compressed length = 8192 == pool capacity (exactly covers top-k)
    assert cfg.compressed_len(cfg.context_train_stage2, 2) == cfg.pool_capacity
    assert cfg.pool_capacity >= cfg.index_topk


def test_engram_zero_init_gate(model):
    for blk in model.blocks:
        if blk.engram is not None:
            assert (blk.engram.gate_scale == 0).all(), "Engram gate must be zero-init (design §4.3)"


def test_mhc_identity_init(model, cfg):
    hc = model.blocks[0].hc_attn
    x = torch.randn(1, 4, cfg.hc_mult, cfg.d_model)
    pre, post, comb = hc(x)
    assert torch.allclose(pre, torch.full_like(pre, 1.0 - hc.eps), atol=1e-4)  # one-hot copy 0
    assert (post == 1.0).all()
    eye = torch.eye(cfg.hc_mult).expand_as(comb)
    assert torch.allclose(comb, eye, atol=1e-3), "comb must be identity at init"


def test_router_bias_zero_init(model):
    for blk in model.blocks:
        assert (blk.ffn.gate.bias == 0).all()
        assert (blk.ffn.gate.bias_vl == 0).all()


def test_config_validation_rejects_bad_maps():
    base = load_config()
    bad = dict(base.__dict__)
    bad["kv_source_layers"] = (20, 8)  # unsorted
    with pytest.raises(AssertionError):
        LiteConfig(**bad).validate()
    bad = dict(base.__dict__)
    bad["candidate_source_layer"] = 5  # not an index source
    with pytest.raises(AssertionError):
        LiteConfig(**bad).validate()


def test_optimizer_footprint(cfg, model):
    total = sum(p.numel() for p in model.parameters())
    training_bytes = total * (4 + 4 * 2 + 2)
    assert training_bytes <= 16 * 2**30
