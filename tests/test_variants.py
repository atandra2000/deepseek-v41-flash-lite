"""Gate-variant construction invariants (Task 11 closeout)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.ced import CedProjection, Compressor
from models.config import config_sha256
from models.engram import Engram
from models.indexer import Indexer
from models.mhc import HCMixes, PlainResidual
from models.moe import Expert, Gate
from models.transformer import Transformer
from models.variants import FULL, NAMED, STAGE_CONTROL, STAGE_VARIANT, VariantSpec, derive_config
from tests.toy import toy_config
from training.pretrain import seed_everything


def gate_toy_config(**over):
    """3-layer toy with every mechanism on: enc layer 0 (m=2, kv+index
    source), dec layer 1 (m=1, index source / pool builder), dec layer 2
    (m=1, Reuse), Engram on layer 2. The C0-C6 ladder at test dims."""
    return toy_config(
        n_layers=3,
        n_encoder_layers=1,
        compress_ratios=(2, 1, 1),
        kv_source_layers=(0,),
        index_source_layers=(0, 1),
        candidate_source_layer=1,
        candidate_topk_blocks=4,
        candidate_block_size=2,
        engram_layer_ids=(2,),
        engram_num_embeddings=(256, 256),
        dspark_target_layer_ids=(1, 2),
        **over,
    )


def build(cfg, spec, seed=0):
    seed_everything(0)
    torch.manual_seed(seed)
    return Transformer(derive_config(cfg, spec), max_seq_len=64, variant=spec)


def has_any(model, cls):
    return any(isinstance(m, cls) for m in model.modules())


def test_dense_c0_strips_every_gate_mechanism():
    cfg = gate_toy_config()
    model = build(cfg, NAMED["dense"])
    for cls in (Gate, Indexer, Compressor, CedProjection, Engram, HCMixes):
        assert not has_any(model, cls), cls.__name__
    assert has_any(model, PlainResidual)
    # dense FFN carries the routed top-k + shared experts' combined active width
    for blk in model.blocks:
        assert isinstance(blk.ffn, Expert)
        assert blk.ffn.w1.out_features == (cfg.n_activated_experts + cfg.n_shared_experts) * cfg.moe_inter_dim
    ids = torch.arange(24).unsqueeze(0) % cfg.vocab_size
    model(ids)[0].sum().backward()


def test_full_variant_matches_plain_production_construction():
    cfg = gate_toy_config()
    plain = build(cfg, None)
    full = build(cfg, FULL)
    assert list(plain.state_dict()) == list(full.state_dict())
    ids = torch.arange(24).unsqueeze(0) % cfg.vocab_size
    a, _ = plain(ids)
    b, _ = full(ids)
    assert torch.allclose(a, b, atol=1e-6)
    # the production config file identity is untouched by any of this
    assert len(config_sha256()) == 64


def test_variant_param_counts_are_cumulative():
    cfg = gate_toy_config()
    counts = {name: sum(p.numel() for p in build(cfg, spec).parameters()) for name, spec in NAMED.items()}
    order = [STAGE_VARIANT[s] for s in ("C0", "C1", "C2", "C3", "C4", "C5", "C6")]
    assert order == list(NAMED)
    for prev, cur in zip(order, order[1:]):
        assert counts[cur] >= counts[prev]
    for added in ("moe", "ced", "csa2", "mhc", "engram"):  # swa adds no parameters
        assert counts[added] > counts[order[order.index(added) - 1]]


def test_swa_off_is_dense_attention_and_swa_on_is_windowed():
    cfg = gate_toy_config()
    dense_cfg = derive_config(cfg, NAMED["moe"])  # moe stage: full attention
    swa_cfg = derive_config(cfg, NAMED["swa"])
    assert dense_cfg.window_size == cfg.context_train_stage2
    assert swa_cfg.window_size == cfg.window_size
    torch.manual_seed(7)
    full_attn = Transformer(dense_cfg, max_seq_len=64, variant=NAMED["moe"])
    torch.manual_seed(7)
    windowed = Transformer(swa_cfg, max_seq_len=64, variant=NAMED["swa"])
    assert torch.equal(full_attn.blocks[0].attn.wq_a.weight, windowed.blocks[0].attn.wq_a.weight)
    ids = torch.arange(24).unsqueeze(0) % cfg.vocab_size
    perturbed = ids.clone()
    perturbed[0, 0] = (perturbed[0, 0] + 1) % cfg.vocab_size
    base, _ = full_attn(ids)
    changed, _ = full_attn(perturbed)
    assert not torch.allclose(base[0, -1], changed[0, -1], atol=1e-6)  # dense sees token 0
    base_w, _ = windowed(ids)
    changed_w, _ = windowed(perturbed)
    assert torch.allclose(base_w[0, -1], changed_w[0, -1], atol=1e-5)  # window 8 cannot


def test_csa2_off_keeps_compressed_global_pathway_without_selection():
    cfg = gate_toy_config()
    dense_global = build(cfg, NAMED["ced"])  # ced on, csa2 off
    assert not has_any(dense_global, Indexer)
    assert has_any(dense_global, Compressor) and has_any(dense_global, CedProjection)
    ids = torch.arange(24).unsqueeze(0) % cfg.vocab_size
    base, _ = dense_global(ids)  # prefill-only reachable-idxs fallback
    perturbed = ids.clone()
    perturbed[0, 0] = (perturbed[0, 0] + 1) % cfg.vocab_size
    changed, _ = dense_global(perturbed)
    # token 0 reaches the far end through the (unselected) compressed stream:
    # the SWA window alone (8) could not carry it 23 positions.
    assert not torch.allclose(base[0, -1], changed[0, -1], atol=1e-6)


def test_mhc_frozen_identity_has_no_trainable_hc_params():
    from dataclasses import replace

    cfg = gate_toy_config()
    frozen = replace(NAMED["mhc"], name="mhc-identity", mhc_frozen=True)
    model = build(cfg, frozen)
    assert has_any(model, HCMixes)
    hc_params = [p for m in model.modules() if isinstance(m, HCMixes) for p in m.parameters()]
    assert hc_params and not any(p.requires_grad for p in hc_params)
    ids = torch.arange(24).unsqueeze(0) % cfg.vocab_size
    model(ids)[0].sum().backward()
    assert all(p.grad is None for p in hc_params)


def test_derive_config_toggles():
    from dataclasses import fields as dc_fields

    cfg = gate_toy_config()
    d = derive_config(cfg, NAMED["dense"])
    assert set(d.compress_ratios) == {0}
    assert d.kv_source_layers == () and d.index_source_layers == () and d.candidate_source_layer == -1
    assert d.window_size == cfg.context_train_stage2 and d.hc_mult == 1
    assert d.engram_layer_ids == () and d.vision_n_layers == 0 and d.dspark_block_size == 0
    full = derive_config(cfg, FULL)
    assert full.compress_ratios == cfg.compress_ratios and full.engram_layer_ids == cfg.engram_layer_ids
    assert full.kv_source_layers == cfg.kv_source_layers and full.hc_mult == cfg.hc_mult
    # derived configs keep the exact LiteConfig field set
    assert [f.name for f in dc_fields(cfg)] == [f.name for f in dc_fields(full)]


def test_stage_maps_are_consistent():
    stages = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    assert [STAGE_VARIANT[s] for s in stages] == list(NAMED)
    assert STAGE_CONTROL["C0"] is None
    for prev, cur in zip(stages, stages[1:]):
        assert STAGE_CONTROL[cur] == STAGE_VARIANT[prev]

