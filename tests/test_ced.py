"""Task 4 verify: CED shapes, causality, producer-map, decoder projection."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.ced import Compressor, CedProjection, ProducerMap, reachable_idxs, rope_latent
from models.transformer import Transformer
from tests.toy import ced_toy_config, toy_config


def test_compressor_shapes_and_remainder():
    cfg = toy_config()
    torch.manual_seed(0)
    comp = Compressor(cfg, 0)  # m=2
    x = torch.randn(2, 7, cfg.d_model)  # odd length: 3 groups + 1 leftover
    out = comp(x, start_pos=0)
    assert out.shape == (2, 3, cfg.head_dim)
    assert comp.kv_state is not None and comp.kv_state[0, 0].abs().sum() > 0  # leftover stashed
    # decode: fill the leftover slot -> group completes
    nxt = comp(torch.randn(2, 1, cfg.d_model), start_pos=7)
    assert nxt is not None and nxt.shape == (2, 1, cfg.head_dim)
    again = comp(torch.randn(2, 1, cfg.d_model), start_pos=8)
    assert again is None  # new group still filling


def test_compressor_gate_is_learned():
    """Flipping the gate weights must change the pooled latent (not a mean)."""
    cfg = toy_config()
    torch.manual_seed(0)
    comp = Compressor(cfg, 0)
    x = torch.randn(1, 8, cfg.d_model)
    y1 = comp(x, 0)
    comp.wgate.weight.data = -comp.wgate.weight.data
    y2 = comp(x, 0)
    assert not torch.allclose(y1, y2, atol=1e-6)


def test_rope_latent_positions():
    """Latent tail must be rotated at group-first positions (ratio 2: rows 0,2,4...)."""
    torch.manual_seed(0)
    freqs = torch.polar(torch.ones(8, 2), torch.outer(torch.arange(8.0), torch.ones(2)))
    lat = torch.randn(1, 3, 4)  # head_dim 4, rope tail 4 (freqs last dim 2)
    base = lat.clone()
    rope_latent(lat, freqs, start_pos=0, seqlen=6, ratio=2)
    # manual: rows use freqs[0], freqs[2], freqs[4]
    import torch as t

    def rot(v, f):
        vc = t.view_as_complex(v.unflatten(-1, (-1, 2)))
        return t.view_as_real(vc * f).flatten(-2)

    assert t.allclose(lat[0, 0], rot(base[0, 0], freqs[0]), atol=1e-5)
    assert t.allclose(lat[0, 1], rot(base[0, 1], freqs[2]), atol=1e-5)
    assert t.allclose(lat[0, 2], rot(base[0, 2], freqs[4]), atol=1e-5)


def test_reachable_idxs():
    idxs = reachable_idxs(4, 0, 6, offset=3)
    assert idxs.shape == (1, 6, 4)
    assert idxs[0, 0].tolist() == [3, -1, -1, -1]  # query 0 sees group 0 (offset)
    assert idxs[0, 3].tolist() == [3, 4, 5, 6]


def test_global_kv_path_table(cfg=None):
    cfg = cfg or ced_toy_config()
    paths = [cfg.global_kv_path(l) for l in range(cfg.n_layers)]
    assert paths[0] == "own" and paths[1] == "project"
    pm = ProducerMap(cfg)
    assert pm.producer_of(0) == 0 and pm.producer_of(1) == 0


def test_full_config_paths():
    from models.config import load_config

    cfg = load_config()
    paths = {l: cfg.global_kv_path(l) for l in range(cfg.n_layers)}
    assert paths[0] == paths[1] == "none"  # upstream head pattern
    assert paths[2] == "own" and paths[3] == "cache" and paths[11] == "cache"
    assert all(paths[l] == "project" for l in range(12, 24))
    pm = ProducerMap(cfg)
    assert pm.producer_of(11) == 8 and pm.producer_of(23) == 8
    assert pm.producer_of(7) == 2


@pytest.mark.parametrize("make_cfg", [toy_config, ced_toy_config], ids=["ced-off", "ced-on"])
def test_e2e_forward_backward(make_cfg):
    cfg = make_cfg()
    torch.manual_seed(1337)
    model = Transformer(cfg, max_seq_len=64)
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    logits, _ = model(x)
    assert logits.shape == (2, 12, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    g = model.embed.weight.grad
    assert g is not None and torch.isfinite(g).all()


def test_causality_ced():
    cfg = ced_toy_config()
    torch.manual_seed(5)
    model = Transformer(cfg, max_seq_len=64)
    x1 = torch.randint(0, cfg.vocab_size, (1, 10))
    x2 = x1.clone()
    x2[0, 8] = (x2[0, 8] + 7) % cfg.vocab_size
    l1, _ = model(x1)
    l2, _ = model(x2)
    assert torch.allclose(l1[0, :8], l2[0, :8], atol=1e-4)
    assert not torch.allclose(l1[0, 8:], l2[0, 8:], atol=1e-4)


def test_decoder_projects_from_producer_hidden():
    """The decoder's global KV comes from W^Z/W^KV on the producer's hidden
    state, never from the producer's cache (design §4.1)."""
    cfg = ced_toy_config()
    torch.manual_seed(1)
    model = Transformer(cfg, max_seq_len=64)
    seen = {}

    orig = model.blocks[1].ced_projection.forward

    def spy(hidden):
        seen["hidden"] = hidden.detach().clone()
        return orig(hidden)

    model.blocks[1].ced_projection.forward = spy
    x = torch.randint(0, cfg.vocab_size, (1, 6))
    model(x)
    assert seen["hidden"].shape == (1, 6, cfg.d_model)
    # and the shared cache was published by the encoder (m=2), not read by the decoder
    # (implicit: the projection spy was fed a [B,S,d] hidden, not a head_dim cache)


def test_producer_map_swap_changes_path(cfg=None):
    """Producer maps are data: swapping kv sources changes the path table."""
    cfg = ced_toy_config(kv_source_layers=(0,))
    assert cfg.global_kv_path(1) == "project"
    cfg2 = ced_toy_config(kv_source_layers=(0,), compress_ratios=(2, 2))
    # same-ratio consumer -> cache read instead of projection
    assert cfg2.global_kv_path(1) == "cache"
