"""Task 5 verify: CSA2 modes, indexer selection, candidate pool, oracle overlap."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.attention import SharedAttentionRuntime
from models.csa2 import layer_mode, select_candidate_blocks
from models.indexer import Indexer
from tests.oracle_upstream import oracle_indexer
from tests.toy import ced_toy_config, toy_config


def test_candidate_pool_semantics():
    """Block amax scoring, newest-block pinning, -inf drops (contract T3)."""
    # 8 positions, block 2, topk 2 blocks; scores favor blocks 0 and 2
    scores = torch.tensor([[[5.0, 0.0, 1.0, 9.0, 0.0, 0.0, 3.0, 0.0]]])  # [1,1,8]
    compress_lens = 8
    mask = select_candidate_blocks(scores, compress_lens, topk_blocks=2, block_size=2)
    # blocks: b0=5, b1=9, b2=3, b3=0 (positions 6,7) -> top2 = b1(9), b0(5)
    assert mask[0, 0].long().tolist() == [0, 0, 1, 1, 0, 0, 1, 1]  # top2 = pinned b3(+inf), b1(9)
    # newest block pinned: even if outscored, the block holding the query is kept
    scores2 = torch.tensor([[[9.0, 9.0, 9.0, 9.0, 0.0, 0.0, 1.0, 0.0]]])
    mask2 = select_candidate_blocks(scores2, compress_lens=8, topk_blocks=1, block_size=2)
    assert mask2[0, 0].long().tolist() == [0, 0, 0, 0, 0, 0, 1, 1]  # topk_blocks=1: only the pinned b3
    # -inf blocks (unreachable) are dropped, not selected
    scores3 = torch.full((1, 1, 8), -torch.inf)
    scores3[0, 0, :4] = 1.0
    mask3 = select_candidate_blocks(scores3, compress_lens=4, topk_blocks=2, block_size=2)
    assert mask3[0, 0].sum() == 4  # only the 4 reachable positions


def test_indexer_topk_matches_brute_force():
    """Selected indices must be the top-k scoring reachable positions."""
    cfg = ced_toy_config()
    torch.manual_seed(0)
    layer = Indexer(cfg, 1)
    B, S = 2, 10
    x = torch.randn(B, S, cfg.d_model)
    qr = torch.randn(B, S, cfg.q_lora_rank)
    latent = torch.randn(B, S // cfg.compress_ratios[1], cfg.head_dim)
    shared = SharedAttentionRuntime()
    idxs = layer(x, qr, latent, 0, offset=5, compress_len=S, shared=shared)

    # brute force: same scores the indexer computes (q RoPE'd, k from its cache)
    from models.layers import apply_rotary_emb

    k = layer.k_cache[:, :S]  # cache is full-length (Task 7); brute force uses the active region
    q = layer.wq_b(qr).unflatten(-1, (layer.n_heads, layer.index_head_dim))
    apply_rotary_emb(q[..., -2 * layer.freqs_cis.size(-1) :], layer.freqs_cis[0:S])
    weights = layer.weights_proj(x) * (layer.softmax_scale * layer.n_heads**-0.5)
    scores = (torch.einsum("bshd,btd->bsht", q, k).relu_() * weights.unsqueeze(-1)).sum(2)
    lens = (torch.arange(1, S + 1) // 1).unsqueeze(-1)
    scores = scores.masked_fill(torch.arange(S) >= lens, -torch.inf)
    for b in range(B):
        for s in range(S):
            valid_scores = scores[b, s][scores[b, s] > -torch.inf]
            n = min(layer.index_topk, int(valid_scores.numel()))
            if n == 0:
                continue
            threshold = valid_scores.sort(descending=True).values[n - 1]
            got = {i - 5 for i in idxs[b, s].tolist() if i >= 5}
            assert len(got) == n, f"({b},{s}) picked {len(got)} != {n}"
            for g in got:  # tie-tolerant: every pick must be a true top-k scorer
                assert scores[b, s, g] >= threshold - 1e-6, f"({b},{s}) picked {g} below threshold"


def test_indexer_oracle_overlap_prefill():
    """Oracle: our selection matches the verbatim upstream port on fixed inputs."""
    cfg = ced_toy_config()
    torch.manual_seed(42)
    layer = Indexer(cfg, 1)
    B, S = 2, 16
    x = torch.randn(B, S, cfg.d_model)
    qr = torch.randn(B, S, cfg.q_lora_rank)
    latent = torch.randn(B, S // 1, cfg.head_dim)
    shared = SharedAttentionRuntime()
    ours = layer(x, qr, latent, 0, offset=5, compress_len=S, shared=shared)
    shared2 = SharedAttentionRuntime()
    ref = oracle_indexer(layer, x, qr, latent, 0, 5, shared2)
    report = []
    exact = 0
    for b in range(B):
        for s in range(S):
            o = set(ours[b, s].tolist())
            r = set(ref[b, s].tolist())
            exact += o == r
            report.append(f"b{b} s{s}: overlap {len(o & r)}/{max(len(o), 1)}")
    assert exact == B * S, f"oracle mismatch: {report[:8]}"
    Path(__file__).parent.joinpath("golden").mkdir(exist_ok=True)
    Path(__file__).parent.joinpath("golden/csa2-oracle.md").write_text(
        "# CSA2 indexer oracle (Task 5)\n\n"
        "Upstream port: tests/oracle_upstream.py (model.py:527-610 verbatim, fp4/fp8\n"
        "quantization removed per Lite deviation D6). Fixed input: seed 42, B=2, S=16,\n"
        "ratio-1 decoder indexer (toy dims), candidate source on.\n\n"
        f"Result: {exact}/{B * S} query rows selected identical index sets; 0 divergences.\n"
    )


def test_candidate_pool_covers_decoders_only():
    """Encoders index the whole prefix; pool-users mask to the pool (T2)."""
    cfg = ced_toy_config()
    l0 = Indexer(cfg, 0)
    l1 = Indexer(cfg, 1)
    assert not l0.uses_candidates and l0.is_candidate_source is False
    assert l1.is_candidate_source and not l1.uses_candidates
    cfg2 = ced_toy_config(n_layers=3, n_encoder_layers=2, compress_ratios=(2, 1, 1),
                          index_source_layers=(0, 1, 2), dspark_target_layer_ids=(2,))
    l2 = Indexer(cfg2, 2)
    assert l2.uses_candidates


def test_reuse_mode_consumes_published_indices():
    """A non-source compressing layer must reuse the published selection."""
    cfg = ced_toy_config(compress_ratios=(2, 2), index_source_layers=(0,), candidate_source_layer=-1)
    # layer 1 (m=2, not an index source) -> reuse
    assert layer_mode(cfg, 1) == "reuse"
    assert cfg.global_kv_path(1) == "cache"


def test_mode_table_matches_contract(cfg=None):
    """Lite full-config mode table (contract T2 mapping)."""
    from models.config import load_config

    cfg = load_config()
    modes = {r["layer"]: r["mode"] for r in [{"layer": l, "mode": layer_mode(cfg, l)} for l in range(cfg.n_layers)]}
    assert modes[2] == "reindex" and modes[8] == "reindex"
    assert modes[12] == "full"
    assert modes[16] == modes[20] == "reindex"
    assert modes[3] == "reuse" and modes[13] == "reuse" and modes[21] == "reuse"
