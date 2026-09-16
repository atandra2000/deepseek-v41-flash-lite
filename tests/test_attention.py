"""Task 3 verify: attention core forward/backward, causality, window boundaries."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.attention import Attention, sparse_attn, window_topk_idxs
from tests.toy import toy_config


@pytest.fixture(scope="module")
def cfg():
    return toy_config()


def build(cfg, layer_id=1):
    torch.manual_seed(1337)
    return Attention(cfg, layer_id, cfg.n_layers, max_seq_len=64)


def test_sparse_attn_matches_dense_reference():
    """sparse_attn vs dense masked softmax on the same kv (window path)."""
    torch.manual_seed(0)
    B, S, H, D, T, K = 2, 6, 4, 8, 6, 4
    q = torch.randn(B, S, H, D)
    kv = torch.randn(B, T, D)
    sink = torch.zeros(H)
    idxs = torch.full((B, S, K), -1, dtype=torch.long)
    for s in range(S):
        pool = list(range(s + 1))
        idxs[0, s, : len(pool[-K:])] = torch.tensor(pool[-K:])
        idxs[1, s, : len(pool[-K:])] = torch.tensor(pool[-K:])
    o = sparse_attn(q, kv, sink, idxs, D**-0.5)

    # dense reference over the SAME idx subset, sink in the denominator
    scores = torch.zeros(B, H, S, T)
    for b in range(B):
        for s in range(S):
            for k in range(K):
                t = int(idxs[b, s, k])
                if t >= 0:
                    scores[b, :, s, t] = torch.einsum("hd,d->h", q[b, s].float(), kv[b, t].float()) * D**-0.5
    scores = scores.masked_fill(scores == 0, -1e30)  # unreachable slots stay out of the softmax
    m = scores.amax(-1, keepdim=True)
    p = (scores - m).exp()
    denom = p.sum(-1, keepdim=True) + (sink.view(1, -1, 1, 1).float() - m).exp()
    ref = torch.einsum("bhst,btd->bhsd", p, kv.float()) / denom
    ref = ref.permute(0, 2, 1, 3)
    assert torch.allclose(o.float(), ref.float(), atol=1e-5)


def test_attn_sink_actually_biases():
    """A large positive sink must approach pure-kv-average-free behavior:
    output -> 0 as sink >> scores (softmax mass moves to the sink)."""
    torch.manual_seed(0)
    B, S, H, D = 1, 3, 2, 8
    q = torch.randn(B, S, H, D)
    kv = torch.randn(B, 5, D)
    idxs = torch.tensor([[0, 1, 2, 3, 4]]).expand(B, S, 5)
    small = sparse_attn(q, kv, torch.zeros(H), idxs, D**-0.5)
    big = sparse_attn(q, kv, torch.full((H,), 50.0), idxs, D**-0.5)
    assert big.abs().sum() < 1e-3 * small.abs().sum()


def test_forward_backward(cfg):
    attn = build(cfg)
    x = torch.randn(2, 12, cfg.d_model)
    y = attn(x, start_pos=0)
    assert y.shape == x.shape
    y.sum().backward()
    assert attn.wq_a.weight.grad is not None
    assert torch.isfinite(attn.wq_a.weight.grad).all()


def test_causality(cfg):
    """Perturbing a future token must not change earlier outputs."""
    attn = build(cfg)
    x1 = torch.randn(1, 10, cfg.d_model)
    x2 = x1.clone()
    x2[0, 7] += 3.0
    y1 = attn(x1, start_pos=0)
    y2 = attn(x2, start_pos=0)
    assert torch.allclose(y1[0, :7], y2[0, :7], atol=1e-5)
    assert not torch.allclose(y1[0, 7:], y2[0, 7:], atol=1e-5)


def test_window_boundary(cfg):
    """Window 8: query 10 sees positions 3..10; token 1 is outside -> no effect."""
    attn = build(cfg)
    x1 = torch.randn(1, 12, cfg.d_model)
    x2 = x1.clone()
    x2[0, 1] += 5.0
    y1 = attn(x1, start_pos=0)
    y2 = attn(x2, start_pos=0)
    assert torch.allclose(y1[0, 10:], y2[0, 10:], atol=1e-5)  # query 10 window starts at 3
    assert not torch.allclose(y1[0, 5], y2[0, 5], atol=1e-5)  # query 5 still sees token 1


def test_window_idxs_prefill_and_decode():
    idxs = window_topk_idxs(4, 6, 0)  # [1, S, W], row t = max(t-3,0)..t with -1 padding
    assert idxs[0, 0].tolist() == [0, -1, -1, -1]  # query 0 attends only itself
    assert idxs[0, 1].tolist() == [0, 1, -1, -1]
    assert idxs[0, 5].tolist() == [2, 3, 4, 5]
    # decode: ring, oldest first; start_pos=6 with window 4 -> slots [3(oldest),0,1,2]
    dec = window_topk_idxs(4, 1, 6)[0, 0]
    assert dec.tolist() == [3, 0, 1, 2]
    # ring still filling: start_pos=2 -> slot 3 not yet written
    dec2 = window_topk_idxs(4, 1, 2)[0, 0]
    assert dec2.tolist() == [-1, 0, 1, 2]


def test_decode_matches_prefill(cfg):
    """Prefill 11 tokens, then decode the 12th incrementally: the decode output
    must match position 11 of a full 12-token prefill (same weights, windows)."""
    torch.manual_seed(3)
    attn = build(cfg)
    x = torch.randn(1, 12, cfg.d_model)
    attn(x[:, :11], start_pos=0)  # prefill 11, seeds the ring
    y_step = attn(x[:, 11:12], start_pos=11)  # decode token 11
    torch.manual_seed(3)
    attn2 = build(cfg)
    y_full = attn2(x, start_pos=0)  # full 12-token prefill, independent module
    assert torch.allclose(y_step[:, 0], y_full[:, 11], atol=1e-4)
