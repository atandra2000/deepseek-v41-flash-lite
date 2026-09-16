"""Task-5 oracle: a verbatim-style torch port of the upstream indexer math
(upstream/inference/model.py:527-610, minus the fp4/fp8 quantization — Lite
trains BF16 KV per design deviation D6). Used to cross-check our Indexer's
selected indices on fixed inputs; the quantization delta is the only
intentional difference and it is reported, not hidden."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch.nn.functional as F


def oracle_select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    """Verbatim port of upstream select_candidate_blocks (model.py:583-610)."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def oracle_indexer(layer, x, qr, latent, start_pos, offset, shared):
    """Port of upstream Indexer.forward (model.py:527-580) against our module's
    weights. `layer` is our Indexer (weights only); semantics follow upstream
    line by line."""
    from models.layers import apply_rotary_emb

    bsz, seqlen, _ = x.size()
    ratio = layer.compress_ratio
    rd = layer.rope_head_dim
    freqs_cis = layer.freqs_cis
    end_pos = start_pos + seqlen

    if layer.owns_k and latent is not None:
        if start_pos == 0:
            freqs = freqs_cis[: seqlen - seqlen % ratio : ratio]
        else:
            freqs = freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
        k = layer.k_norm(layer.wk(latent))
        apply_rotary_emb(k[..., -rd:], freqs)
        shared.index_k = k  # oracle: no fp4 quantization, direct publish

    q = layer.wq_b(qr).unflatten(-1, (layer.n_heads, layer.index_head_dim))
    apply_rotary_emb(q[..., -rd:], freqs_cis[start_pos:end_pos])

    index_k = shared.index_k[:bsz, : end_pos // ratio]
    weights = layer.weights_proj(x) * (layer.softmax_scale * layer.n_heads**-0.5)
    index_score = torch.einsum("bshd,btd->bsht", q, index_k)
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

    if start_pos == 0:
        compress_lens = (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1)
        index_score = index_score.masked_fill(torch.arange(seqlen // ratio) >= compress_lens, -torch.inf)
    else:
        compress_lens = end_pos // ratio

    if layer.is_candidate_source:
        shared.candidates = oracle_select_candidate_blocks(
            index_score, compress_lens, layer.cfg.candidate_topk_blocks, layer.cfg.candidate_block_size
        )
    elif layer.uses_candidates:
        index_score = index_score.masked_fill(~shared.candidates, -torch.inf)

    topk = min(layer.index_topk, end_pos // ratio)
    idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    return torch.where(idxs < compress_lens, idxs + offset, -1)
