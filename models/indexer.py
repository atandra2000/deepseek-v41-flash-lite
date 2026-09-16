"""Hierarchical sparse indexer (Task 5).

Contract: docs/architecture-contract.md T4/T2. Owner layers (KV sources, plus
the candidate source for the decoder CED path) project index keys from the
pre-RoPE latent and publish them; every index source projects its own queries
from the q-Lora latent, scores against the published keys (optionally masked
to the candidate pool), and publishes top-k indices that consumers reuse.
"""

import torch
from torch import nn

from .layers import RMSNorm, apply_rotary_emb, init_std_, precompute_freqs_cis


class Indexer(nn.Module):
    def __init__(self, cfg, layer_id: int, max_seq_len: int | None = None):
        super().__init__()
        self.layer_id = layer_id
        self.cfg = cfg
        self.compress_ratio = cfg.compress_ratios[layer_id]
        self.owns_k = cfg.is_kv_source(layer_id) or layer_id == cfg.candidate_source_layer
        self.is_candidate_source = layer_id == cfg.candidate_source_layer
        self.uses_candidates = 0 <= cfg.candidate_source_layer < layer_id
        self.n_heads = cfg.index_n_heads
        self.index_head_dim = cfg.index_head_dim
        self.index_topk = cfg.index_topk
        self.rope_head_dim = cfg.rope_head_dim
        self.softmax_scale = self.index_head_dim**-0.5
        self.max_seq_len = max_seq_len or cfg.context_train_stage2

        self.wq_b = nn.Linear(cfg.q_lora_rank, self.n_heads * self.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(cfg.d_model, self.n_heads, bias=False)
        # same rotary table as the owning attention (compress theta for m>1)
        theta = cfg.compress_rope_theta if self.compress_ratio else cfg.rope_theta
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.rope_head_dim, self.max_seq_len, theta),
                             persistent=False)
        self.k_cache: torch.Tensor | None = None
        if self.owns_k:
            self.wk = nn.Linear(cfg.head_dim, self.index_head_dim, bias=False)
            self.k_norm = RMSNorm(self.index_head_dim, cfg.norm_eps)
        init_std_(self.wq_b.weight)
        init_std_(self.weights_proj.weight)
        if self.owns_k:
            init_std_(self.wk.weight)

    def forward(self, x, qr, latent, start_pos, offset, compress_len, shared):
        """x [B,S,d]; qr [B,S,q_lora]; latent pre-RoPE [B,T,head_dim] (owners)
        or None; freqs_cis: the owning attention's rotary table (compress theta
        for compressing layers). Returns idxs [B,S,K] into the concatenated
        [window, compressed] kv axis (upstream model.py:527-580)."""
        from .csa2 import select_candidate_blocks

        bsz, seqlen, _ = x.shape
        ratio = self.compress_ratio
        freqs_cis = self.freqs_cis
        end_pos = start_pos + seqlen

        if self.owns_k and latent is not None:
            k = self.k_norm(self.wk(latent))
            if start_pos == 0:
                freqs = freqs_cis[: seqlen - seqlen % ratio : ratio]
            else:
                freqs = freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            apply_rotary_emb(k[..., -2 * freqs.size(-1) :], freqs)
            if self.k_cache is None or self.k_cache.size(1) < start_pos // ratio + k.size(1):
                # full-length buffer from the start, or decode writes no-op
                full = torch.zeros(bsz, self.max_seq_len // ratio, self.index_head_dim, dtype=k.dtype, device=k.device)
                full[:bsz, : k.size(1)] = k
                self.k_cache = full
            else:
                self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
            shared.index_k = self.k_cache

        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.index_head_dim))
        apply_rotary_emb(q[..., -2 * freqs_cis.size(-1) :], freqs_cis[start_pos:end_pos])

        assert shared.index_k is not None, "no index keys published yet"
        index_k = shared.index_k[:bsz, : end_pos // ratio]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        scores = torch.einsum("bshd,btd->bsht", q.float(), index_k.float())
        scores = (scores.relu_() * weights.unsqueeze(-1)).sum(dim=2)  # [B,S,T]

        # reachability: a group is visible once the query passed its first token
        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)  # [S,1]
            scores = scores.masked_fill(
                torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf
            )
        else:
            compress_lens = end_pos // ratio

        if self.is_candidate_source:
            shared.candidates = select_candidate_blocks(
                scores, compress_lens, self.cfg.candidate_topk_blocks, self.cfg.candidate_block_size
            )
        elif self.uses_candidates:
            assert shared.candidates is not None, "candidate pool not built yet"
            scores = scores.masked_fill(~shared.candidates, -torch.inf)

        topk = min(self.index_topk, end_pos // ratio)
        idxs = scores.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1)
