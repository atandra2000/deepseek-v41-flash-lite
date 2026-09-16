"""Latent attention core (Task 3).

Contract: docs/architecture-contract.md T7. One KV latent of head_dim per
position shared across heads (upstream `num_key_value_heads: 1`), q low-rank,
grouped low-rank output projection, per-head attention sink, SWA-128 window
branch, optional compressed-global branch (models/ced.py). Pure-torch
sparse_attn replaces the upstream tilelang kernel; numerics follow
kernel.py:310-403 (finite -1e30 bound, sink in the denominator only).
"""

import torch
from torch import nn

from .layers import RMSNorm, apply_rotary_emb, init_scaled_output_, init_std_, precompute_freqs_cis


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
                softmax_scale: float, query_chunk: int = 1024) -> torch.Tensor:
    """q [B,S,H,D], kv [B,T,D] shared across heads, topk_idxs [B,S,K] (-1 = skip).

    Dense re-implementation of kernel.py:310-403 with the same convention:
    finite -1e30 mask bound (all-skip rows output 0), sink in the denominator
    only. Queries processed in chunks so the gathered kv never materializes
    at B*S*K*D (production would be ~3 GiB).
    """
    B, S = q.size(0), q.size(1)
    D = kv.size(-1)
    outs = []
    for s0 in range(0, S, query_chunk):
        qs = q[:, s0 : s0 + query_chunk]
        idx = topk_idxs[:, s0 : s0 + query_chunk]
        valid = idx >= 0
        # gather kv rows per query: [B, s, K, D]
        kv_g = kv.gather(
            1, idx.clamp_min(0).unsqueeze(-1).expand(B, -1, -1, D).reshape(B, -1, D)
        ).view(B, qs.size(1), -1, D)
        scores = torch.einsum("bshd,bskd->bhsk", qs.float(), kv_g.float()) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(1), -1e30)
        m = scores.amax(-1, keepdim=True)
        p = torch.exp(scores - m)
        p = p.masked_fill(~valid.unsqueeze(1), 0.0)
        denom = p.sum(-1, keepdim=True) + torch.exp(attn_sink.float().view(1, -1, 1, 1) - m)
        o = torch.einsum("bhsk,bskd->bhsd", p, kv_g.float()) / denom
        outs.append(o.to(q.dtype))
    o = torch.cat(outs, dim=2)  # [B,H,S,D]
    return o.permute(0, 2, 1, 3)  # [B,S,H,D]


def window_topk_idxs(window_size: int, seqlen: int, start_pos: int) -> torch.Tensor:
    """Sliding-window cache slots each query attends to (upstream
    get_window_topk_idxs, model.py:409-426). Prefill: causal window over the
    chunk layout. Decode (one query): the whole ring, oldest first."""
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size)).unsqueeze(0)
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs).unsqueeze(0)
    return idxs.unsqueeze(0)  # [1, S, W] broadcast over batch


class SharedAttentionRuntime:
    """Cross-layer publishes (upstream SharedAttentionRuntime, model.py:1166-1180):
    compress_kv + index_k from kv sources, topk_idxs from index sources,
    candidates from the candidate source. Sources write before consumers read."""

    def __init__(self):
        self.compress_kv: torch.Tensor | None = None
        self.index_k: torch.Tensor | None = None
        self.topk_idxs: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None


class Attention(nn.Module):
    def __init__(self, cfg, layer_id: int, n_layers: int, max_seq_len: int | None = None):
        super().__init__()
        d = cfg.d_model
        self.layer_id = layer_id
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.n_groups = cfg.o_groups
        self.o_lora_rank = cfg.o_lora_rank
        self.window_size = cfg.window_size
        self.compress_ratio = cfg.compress_ratios[layer_id]
        self.norm_eps = cfg.norm_eps
        self.max_seq_len = max_seq_len or cfg.context_train_stage2
        self.is_kv_source = cfg.is_kv_source(layer_id)
        self.is_index_source = cfg.is_index_source(layer_id)

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.wq_a = nn.Linear(d, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps)
        self.wq_b = nn.Linear(cfg.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(d, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        heads_per_group = self.n_heads * self.head_dim // self.n_groups
        # wo_a is block-diagonal over groups (upstream: einsum, not dense Linear)
        self.wo_a = nn.Linear(heads_per_group, self.n_groups * cfg.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * cfg.o_lora_rank, d, bias=False)
        self.softmax_scale = self.head_dim**-0.5

        # compressing layers rotate q/kv/latents at compress_rope_theta; pure-SWA
        # layers at base theta (upstream model.py:680-687). No YaRN at Lite dims.
        theta = cfg.compress_rope_theta if self.compress_ratio else cfg.rope_theta
        freqs = precompute_freqs_cis(self.rope_head_dim, self.max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs, persistent=False)

        # decode-time ring/global caches, allocated lazily on first decode
        self.window_kv_cache: torch.Tensor | None = None
        if self.is_kv_source:
            self.compress_kv_cache: torch.Tensor | None = None

        self.reset_parameters(n_layers)

    def reset_parameters(self, n_layers: int) -> None:
        for m in (self.wq_a, self.wq_b, self.wkv, self.wo_a):
            init_std_(m.weight)
        init_scaled_output_(self.wo_b.weight, n_layers)  # type: ignore[arg-type]
        nn.init.zeros_(self.attn_sink)

    # ---- window branch ----

    def _window_kv(self, x: torch.Tensor, start_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """This layer's sliding-window K (post-norm, post-RoPE) + window indices."""
        win = self.window_size
        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -self.rope_head_dim :], self.freqs_cis[start_pos : start_pos + kv.size(1)])
        if start_pos == 0:
            window_kv = kv
            # seed the ring for later decode steps (upstream model.py:708-716)
            bsz, seqlen, _ = kv.shape
            if self.window_kv_cache is None:
                self.window_kv_cache = torch.zeros(bsz, win, self.head_dim, dtype=kv.dtype, device=kv.device)
            if seqlen <= win:
                self.window_kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = kv[:, -win:].split(
                    [win - cutoff, cutoff], dim=1
                )
        else:  # decode: one token into the ring, attend the whole window
            bsz = x.size(0)
            if self.window_kv_cache is None:
                self.window_kv_cache = torch.zeros(bsz, win, self.head_dim, dtype=kv.dtype, device=kv.device)
            self.window_kv_cache[:bsz, start_pos % win] = kv[:, 0]
            window_kv = self.window_kv_cache[:bsz]
        return window_kv, window_topk_idxs(win, x.size(1), start_pos)

    # ---- attention over [window (+ compressed via ced wiring)] ----

    def q_proj(self, x: torch.Tensor, start_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """q heads [B,S,H,D] (RoPE'd tail) and the q-Lora latent qr for the indexer."""
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -self.rope_head_dim :], self.freqs_cis[start_pos : start_pos + x.size(1)])
        return q, qr

    def attend(self, q: torch.Tensor, kv: torch.Tensor, topk_idxs: torch.Tensor, start_pos: int, seqlen: int) -> torch.Tensor:
        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -self.rope_head_dim :], self.freqs_cis[start_pos : start_pos + seqlen], inverse=True)
        bsz = o.size(0)
        o = o.reshape(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a.float())
        return self.wo_b(o.flatten(2).to(o.dtype))

    def forward(self, x: torch.Tensor, start_pos: int, qr: torch.Tensor | None = None, shared: SharedAttentionRuntime | None = None) -> torch.Tensor:
        """SWA-only path (ratio 0). Compressing layers are driven by
        Transformer/ced wiring, which reuses q_proj/attend around the indexer
        and compressor. qr: precomputed q-Lora latent (ced wiring path)."""
        assert self.compress_ratio == 0, "compressing layers go through the CED wiring"
        q = None
        if qr is None:
            q, _ = self.q_proj(x, start_pos)
        else:
            q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
            apply_rotary_emb(q[..., -self.rope_head_dim :], self.freqs_cis[start_pos : start_pos + x.size(1)])
        window_kv, window_idxs = self._window_kv(x, start_pos)
        o = self.attend(q, window_kv, window_idxs, start_pos, x.size(1))
        return o
