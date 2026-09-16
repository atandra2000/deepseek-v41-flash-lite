"""Latent attention core (Task 3 implements forward; this file pins parameters).

Contract: docs/architecture-contract.md T7. One KV latent of head_dim per
position shared across heads (upstream `num_key_value_heads: 1`), q low-rank,
grouped low-rank output projection, per-head attention sink, SWA-128 window
branch, optional compressed-global branch (CED, models/ced.py).
"""

import torch
from torch import nn

from .layers import RMSNorm, init_scaled_output_, init_std_


class Attention(nn.Module):
    def __init__(self, cfg, layer_id: int, n_layers: int):
        super().__init__()
        d = cfg.d_model
        self.layer_id = layer_id
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.n_groups = cfg.o_groups
        self.window_size = cfg.window_size
        self.compress_ratio = cfg.compress_ratios[layer_id]
        self.norm_eps = cfg.norm_eps

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
        self.reset_parameters(n_layers)

    def reset_parameters(self, n_layers: int) -> None:
        for m in (self.wq_a, self.wq_b, self.wkv, self.wo_a):
            init_std_(m.weight)
        init_scaled_output_(self.wo_b.weight, n_layers)  # type: ignore[arg-type]
        nn.init.zeros_(self.attn_sink)
