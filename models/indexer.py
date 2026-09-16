"""Hierarchical sparse indexer (Task 5 implements forward; parameters pinned here).

Contract: docs/architecture-contract.md T4. Owner (KV-source) layers project
index keys from the pre-RoPE compressed latent; every index source projects its
own queries from the q-Lora latent and scores against the published keys.
"""

import torch
from torch import nn

from .layers import RMSNorm, init_std_


class Indexer(nn.Module):
    def __init__(self, cfg, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.compress_ratio = cfg.compress_ratios[layer_id]
        self.owns_k = cfg.is_kv_source(layer_id)
        self.is_candidate_source = layer_id == cfg.candidate_source_layer
        self.uses_candidates = cfg.candidate_source_layer < layer_id
        self.n_heads = cfg.index_n_heads
        self.index_head_dim = cfg.index_head_dim
        self.index_topk = cfg.index_topk
        self.rope_head_dim = cfg.rope_head_dim
        self.softmax_scale = self.index_head_dim**-0.5

        self.wq_b = nn.Linear(cfg.q_lora_rank, self.n_heads * self.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(cfg.d_model, self.n_heads, bias=False)
        if self.owns_k:
            self.wk = nn.Linear(cfg.head_dim, self.index_head_dim, bias=False)
            self.k_norm = RMSNorm(self.index_head_dim, cfg.norm_eps)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_std_(self.wq_b.weight)
        init_std_(self.weights_proj.weight)
        if self.owns_k:
            init_std_(self.wk.weight)

    def forward(self, x, qr, latent, start_pos, offset):
        raise NotImplementedError("Task 5: indexer forward")
