"""CED global-KV pathway (Task 4 implements the wiring; this file pins parameters).

Contract: docs/architecture-contract.md T1. The producer map is config data
(kv_source_layers); encoder layers compress global KV (softmax-gated pooling,
m=2), decoder layers read the shared compressed stream — never re-project it.
"""

import torch
from torch import nn

from .layers import RMSNorm, init_std_


class Compressor(nn.Module):
    """Pools `compress_ratio` consecutive tokens into one KV latent with a
    learned softmax gate. m=1 is a plain projection (no gate). fp32 compute for
    m>1 pooling (upstream promotes the m=2 weights to fp32)."""

    def __init__(self, cfg, layer_id: int):
        super().__init__()
        ratio = cfg.compress_ratios[layer_id]
        assert ratio > 0, "Compressor belongs to layers with compress_ratio > 0"
        self.compress_ratio = ratio
        self.head_dim = cfg.head_dim
        self.norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.wkv = nn.Linear(cfg.d_model, cfg.head_dim, bias=False)
        if ratio > 1:
            self.wgate = nn.Linear(cfg.d_model, cfg.head_dim, bias=False)
        self.reset_parameters(ratio)

    def reset_parameters(self, ratio: int) -> None:
        dtype = torch.float32 if ratio > 1 else torch.get_default_dtype()
        self.wkv.weight.data = init_std_(self.wkv.weight.data.float()).to(dtype)
        if ratio > 1:
            self.wgate.weight.data = init_std_(self.wgate.weight.data.float()).to(dtype)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor | None:
        """x: [B, S, d] -> [B, S/ratio, head_dim] latent pre-RoPE (None mid-group at decode)."""
        raise NotImplementedError("Task 4: CED compression forward")


class ProducerMap:
    """Data-table view of the KV consumer map (contract T1): every ratio>0 layer
    reads the cache of the last kv_source at or before it. Swappable for gate C3."""

    def __init__(self, cfg):
        self.cfg = cfg

    def producer_of(self, layer_id: int) -> int | None:
        """Layer whose compressed cache `layer_id` reads; None if ratio 0."""
        if self.cfg.compress_ratios[layer_id] == 0:
            return None
        producers = [l for l in self.cfg.kv_source_layers if l <= layer_id]
        return producers[-1] if producers else None
