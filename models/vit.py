"""DeepSeek ViT + 2-layer MLP aligner (contract T10; forward in Task 8)."""

import torch
from torch import nn

from .layers import RMSNorm, init_std_


class PatchEmbed(nn.Module):
    """Linear over flattened patch pixels; the 3x3 pixel-unshuffle is a reshape
    done by the data pipeline (contract T10)."""

    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Linear(3 * cfg.vision_patch_size**2, cfg.vision_dim)
        init_std_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class VitAttention(nn.Module):
    """Full bidirectional attention with 2D-RoPE (upstream SDPA)."""

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.vision_n_heads
        self.head_dim = cfg.vision_dim // cfg.vision_n_heads
        self.wqkv = nn.Linear(cfg.vision_dim, 3 * cfg.vision_dim, bias=False)
        self.wo = nn.Linear(cfg.vision_dim, cfg.vision_dim, bias=False)
        init_std_(self.wqkv.weight)
        init_std_(self.wo.weight)

    def forward(self, x, cos, sin):
        raise NotImplementedError("Task 8")


class VitMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.vision_dim, 2 * cfg.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(cfg.vision_inter_dim, cfg.vision_dim, bias=False)
        init_std_(self.w1.weight)
        init_std_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(torch.nn.functional.silu(gate) * up)


class VitBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.vision_dim, cfg.norm_eps)
        self.attn = VitAttention(cfg)
        self.norm2 = RMSNorm(cfg.vision_dim, cfg.norm_eps)
        self.mlp = VitMLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.rope_dim = cfg.vision_dim // cfg.vision_n_heads // 2
        self.rope_theta = cfg.rope_theta
        self.patch_embed = PatchEmbed(cfg)
        self.blocks = nn.ModuleList([VitBlock(cfg) for _ in range(cfg.vision_n_layers)])
        self.norm = RMSNorm(cfg.vision_dim, cfg.norm_eps)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        raise NotImplementedError("Task 8")


class Aligner(nn.Module):
    """2-layer MLP projector: vit_dim*9 -> d_model (GELU, per contract T10)."""

    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg.vision_dim * cfg.vision_downsample_ratio**2
        self.w1 = nn.Linear(in_dim, cfg.d_model)
        self.w2 = nn.Linear(cfg.d_model, cfg.d_model)
        init_std_(self.w1.weight)
        init_std_(self.w2.weight)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        raise NotImplementedError("Task 8")
