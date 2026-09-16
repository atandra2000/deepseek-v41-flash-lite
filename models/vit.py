"""DeepSeek ViT + 2-layer MLP aligner (contract T10).

Span types mirror upstream image_processor.py: every span position carries
`image_token_id` in input_ids; only the type distinguishes the slots. The
IMAGE slots take aligner rows in reading order, delimiters take learned
embeddings (Transformer.merge_image_embeddings).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .layers import RMSNorm, init_std_

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


@dataclass
class ImageSpan:
    """One image's place in a sample's token sequence (upstream ImageInput)."""

    start: int
    patches: torch.Tensor  # [n_vit_h * n_vit_w, 3, patch, patch]
    n_vit_h: int
    n_vit_w: int
    types: torch.Tensor  # [span_len] one of the constants above


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    """[IMAGE_START] + ([IMAGE]*n_w + [NEWLINE])*n_h + [IMAGE_END]."""
    return n_llm_h * (n_llm_w + 1) + 2


class PatchEmbed(nn.Module):
    """Linear over flattened patch pixels; the 3x3 pixel-unshuffle is a reshape
    done by the data pipeline (contract T10)."""

    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Linear(3 * cfg.vision_patch_size**2, cfg.vision_dim)
        init_std_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


def _vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    """2D-RoPE tables (upstream get_vision_cos_sin): h/w position pairs
    interleaved into one freq vector, [n, dim] cos/sin."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    return freqs.flatten(1).cos().unsqueeze(1), freqs.flatten(1).sin().unsqueeze(1)


def _apply_vision_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Half-split rotation (upstream vision.apply_rotary — NOT the adjacent-pair
    convention of layers.apply_rotary_emb)."""
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


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
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        q = _apply_vision_rotary(q, cos, sin)
        k = _apply_vision_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return self.wo(o.transpose(0, 1).reshape(n, -1))


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
        """patches [n_h*n_w, 3, patch, patch] -> normalized features [n_h*n_w, vision_dim]."""
        assert patches.size(0) == n_h * n_w, "patch count must fill the grid"
        x = self.patch_embed(patches)
        cos, sin = _vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    """2-layer MLP projector: vit_dim*9 -> d_model (GELU, per contract T10)."""

    def __init__(self, cfg):
        super().__init__()
        self.downsample_ratio = cfg.vision_downsample_ratio
        in_dim = cfg.vision_dim * cfg.vision_downsample_ratio**2
        self.w1 = nn.Linear(in_dim, cfg.d_model)
        self.w2 = nn.Linear(cfg.d_model, cfg.d_model)
        init_std_(self.w1.weight)
        init_std_(self.w2.weight)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """Patch features [n_h*n_w, vision_dim] -> [ceil(n_h/r)*ceil(n_w/r), d_model]."""
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))
