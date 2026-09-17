"""Shared primitives: RMSNorm, rotary embeddings, init helpers.

Semantics pinned from the upstream reference (docs/architecture-contract.md);
upstream line references live in the contract, not here.
"""

import torch
from torch import nn


def init_std_(t: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    nn.init.normal_(t, std=std)
    return t


def init_scaled_output_(t: torch.Tensor, n_layers: int, std: float = 0.02) -> torch.Tensor:
    """Output projection of a block: std scaled by 1/sqrt(2*n_layers) (design §8.2)."""
    return init_std_(t, std / (2.0 * n_layers) ** 0.5)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@torch.no_grad()
def precompute_freqs_cis(dim: int, seqlen: int, base: float) -> torch.Tensor:
    """Rotary frequencies, one row per position; Lite uses no YaRN."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = torch.outer(torch.arange(seqlen, dtype=torch.float32), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate adjacent element pairs as complex numbers (upstream model.py:392-406).

    Accepts [b, s, d] and [b, s, h, d]; `inverse` conjugates the rotation."""
    orig_dtype = x.dtype
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    half = xc.size(-1)
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), half)
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, half)
    out = torch.view_as_real(xc * freqs_cis).flatten(-2)
    return out.to(orig_dtype)
