"""Shared primitives: RMSNorm, rotary embeddings, linear, init helpers.

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


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    return nn.functional.linear(x, weight, bias)


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
def precompute_freqs_cis(dim: int, seqlen: int, base: float, original_seq_len: int = 0,
                         factor: float = 1.0, beta_fast: int = 32, beta_slow: int = 1) -> torch.Tensor:
    """Rotary frequencies as complex exponentials, one row per position.

    With original_seq_len > 0 applies YaRN (upstream model.py:368-389). Lite
    trains at 4K/16K <= the 65,536 original context, so stage configs pass
    original_seq_len=0 and keep base theta.
    """
    import math

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations: float) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    freqs = torch.outer(torch.arange(seqlen, dtype=torch.float32), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate adjacent element pairs as complex numbers (upstream model.py:392-406).

    Accepts [b, s, d] and [b, s, h, d]; `inverse` conjugates the rotation."""
    orig_dtype = x.dtype
    x_ = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_.size(1), x_.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_.size(1), 1, x_.size(-1))
    out = torch.view_as_real(xc * freqs_cis).flatten(-2)
    x_.copy_(out.to(orig_dtype))
    return x_
