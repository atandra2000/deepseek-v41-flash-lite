"""Hyper-Connections: hc_mult parallel residual copies, Sinkhorn-balanced mixing (contract T8)."""

import torch
from torch import nn


class PlainResidual(nn.Module):
    """mHC-off variant stand-in (hc_mult 1): standard residual, no mixing
    parameters. Same forward interface as HCMixes so Block wiring is
    unchanged; pre/post/comb collapse to the identity."""

    def __init__(self, cfg, which: str):
        super().__init__()
        assert cfg.hc_mult == 1, "PlainResidual pairs with the derived hc_mult 1"

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ones = x.new_ones(x.size(0), x.size(1), 1)
        return ones, ones, ones.unsqueeze(-1)


class HCMixes(nn.Module):
    """Per-block coefficient generator (upstream Block.hc_mixes + hc_split_sinkhorn).

    fp32 params: hc_fn [(2+hc)*hc, hc*d], hc_base [(2+hc)*hc], hc_scale [3]."""

    def __init__(self, cfg, which: str):
        super().__init__()
        hc = cfg.hc_mult
        mix_hc = (2 + hc) * hc
        assert which in ("attn", "ffn")
        self.hc_mult = hc
        self.sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.eps = cfg.hc_eps
        self.norm_eps = cfg.norm_eps
        self.fn = nn.Parameter(torch.empty(mix_hc, hc * cfg.d_model, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Identity-init (contract T8 Lite decision): pre one-hot copy 0, post 1,
        comb one-hot diagonal. fn columns encode the logits; scale = 1."""
        hc = self.hc_mult
        mix_hc = (2 + hc) * hc
        with torch.no_grad():
            self.fn.zero_()
            base = torch.zeros(mix_hc)
            base[:hc] = torch.logit(torch.tensor(1.0 - self.eps))  # pre -> 1 - eps ~= 1, one-hot copy 0
            base[hc : 2 * hc] = 0.0  # post = 2*sigmoid(0) = 1
            big = 20.0  # softmax gap far above any activation-scale noise
            for j in range(hc):  # comb block: mix[j*hc + k + 2hc] -> identity
                base[2 * hc + j * hc : 2 * hc + (j + 1) * hc] = -big
                base[2 * hc + j * hc + j] = big
            self.base.copy_(base)
            self.scale.fill_(1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [b, s, hc, d] -> pre [b,s,hc], post [b,s,hc], comb [b,s,hc,hc]."""
        b, s = x.size(0), x.size(1)
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = torch.nn.functional.linear(flat, self.fn) * rsqrt

        hc = self.hc_mult
        pre = torch.sigmoid(mixes[..., :hc] * self.scale[0] + self.base[:hc]) + self.eps
        post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * self.scale[1] + self.base[hc : 2 * hc])
        comb = mixes[..., 2 * hc :] * self.scale[2] + self.base[2 * hc :]
        comb = comb.view(b, s, hc, hc)
        # upstream: softmax over last dim, + eps, col-normalize, then (iters-1) row/col passes
        comb = comb.softmax(dim=-1) + self.eps
        for _ in range(self.sinkhorn_iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        return pre, post, comb

    @staticmethod
    def hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Collapse hc copies: [b,s,hc,d] x [b,s,hc] -> [b,s,d]."""
        return torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)

    @staticmethod
    def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        """Expand sublayer output and mix residual: -> [b,s,hc,d]."""
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)
