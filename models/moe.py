"""MoE: 16 routed + 1 shared, top-2, noaux_tc bias balancing (contract T9)."""

import torch
import torch.nn.functional as F
from torch import nn

from .layers import init_scaled_output_, init_std_


class Expert(nn.Module):
    """SwiGLU FFN with the upstream training clamps (up clamped both sides,
    gate clamped above)."""

    def __init__(self, d_model: int, inter_dim: int, swiglu_limit: float = 0.0, n_layers: int | None = None):
        super().__init__()
        self.w1 = nn.Linear(d_model, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit
        self.reset_parameters(n_layers)

    def reset_parameters(self, n_layers: int | None = None) -> None:
        init_std_(self.w1.weight)
        init_std_(self.w3.weight)
        if n_layers is None:  # routed experts: plain init (each is a sub-path, scaled by gate weights)
            init_std_(self.w2.weight)
        else:
            init_scaled_output_(self.w2.weight, n_layers)

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class Gate(nn.Module):
    """noaux_tc: bias steers selection, weights come from raw scores."""

    def __init__(self, cfg):
        super().__init__()
        self.topk = cfg.n_activated_experts
        self.score_func = cfg.score_func
        self.norm_topk_prob = cfg.norm_topk_prob
        self.route_scale = cfg.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.d_model))
        self.bias = nn.Parameter(torch.empty(cfg.n_routed_experts, dtype=torch.float32))
        self.bias_vl = nn.Parameter(torch.empty(cfg.n_routed_experts, dtype=torch.float32))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.bias_vl)

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None):
        """x: [n, d]; image_mask: [n] bool -> (weights [n, topk], indices [n, topk])."""
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        elif self.score_func == "sqrtsoftplus":
            scores = F.softplus(scores).sqrt()
        else:
            raise ValueError(self.score_func)
        bias = self.bias if image_mask is None else torch.where(image_mask.unsqueeze(-1), self.bias_vl, self.bias)
        indices = (scores + bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)  # upstream: not norm_eps
        weights = weights * self.route_scale
        return weights, indices


class MoE(nn.Module):
    def __init__(self, cfg, n_layers: int):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_routed_experts = cfg.n_routed_experts
        self.n_activated_experts = cfg.n_activated_experts
        self.gate = Gate(cfg)
        self.experts = nn.ModuleList(
            [Expert(cfg.d_model, cfg.moe_inter_dim, cfg.swiglu_limit) for _ in range(cfg.n_routed_experts)]
        )
        self.shared_experts = Expert(cfg.d_model, cfg.moe_inter_dim, cfg.swiglu_limit, n_layers=n_layers)

    def load_counts(self, indices: torch.Tensor) -> torch.Tensor:
        """Per-expert token counts for the C1 gate (expert starvation check)."""
        return torch.bincount(indices.flatten(), minlength=self.n_routed_experts)

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.d_model)
        weights, indices = self.gate(x, image_mask.flatten() if image_mask is not None else None)
        y = torch.zeros_like(x, dtype=torch.float32)
        flat_indices = indices.flatten()
        sorted_idx = torch.argsort(flat_indices, stable=True)
        counts = torch.bincount(flat_indices, minlength=self.n_routed_experts)
        offsets = torch.cumsum(counts, 0) - counts
        for i in range(self.n_routed_experts):
            n = int(counts[i])
            if n == 0:
                continue
            rows = sorted_idx[offsets[i] : offsets[i] + n]
            y[rows] += self.experts[i](x[rows], weights[rows].float().unsqueeze(-1))
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)
