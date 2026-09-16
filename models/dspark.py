"""DSpark drafter blocks (contract T6; forward in Task 8). Lite: 2 dense blocks,
Markov rank 64, backbone frozen during P2."""

import torch
from torch import nn

from .attention import Attention
from .layers import RMSNorm, init_scaled_output_, init_std_
from .mhc import HCMixes


class DSparkExpertFFN(nn.Module):
    """Dense SwiGLU FFN (Lite: no routed experts in the drafter)."""

    def __init__(self, cfg, n_layers: int):
        super().__init__()
        d, inter = cfg.d_model, cfg.dspark_inter_dim
        self.w1 = nn.Linear(d, inter, bias=False)
        self.w2 = nn.Linear(inter, d, bias=False)
        self.w3 = nn.Linear(d, inter, bias=False)
        init_std_(self.w1.weight)
        init_std_(self.w3.weight)
        init_scaled_output_(self.w2.weight, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dense SwiGLU (no swiglu_limit clamp — that belongs to the routed MoE experts)
        gate = self.w1(x).float()
        up = self.w3(x).float()
        return self.w2((torch.nn.functional.silu(gate) * up).to(x.dtype))


class DSparkMarkovHead(nn.Module):
    """markov_rank embedding + vocab head over the full vocab (contract T6)."""

    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dspark_markov_rank)
        self.head = nn.Linear(cfg.dspark_markov_rank, cfg.vocab_size, bias=False)
        init_std_(self.embed.weight)
        init_std_(self.head.weight)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Task 8")


class DSparkConfidenceHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Linear(cfg.d_model + cfg.dspark_markov_rank, 1)
        init_std_(self.proj.weight)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Task 8")


class DSparkBlock(nn.Module):
    """One drafter block. Block 0 owns main_proj/main_norm (target-layer concat);
    the last block owns norm + markov head + confidence head."""

    def __init__(self, cfg, stage_id: int, n_stages: int, n_layers: int):
        super().__init__()
        self.stage_id = stage_id
        self.block_size = cfg.dspark_block_size
        self.noise_token_id = cfg.dspark_noise_token_id
        self.attn = Attention(cfg, layer_id=cfg.n_layers + stage_id, n_layers=n_layers)  # ratio 0 required
        assert self.attn.compress_ratio == 0, "DSpark blocks are SWA-only (ratio 0)"
        self.ffn = DSparkExpertFFN(cfg, n_layers)
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.hc_attn = HCMixes(cfg, "attn")
        self.hc_ffn = HCMixes(cfg, "ffn")
        if stage_id == 0:
            self.main_proj = nn.Linear(cfg.d_model * len(cfg.dspark_target_layer_ids), cfg.d_model, bias=False)
            self.main_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
            init_std_(self.main_proj.weight)
        if stage_id == n_stages - 1:
            self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.markov_head = DSparkMarkovHead(cfg)
            self.confidence_head = DSparkConfidenceHead(cfg)

    def forward(self, x, start_pos, pre_mix, main_x):
        raise NotImplementedError("Task 8")
