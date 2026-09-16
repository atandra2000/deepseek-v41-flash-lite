"""Transformer root: assembles embed -> blocks (CED/CSA2/mHC/Engram/MoE) -> head,
plus ViT/aligner and DSpark drafter. Forward lands in Tasks 3-8; this module
already owns the complete parameter tree so the Task-2 ledger is real."""

import torch
from torch import nn

from .attention import Attention
from .ced import Compressor
from .engram import Engram
from .indexer import Indexer
from .layers import RMSNorm, init_scaled_output_, init_std_
from .mhc import HCMixes
from .moe import MoE


class Block(nn.Module):
    def __init__(self, cfg, layer_id: int, n_layers: int):
        super().__init__()
        self.layer_id = layer_id
        self.attn = Attention(cfg, layer_id, n_layers)
        self.ffn = MoE(cfg, n_layers)
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.hc_attn = HCMixes(cfg, "attn")
        self.hc_ffn = HCMixes(cfg, "ffn")
        self.compressor: Compressor | None = None
        if cfg.is_kv_source(layer_id):
            self.compressor = Compressor(cfg, layer_id)
        self.indexer: Indexer | None = None
        if cfg.is_index_source(layer_id):
            self.indexer = Indexer(cfg, layer_id)
        self.engram: Engram | None = None
        if layer_id in cfg.engram_layer_ids:
            self.engram = Engram(cfg, layer_id)


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        init_std_(self.embed.weight)
        self.blocks = nn.ModuleList([Block(cfg, l, cfg.n_layers) for l in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        if cfg.tie_word_embeddings:
            self.head: nn.Linear | None = None
        else:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            init_std_(self.head.weight)

        self.vision = None
        self.aligner = None
        self.image_start: nn.Parameter | None = None
        self.image_end: nn.Parameter | None = None
        self.image_newline: nn.Parameter | None = None
        if cfg.vision_n_layers > 0:
            from .vit import Aligner, ViT

            self.vision = ViT(cfg)
            self.aligner = Aligner(cfg)
            self.image_start = nn.Parameter(torch.empty(cfg.d_model))
            self.image_end = nn.Parameter(torch.empty(cfg.d_model))
            self.image_newline = nn.Parameter(torch.empty(cfg.d_model))
            for p in (self.image_start, self.image_end, self.image_newline):
                init_std_(p)

        self.dspark: nn.ModuleList | None = None
        if cfg.dspark_block_size:
            from .dspark import DSparkBlock

            self.dspark = nn.ModuleList(
                [
                    DSparkBlock(cfg, s, cfg.num_nextn_predict_layers, cfg.n_layers)
                    for s in range(cfg.num_nextn_predict_layers)
                ]
            )

    def head_weight(self) -> torch.Tensor:
        """Tied head: the embedding weight (D4). Untied config would own it."""
        if self.head is not None:
            return self.head.weight
        return self.embed.weight

    def forward(self, input_ids, start_pos: int = 0, images=None, token_types=None):
        raise NotImplementedError("Tasks 3-8 wire the full forward")
