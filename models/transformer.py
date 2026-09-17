"""Transformer root (Task 4 wiring; Engram lands in Task 6).

embed -> hc_mult copies -> blocks (CED/CSA2 attention + MoE, chained mHC
pre-mix) -> collapse -> tied head. ViT/aligner and DSpark blocks attach in
Phase 2. Producer hidden states are stashed for the decoder CED projection.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import Attention, SharedAttentionRuntime
from .ced import CEDRuntime, CedProjection, Compressor, ced_attention_forward
from .engram import Engram
from .indexer import Indexer
from .layers import RMSNorm, init_std_
from .mhc import HCMixes, PlainResidual
from .moe import Expert, MoE
from .variants import FULL, VariantSpec
from .vit import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START


class Block(nn.Module):
    def __init__(self, cfg, layer_id: int, n_layers: int, max_seq_len: int | None = None,
                 variant: VariantSpec | None = None):
        super().__init__()
        self.layer_id = layer_id
        self.cfg = cfg
        self.variant = FULL if variant is None else variant
        self.attn = Attention(cfg, layer_id, n_layers, max_seq_len=max_seq_len)
        if self.variant.moe:
            self.ffn = MoE(cfg, n_layers)
        else:
            # Dense control with the routed top-k + shared experts' combined
            # active FFN width (C0 baseline; matched active parameters).
            inter = (cfg.n_activated_experts + cfg.n_shared_experts) * cfg.moe_inter_dim
            self.ffn = Expert(cfg.d_model, inter, swiglu_limit=0, n_layers=n_layers)
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.hc_attn = HCMixes(cfg, "attn") if self.variant.mhc else PlainResidual(cfg, "attn")
        self.hc_ffn = HCMixes(cfg, "ffn") if self.variant.mhc else PlainResidual(cfg, "ffn")
        if self.variant.mhc_frozen:  # C5 fallback: identity-frozen mHC
            for hc in (self.hc_attn, self.hc_ffn):
                for p in hc.parameters():
                    p.requires_grad_(False)
        self.compressor: Compressor | None = None
        if cfg.global_kv_path(layer_id) == "own":
            self.compressor = Compressor(cfg, layer_id)
        self.ced_projection: CedProjection | None = None
        if cfg.global_kv_path(layer_id) == "project":
            self.ced_projection = CedProjection(cfg)
        self.indexer: Indexer | None = None
        if self.variant.csa2 and cfg.is_index_source(layer_id):
            self.indexer = Indexer(cfg, layer_id)
        self.engram: Engram | None = None
        if layer_id in cfg.engram_layer_ids:
            self.engram = Engram(cfg, layer_id)

    def forward(self, x, start_pos, pre_mix, ced: CEDRuntime, ngram_hashes=None, engram_mask=None,
                image_mask=None):
        """x: [B,S,hc,d]; pre_mix: previous block's ffn pre-coefficients.
        Returns (h, ffn_pre) — ffn_pre feeds the next block's attention."""
        if self.engram is not None and ngram_hashes is not None:
            h = self.engram(x, ngram_hashes[:, :, self.engram.hash_index, :], engram_mask)
        else:
            h = x
        residual = h
        a_pre, a_post, a_comb = self.hc_attn(h)
        x_c = HCMixes.hc_pre(h, pre_mix)
        x_c = self.attn_norm(x_c)
        producer_hidden = None
        if self.cfg.global_kv_path(self.layer_id) == "project":
            producers = [p for p in self.cfg.kv_source_layers if p < self.layer_id]
            producer_hidden = ced.producer_hidden[producers[-1]]
        x_c = ced_attention_forward(self.attn, x_c, start_pos, ced, indexer=self.indexer,
                                    compressor=self.compressor, ced_projection=self.ced_projection,
                                    producer_hidden=producer_hidden)
        h = HCMixes.hc_post(x_c, residual, a_post, a_comb)

        residual = h
        f_pre, f_post, f_comb = self.hc_ffn(h)
        x_c = HCMixes.hc_pre(h, a_pre)
        x_c = self.ffn_norm(x_c)
        if self.variant.moe:
            x_c = self.ffn(x_c, image_mask)
        else:
            x_c = self.ffn(x_c)
        h = HCMixes.hc_post(x_c, residual, f_post, f_comb)
        return h, f_pre


class Transformer(nn.Module):
    def __init__(self, cfg, max_seq_len: int | None = None, variant: VariantSpec | None = None):
        super().__init__()
        self.cfg = cfg
        self.variant = FULL if variant is None else variant
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        init_std_(self.embed.weight)
        self.blocks = nn.ModuleList(
            [Block(cfg, l, cfg.n_layers, max_seq_len=max_seq_len, variant=self.variant)
             for l in range(cfg.n_layers)]
        )
        # cross-layer publishes persist across prefill/decode calls (upstream
        # keeps one global SharedAttentionRuntime per process)
        self.shared_attn = SharedAttentionRuntime()
        self.ced_runtime = CEDRuntime(self.shared_attn)
        self.ngram_hash = None
        if cfg.engram_layer_ids:
            from .engram import NgramHashState

            self.ngram_hash = NgramHashState(cfg, max_seq_len or cfg.context_train_stage2)
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = None  # Lite requires a tied embedding/head.

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
        return self.embed.weight

    # ---- vision (contract T10) ----

    def encode_image(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        """Patch grid -> aligner rows [ceil(n_h/r)*ceil(n_w/r), d_model]."""
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def merge_image_embeddings(self, images, h: torch.Tensor) -> None:
        """Overwrite each image's token span in h in place (upstream
        model.py:1228-1239): IMAGE slots take aligner rows in reading order,
        delimiters take learned embeddings. images: per-sample lists of
        vit.ImageSpan (or None)."""
        for i, sample in enumerate(images):
            for img in sample or ():
                types = img.types.to(h.device)
                span = h[i, img.start : img.start + types.numel()]
                span[types == IMAGE_START] = self.image_start.to(h.dtype)
                span[types == IMAGE_END] = self.image_end.to(h.dtype)
                span[types == IMAGE_NEW_LINE] = self.image_newline.to(h.dtype)
                embeds = self.encode_image(img.patches.to(h.device), img.n_vit_h, img.n_vit_w)
                assert (types == IMAGE).sum() == embeds.size(0), "IMAGE slot count != aligner rows"
                span[types == IMAGE] = embeds.to(h.dtype)

    # ---- DSpark drafting (contract T6) ----

    def forward_spec(self, input_ids, main_hidden, start_pos: int = 0, temperature: float = 0.0):
        """Draft step. Prefill (start_pos 0) only seeds the rings -> None;
        decode returns (output_ids [B, block+1], logits [B, block, vocab]
        fp32, confidence [B, block]). main_hidden: the target layers'
        attention-input hiddens concatenated on the feature dim (forward's
        main_hiddens, cat'd)."""
        if not self.dspark:
            return None
        bsz = input_ids.size(0)
        h, main_x = self.dspark[0].forward_embed(self, main_hidden, input_ids)
        pre_mix = h.new_zeros(bsz, h.size(1), self.cfg.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0
        for blk in self.dspark:
            h, pre_mix = blk(h, start_pos, pre_mix, main_x)
        if start_pos == 0:
            return None
        return self.dspark[-1].forward_head(h, pre_mix, input_ids, self, temperature)

    def forward(self, input_ids, start_pos: int = 0, ngram_hashes=None, engram_mask=None, image_mask=None,
                images=None, checkpoint_activations: bool = False):
        """Full forward (prefill and single-token decode). Returns (logits fp32
        [B,S,vocab], main_hiddens list-or-None). images: per-sample lists of
        vit.ImageSpan — spans must lie inside the first (start_pos 0) chunk;
        image_mask marks every span position (engram skip + MoE bias_vl).
        checkpoint_activations: uses torch.utils.checkpoint on each block."""
        cfg = self.cfg
        bsz, seqlen = input_ids.shape
        shared = self.shared_attn
        ced = self.ced_runtime
        engram_mask = None if image_mask is None else ~image_mask  # image spans take no n-grams
        hashes = self.ngram_hash(input_ids, start_pos, engram_mask) if self.ngram_hash is not None else None

        h = self.embed(input_ids)
        if images is not None:
            assert start_pos == 0, "image spans must be prefilled in a single chunk"
            self.merge_image_embeddings(images, h)
        h = h.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        pre_mix = h.new_zeros(bsz, seqlen, cfg.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0

        main_hiddens = []
        for blk in self.blocks:
            if blk.layer_id in cfg.dspark_target_layer_ids:
                # the MTP head reads the ATTENTION INPUT of its target layers
                # (contract T6, upstream model.py:1264-1266), i.e. the stream
                # entering the block
                main_hiddens.append(h.mean(dim=2))
            if checkpoint_activations and self.training and start_pos == 0:
                def make_custom_forward(b):
                    def custom_forward(x_in, p_mix):
                        return b(x_in, start_pos, p_mix, ced, hashes, engram_mask, image_mask)
                    return custom_forward

                h, pre_mix = checkpoint(make_custom_forward(blk), h, pre_mix, use_reentrant=False)
            else:
                h, pre_mix = blk(h, start_pos, pre_mix, ced, hashes, engram_mask, image_mask)
            if blk.layer_id in cfg.kv_source_layers:
                # producers publish their OUTPUT hidden state (collapsed to one
                # stream) for the decoder CED projection (design §4.1)
                ced.producer_hidden[blk.layer_id] = HCMixes.hc_pre(h, pre_mix)

        # final collapse uses the last block's ffn pre-coefficients
        h = HCMixes.hc_pre(h, pre_mix)
        logits = F.linear(self.norm(h).float(), self.head_weight().float())
        return logits, (main_hiddens or None)
