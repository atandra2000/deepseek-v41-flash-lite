"""DSpark drafter (contract T6). Lite: 2 dense blocks, Markov rank 64,
backbone frozen during P2.

Prefill only seeds each block's SWA ring from the projected main hidden
(upstream DSparkAttention, model.py:1032-1074); decode drafts a block of
`dspark_block_size` tokens attending [window ring, own draft KV]. Embedding
and LM head are tied to the backbone's (wired by Transformer).
"""

import torch
import torch.nn.functional as F
from torch import nn

from .attention import Attention
from .layers import RMSNorm, apply_rotary_emb, init_std_
from .mhc import HCMixes
from .moe import Expert


def get_dspark_topk_idxs(window_size: int, bsz: int, block_size: int, start_pos: int) -> torch.Tensor:
    """Indices into the concatenated [window ring, own draft KV] axis
    (upstream model.py:1020-1029); ordering is irrelevant to softmax, only
    the set matters."""
    assert start_pos > 0, "draft runs on the decode path only (prefill seeds the ring)"
    matrix = torch.cat(
        [
            torch.arange(min(window_size, start_pos + 1)),
            window_size + torch.arange(block_size),
        ]
    )
    return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """Greedy at temperature 0; upstream gumbel-max otherwise (model.py:1285-1292)."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


def freeze_backbone(model) -> None:
    """P2 mode (contract T6 Lite): only DSpark blocks train. The tied
    embed/head, backbone, and vision stack get requires_grad=False."""
    dspark_ids = {id(p) for blk in (model.dspark or ()) for p in blk.parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in dspark_ids)


class DSparkAttention(Attention):
    """SWA-only attention whose ring is seeded from the projected main hidden,
    not from the block's own input (upstream DSparkAttention)."""

    def forward(self, x: torch.Tensor, start_pos: int, main_x: torch.Tensor) -> torch.Tensor:
        assert self.compress_ratio == 0
        win = self.window_size
        rd = self.rope_head_dim
        bsz, main_len, _ = main_x.size()

        main_kv = self.kv_norm(self.wkv(main_x))
        main_kv_tail = apply_rotary_emb(main_kv[..., -rd:], self.freqs_cis[start_pos : start_pos + main_len])
        main_kv = torch.cat([main_kv[..., :-rd], main_kv_tail], dim=-1)

        if start_pos == 0:
            if self.window_kv_cache is None:
                self.window_kv_cache = torch.zeros(bsz, win, self.head_dim, dtype=main_kv.dtype, device=main_kv.device)
            if main_len <= win:
                self.window_kv_cache[:bsz, :main_len] = main_kv.detach()
            else:
                cutoff = main_len % win
                self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = main_kv[:, -win:].detach().split(
                    [win - cutoff, cutoff], dim=1
                )
            return x  # prefill only seeds the ring (upstream model.py:1122-1126)

        block_size = x.size(1)
        # the main token occupies position start_pos; drafts sit at start_pos+1..
        draft_freqs = self.freqs_cis[start_pos + main_len : start_pos + main_len + block_size]

        q, _ = self.q_proj(x, start_pos + main_len)
        kv = self.kv_norm(self.wkv(x))
        kv_tail = apply_rotary_emb(kv[..., -rd:], draft_freqs)
        kv = torch.cat([kv[..., :-rd], kv_tail], dim=-1)

        self.window_kv_cache[:bsz, start_pos % win] = main_kv[:, -1].detach()
        ring_state = self.window_kv_cache[:bsz].clone()
        ring_state[:, start_pos % win] = main_kv[:, -1]
        kv_cat = torch.cat([ring_state, kv], dim=1)
        topk_idxs = get_dspark_topk_idxs(win, bsz, block_size, start_pos)
        return self.attend(q, kv_cat, topk_idxs, start_pos + main_len, block_size)


class DSparkMarkovHead(nn.Module):
    """markov_rank embedding + vocab head over the full vocab (contract T6)."""

    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dspark_markov_rank)
        self.head = nn.Linear(cfg.dspark_markov_rank, cfg.vocab_size, bias=False)
        init_std_(self.embed.weight)
        init_std_(self.head.weight)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embed(token_ids)
        logits = F.linear(embed.float(), self.head.weight.float())
        return logits, embed


class DSparkConfidenceHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Linear(cfg.d_model + cfg.dspark_markov_rank, 1)
        init_std_(self.proj.weight)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(hidden.float()).squeeze(-1)


class DSparkBlock(nn.Module):
    """One drafter block. Block 0 owns main_proj/main_norm (target-layer concat);
    the last block owns norm + markov head + confidence head."""

    def __init__(self, cfg, stage_id: int, n_stages: int, n_layers: int):
        super().__init__()
        self.stage_id = stage_id
        self.block_size = cfg.dspark_block_size
        self.noise_token_id = cfg.dspark_noise_token_id
        self.attn = DSparkAttention(cfg, layer_id=cfg.n_layers + stage_id, n_layers=n_layers)  # ratio 0 required
        assert self.attn.compress_ratio == 0, "DSpark blocks are SWA-only (ratio 0)"
        self.ffn = Expert(cfg.d_model, cfg.dspark_inter_dim, swiglu_limit=0, n_layers=n_layers)
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
        """x: [B,S,hc,d]; decode S == block_size. Prefill returns x unchanged."""
        if start_pos == 0:
            self.attn(x, start_pos, main_x)  # only main_x matters: seeds the ring
            return x, pre_mix
        residual = x
        a_pre, a_post, a_comb = self.hc_attn(x)
        x_c = HCMixes.hc_pre(x, pre_mix)
        x_c = self.attn_norm(x_c)
        x_c = self.attn(x_c, start_pos, main_x)
        h = HCMixes.hc_post(x_c, residual, a_post, a_comb)

        residual = h
        f_pre, f_post, f_comb = self.hc_ffn(h)
        x_c = HCMixes.hc_pre(h, a_pre)
        x_c = self.ffn_norm(x_c)
        x_c = self.ffn(x_c)
        return HCMixes.hc_post(x_c, residual, f_post, f_comb), f_pre

    def forward_embed(self, model, main_hidden: torch.Tensor, input_ids: torch.Tensor):
        """Target-layer hiddens [B, d*n_targets] -> (draft stream hc-expanded,
        projected main_x). input_ids: [B, 1] the committed token; draft slots
        are noise tokens with slot 0 replaced by it (upstream model.py:1128-1135)."""
        assert hasattr(self, "main_proj"), "forward_embed runs on stage 0"
        main_hidden = main_hidden.detach()  # P2: frozen backbone, no grads through the projection input
        main_x = self.main_norm(self.main_proj(main_hidden))
        draft_input_ids = input_ids.new_full([input_ids.size(0), self.block_size], self.noise_token_id)
        draft_input_ids[:, 0] = input_ids[:, 0]
        x = model.embed(draft_input_ids)
        x = x.unsqueeze(2).repeat(1, 1, model.cfg.hc_mult, 1)
        return x, main_x

    def forward_head(self, x, pre_mix, input_ids, model, temperature: float = 0.0):
        """Tied backbone head over the drafted hidden, plus per-position Markov
        bias and confidence scores (upstream model.py:1137-1156). Returns
        (output_ids [B, block+1], logits [B, block, vocab] fp32, confidence [B, block])."""
        assert hasattr(self, "norm"), "forward_head runs on the last stage"
        x = HCMixes.hc_pre(x, pre_mix)
        logits = F.linear(self.norm(x).float(), model.head_weight().float())
        # chain ids through a list, never in-place on one tensor: the markov
        # embed's backward saves its index tensor (in-place writes would
        # invalidate it under grad)
        ids = [input_ids[:, 0]]
        markov_embeds = []
        for i in range(self.block_size):
            logits_bias, markov_embed = self.markov_head(ids[i])
            logits[:, i].add_(logits_bias)
            markov_embeds.append(markov_embed)
            ids.append(sample(logits[:, i], temperature))
        markov_embed = torch.stack(markov_embeds, dim=1)
        confidence = self.confidence_head(x, markov_embed)
        return torch.stack(ids, dim=1), logits, confidence
