"""Task 8 verify: ViT + aligner + image-span merge, DSpark draft loop,
backbone freeze."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.dspark import freeze_backbone
from models.transformer import Transformer
from models.vit import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START, ImageSpan, num_image_tokens
from tests.toy import toy_config

# 4x4 patch grid, ratio 2 -> 2x2 LLM grid -> 4 IMAGE slots, span len 8
N_VIT = 4
SPAN_AT = 3


def draft_toy_config(**over):
    """Toy with ViT and the 2-block DSpark drafter enabled."""
    base = dict(
        compress_ratios=(2, 0, 0, 0),  # backbone (2) + DSpark layers (2), all draft SWA-only
        vision_n_layers=2,
        vision_dim=64,
        vision_n_heads=4,
        vision_inter_dim=32,
        vision_patch_size=4,
        vision_downsample_ratio=2,
        num_nextn_predict_layers=2,
        dspark_block_size=5,
        dspark_noise_token_id=510,
        dspark_target_layer_ids=(1,),
        dspark_markov_rank=16,
        dspark_inter_dim=32,
    )
    base.update(over)
    return toy_config(**base)


def make_model(seed=7, **over):
    cfg = draft_toy_config(**over)
    torch.manual_seed(seed)
    return Transformer(cfg, max_seq_len=64), cfg


def make_span(seed=1):
    """A random image + its span layout inside a text sequence."""
    g = torch.Generator().manual_seed(seed)
    patches = torch.rand(N_VIT * N_VIT, 3, 4, 4, generator=g)
    n_llm_h = n_llm_w = N_VIT // 2
    types = torch.empty(num_image_tokens(n_llm_h, n_llm_w), dtype=torch.long)
    types[0], types[-1] = IMAGE_START, IMAGE_END
    inner = types[1:-1].view(n_llm_h, n_llm_w + 1)
    inner[:, :n_llm_w] = IMAGE
    inner[:, n_llm_w] = IMAGE_NEW_LINE
    return patches, n_llm_h, n_llm_w, types


def span_ids(cfg, length):
    ids = torch.randint(0, cfg.vocab_size - 2, (1, length))  # keep clear of image/noise ids
    ids[0, SPAN_AT : SPAN_AT + num_image_tokens(2, 2)] = cfg.image_token_id
    return ids


# ---- ViT + aligner ----

def test_vit_bidirectional_and_aligner_grid():
    model, cfg = make_model()
    patches, n_llm_h, n_llm_w, _ = make_span()
    with torch.no_grad():
        feats = model.vision(patches, N_VIT, N_VIT)
        assert feats.shape == (N_VIT * N_VIT, cfg.vision_dim)
        rows = model.aligner(feats, N_VIT, N_VIT)
    assert rows.shape == (n_llm_h * n_llm_w, cfg.d_model)

    # full (non-causal) attention: changing the LAST patch must change the FIRST token's output
    pert = patches.clone()
    pert[-1] += 1.0
    with torch.no_grad():
        feats2 = model.vision(pert, N_VIT, N_VIT)
    assert not torch.allclose(feats[0], feats2[0], atol=1e-5), "ViT attention looks causal"


def test_vision_rope_tables():
    """2D-RoPE table: per token the pair angles are [h*f, w*f] (upstream
    get_vision_cos_sin). 1x2 grid, dim 2, theta 1 -> inv_freq [1]."""
    import math

    from models.vit import _vision_cos_sin

    cos, sin = _vision_cos_sin(1, 2, 2, theta=1.0)
    expected_cos = torch.tensor([[[1.0, 1.0]], [[1.0, math.cos(1.0)]]])
    expected_sin = torch.tensor([[[0.0, 0.0]], [[0.0, math.sin(1.0)]]])
    assert torch.allclose(cos, expected_cos)
    assert torch.allclose(sin, expected_sin)


def test_merge_image_embeddings_exact():
    model, cfg = make_model()
    patches, _, _, types = make_span()
    images = [[ImageSpan(start=SPAN_AT, patches=patches, n_vit_h=N_VIT, n_vit_w=N_VIT, types=types)]]

    h = model.embed(span_ids(cfg, 20))
    before = h.clone()
    model.merge_image_embeddings(images, h)
    with torch.no_grad():
        embeds = model.encode_image(patches, N_VIT, N_VIT)
    span = h[0, SPAN_AT : SPAN_AT + types.numel()]
    # delimiters take the learned embeddings, IMAGE slots the aligner rows in reading order
    assert (span[types == IMAGE_START] == model.image_start).all()
    assert (span[types == IMAGE_END] == model.image_end).all()
    assert (span[types == IMAGE_NEW_LINE] == model.image_newline).all()
    assert torch.allclose(span[types == IMAGE], embeds, atol=1e-6)
    outside = h[0]
    mask = torch.ones(20, dtype=torch.bool)
    mask[SPAN_AT : SPAN_AT + types.numel()] = False
    assert torch.equal(outside[mask], before[0][mask]), "merge touched positions outside the span"


def test_interleaved_forward_parity_no_images():
    """The interleaved path with no images must be bit-identical to text-only."""
    model, cfg = make_model()
    ids = span_ids(cfg, 20)
    with torch.no_grad():
        base_logits, _ = model(ids)
        empty_logits, _ = model(ids, images=[])
        none_logits, _ = model(ids, images=[None])
    assert torch.equal(base_logits, empty_logits)
    assert torch.equal(base_logits, none_logits)

    # a real image changes the logits (embeddings differ) and runs the full stack
    patches, _, _, types = make_span()
    images = [[ImageSpan(start=SPAN_AT, patches=patches, n_vit_h=N_VIT, n_vit_w=N_VIT, types=types)]]
    image_mask = (ids == cfg.image_token_id)
    with torch.no_grad():
        img_logits, _ = model(ids, image_mask=image_mask, images=images)
    assert img_logits.shape == base_logits.shape
    assert not torch.allclose(img_logits, base_logits, atol=1e-4)


# ---- DSpark ----

def test_dspark_draft_mechanics():
    """Prefill seeds rings -> None; decode drafts block+1 ids chained by the
    returned (bias-included) logits, confidence shaped [B, block]."""
    model, cfg = make_model()
    tokens = torch.randint(0, cfg.vocab_size - 2, (1, 10))
    with torch.no_grad():
        logits, main_hiddens = model(tokens)
        assert main_hiddens is not None and len(main_hiddens) == len(cfg.dspark_target_layer_ids)
        assert model.forward_spec(tokens[:, -1:], torch.cat(main_hiddens, dim=-1), start_pos=0) is None
        for blk in model.dspark:
            assert blk.attn.window_kv_cache is not None, "prefill must seed the draft ring"

        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
        logits2, main2 = model(next_tok, start_pos=10)
        out = model.forward_spec(next_tok, torch.cat(main2, dim=-1), start_pos=10)
    assert out is not None
    output_ids, draft_logits, confidence = out
    assert output_ids.shape == (1, cfg.dspark_block_size + 1)
    assert draft_logits.shape == (1, cfg.dspark_block_size, cfg.vocab_size)
    assert confidence.shape == (1, cfg.dspark_block_size)
    assert torch.equal(output_ids[:, :1], next_tok)
    assert torch.isfinite(confidence).all()
    # greedy chain: every drafted id is argmax of the bias-included logits at its slot
    assert torch.equal(output_ids[:, 1:], draft_logits.argmax(dim=-1))
    # decode wrote the main token into each ring
    for blk in model.dspark:
        assert torch.isfinite(blk.attn.window_kv_cache).all()


def test_dspark_isolation_from_backbone():
    """Drafting must not disturb backbone caches: identical logits around a
    forward_spec call."""
    model, cfg = make_model()
    tokens = torch.randint(0, cfg.vocab_size - 2, (1, 10))
    with torch.no_grad():
        logits_a, _ = model(tokens)
        _, main_hiddens = model(tokens)
        model.forward_spec(tokens[:, -1:], torch.cat(main_hiddens, dim=-1), start_pos=0)
        logits_b, _ = model(tokens)
    assert torch.equal(logits_a, logits_b)


def test_backbone_freeze_no_grads():
    """P2 flag: after backward through the draft path, backbone params have no
    grads; every DSpark param does."""
    model, cfg = make_model()
    freeze_backbone(model)
    backbone_ids = {id(p) for p in model.parameters() if p.requires_grad is False}
    assert {id(p) for blk in model.dspark for p in blk.parameters()}.isdisjoint(backbone_ids)
    assert not model.embed.weight.requires_grad  # tied head frozen too

    tokens = torch.randint(0, cfg.vocab_size - 2, (1, 10))
    with torch.no_grad():
        _, main_hiddens = model(tokens)
        model.forward_spec(tokens[:, -1:], torch.cat(main_hiddens, dim=-1), start_pos=0)  # seed rings
        _, main2 = model(tokens[:, -1:], start_pos=10)
    out = model.forward_spec(tokens[:, -1:], torch.cat(main2, dim=-1), start_pos=10)
    output_ids, draft_logits, confidence = out
    (draft_logits.sum() + confidence.sum()).backward()

    for name, p in model.named_parameters():
        if name.startswith("dspark."):
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"dspark param untrained: {name}"
        else:
            assert p.grad is None, f"backbone param got a grad: {name}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
