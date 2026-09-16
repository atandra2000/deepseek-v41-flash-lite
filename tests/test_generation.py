"""Task 7 verify: incremental decode == full forward (logit parity),
measured cache accounting, generation bounds."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.cache import GenerationCache
from models.transformer import Transformer
from tests.toy import ced_toy_config, toy_config

# window 8 -> prompt wraps the ring; m=2 producer + decoder projection both exercised
PROMPT_LEN, CONT_LEN = 10, 8


def _run_pair(cfg, seed=7):
    """Teacher-forced decode on one seeded model vs full forward on an
    identically-seeded twin. outs[j] predicts token position prompt_len+j,
    i.e. pairs with full_logits[:, prompt_len-1+j]."""
    gen = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, cfg.vocab_size, (1, PROMPT_LEN + CONT_LEN), generator=gen)
    prompt, cont = tokens[:, :PROMPT_LEN], tokens[:, PROMPT_LEN:]

    torch.manual_seed(seed)
    m_dec = Transformer(cfg, max_seq_len=64)
    with torch.no_grad():
        pre_logits, _ = m_dec(prompt)
        outs = [pre_logits[:, -1:]]
        for k in range(CONT_LEN - 1):
            logits, _ = m_dec(cont[:, k : k + 1], start_pos=PROMPT_LEN + k)
            outs.append(logits)
        decode_logits = torch.cat(outs, dim=1)

    torch.manual_seed(seed)
    m_full = Transformer(cfg, max_seq_len=64)
    with torch.no_grad():
        full_logits, _ = m_full(tokens)
    return m_dec, decode_logits, full_logits[:, PROMPT_LEN - 1 : PROMPT_LEN - 1 + CONT_LEN]


@pytest.mark.parametrize("make_cfg", [toy_config, ced_toy_config], ids=["swa-toy", "ced-toy"])
def test_incremental_decode_parity(make_cfg):
    """Decode vs full forward with EXHAUSTIVE global selection (index_topk >=
    every compressed length): the attended set is identical on both paths, so
    logits must match to fp noise. This pins rings, rope positions, compressor
    state, decoder projection and producer-hidden plumbing end to end."""
    cfg = make_cfg(index_topk=64, candidate_topk_blocks=64)
    _, decode_logits, full_logits = _run_pair(cfg)
    diff = (decode_logits - full_logits).abs().max().item()
    assert diff < 1e-3, f"decode/full logit parity broken: max abs diff {diff:.3e}"


@pytest.mark.parametrize("make_cfg", [toy_config, ced_toy_config], ids=["swa-toy", "ced-toy"])
def test_greedy_token_parity_real_topk(make_cfg):
    """With the production top-k, greedy tokens from incremental decode must
    equal greedy tokens re-derived from full forwards. Logits themselves are
    only tie-near-equal: relu-zeroed indexer scores tie exactly and topk's
    pick among tied groups is shape-dependent (batched prefill vs 1-row
    decode), so one swapped group cascades (~1e-1 logit) through the next
    layer. Selection semantics are pinned by the Task-5 oracle, not here."""
    import scripts.gen as gen

    cfg = make_cfg()
    torch.manual_seed(11)
    model = Transformer(cfg, max_seq_len=64)
    prompt = torch.randint(0, cfg.vocab_size, (1, PROMPT_LEN), generator=torch.Generator().manual_seed(11))
    ids, _ = gen.generate(model, prompt, max_new_tokens=CONT_LEN)

    torch.manual_seed(11)
    twin = Transformer(cfg, max_seq_len=64)
    full_ids = prompt.clone()
    with torch.no_grad():
        for _ in range(CONT_LEN):
            logits, _ = twin(full_ids)
            full_ids = torch.cat([full_ids, logits[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
    assert torch.equal(ids[:, PROMPT_LEN:], full_ids[:, PROMPT_LEN:]), (
        f"greedy tokens diverge:\ndecode {ids[0, PROMPT_LEN:].tolist()}\nfull   {full_ids[0, PROMPT_LEN:].tolist()}"
    )


def test_cache_accounting_exact():
    """Measured bytes/token must match the first-principles counts: encoder
    m=2 cache + indexer K over complete groups only; decoder m=1 over all
    tokens. Storage widths: bf16 2 B/elem, fp8 1 B/elem."""
    cfg = ced_toy_config()
    m, _, _ = _run_pair(cfg)
    n = PROMPT_LEN + CONT_LEN
    cache = GenerationCache(m, context_len=n)
    # enc m=2: (n/2)*(head_dim + index_head_dim) elems; dec m=1: n*(head_dim + index_head_dim)
    enc = (n // 2) * (cfg.head_dim + cfg.index_head_dim)
    dec = n * (cfg.head_dim + cfg.index_head_dim)
    expected_bf16 = (enc + dec) * 2 / n
    assert cache.global_kv_bytes("bf16") == pytest.approx(expected_bf16)
    assert cache.global_kv_bytes("fp8") == pytest.approx(expected_bf16 / 2)
    # local: window ring, window_size x head_dim elems per layer, both layers
    assert cache.local_kv_bytes("bf16") == pytest.approx(cfg.n_layers * cfg.window_size * cfg.head_dim * 2)
    assert cache.pool_capacity == cfg.candidate_topk_blocks * cfg.candidate_block_size
    assert cache.compress_len(0) == n // 2


def test_generate_bounds():
    """generate() stops at max_new_tokens, and at EOS (first greedy token)."""
    import scripts.gen as gen

    cfg = toy_config()
    torch.manual_seed(3)
    model = Transformer(cfg, max_seq_len=64)
    prompt = torch.randint(0, cfg.vocab_size, (1, 6))
    ids, report = gen.generate(model, prompt, max_new_tokens=3)
    assert ids.shape == (1, 9) and report["context_len"] == 8
    assert report["global_kv_bytes_per_token_bf16"] > 0
    first = ids[:, 6]
    ids_eos, _ = gen.generate(model, prompt, max_new_tokens=5, eos_id=int(first))
    assert ids_eos.shape == (1, 7)
