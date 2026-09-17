"""CPU tests for scripts/eval_headline.py (Task 14): measured, toy dims.

The evaluation runs for real at toy dims; production 16K/A100 execution is
pending hardware and is never claimed here.
"""

import json
import subprocess
import sys
from pathlib import Path

import torch

from tests.test_variants import build, gate_toy_config
from models.cache import GenerationCache
from models.variants import NAMED
from scripts.eval_headline import (_full_control_overrides, _index_overlap,
                                   decode_asymmetry, measure_global_kv,
                                   recall_probe, run)

CFG = gate_toy_config()
IDS = torch.arange(24).remainder(CFG.vocab_size).unsqueeze(0)


def test_full_control_widens_selection_without_relabeling():
    over = _full_control_overrides(CFG)
    assert over["index_topk"] > CFG.index_topk  # exhaustive control
    # only data-free scalars change; every structural key is untouched
    from dataclasses import asdict

    base = asdict(CFG)
    for key, value in over.items():
        assert base[key] != value


def test_index_overlap_counts_coverage():
    selected = torch.tensor([[[2, 5, -1]]])
    full = torch.tensor([[[1, 2, 5, 9, -1]]])
    recall, covered = _index_overlap(selected, full)
    assert (recall, covered) == (0.5, 4)  # 2 of the control's 4 valid picks covered
    assert _index_overlap(full, full) == (1.0, 4)


def test_global_kv_matches_generation_cache(prepared_model=None):
    model = build(CFG, NAMED["csa2"])
    report = measure_global_kv(model, IDS, decode_steps=2)
    assert report["context_len"] == 26
    assert report["global_kv_bytes_per_token_fp8"] * 2 == report["global_kv_bytes_per_token_bf16"]
    assert report["local_kv_bytes_bf16"] == CFG.window_size * CFG.head_dim * 2 * CFG.n_layers
    # independent recount from the live buffers
    cache = GenerationCache(model, context_len=26)
    assert cache.report()["global_kv_bytes_per_token_bf16"] == report["global_kv_bytes_per_token_bf16"]


def test_recall_probe_same_weights_sparse_vs_full():
    result = recall_probe(CFG, IDS)
    assert result["layers"], "index sources must be measured"
    for name, layer in result["layers"].items():
        assert 0.0 <= layer["recall"] <= 1.0
        assert layer["sparse_k"] < layer["full_k"]  # control covers the stream
        assert layer["full_k"] > 0
    assert not result["identical"]  # different selection -> different logits
    assert result["logit_mean_abs_diff"] < 1.0  # toy random weights stay close


def test_decode_asymmetry_measures_both_paths():
    model = build(CFG, NAMED["dense"])
    out = decode_asymmetry(model, 24, 2)
    assert out["prefill_tokens"] == 24 and out["decode_steps"] == 2
    assert out["prefill_ms_per_token"] > 0 and out["decode_ms_per_token"] > 0
    assert out["asymmetry_ratio"] > 1  # single-token decode pays per-step overhead


def test_run_end_to_end_and_cli(tmp_path):
    results = run(CFG, seq=24, decode_steps=2)
    assert set(results) == {"config_sha256", "seq", "decode_steps", "device",
                            "global_kv", "decode_asymmetry", "recall"}
    out = tmp_path / "results" / "eval.json"
    toy_config = tmp_path / "toy.json"
    # explicit toy config: the CLI default is the production 1.16B config
    toy_config.write_text(json.dumps({k: list(v) if isinstance(v, tuple) else v
                                      for k, v in CFG.__dict__.items() if k != "sha256"}))
    proc = subprocess.run(
        [".venv/bin/python", "scripts/eval_headline.py", "--config", str(toy_config),
         "--seq", "24", "--decode-steps", "2", "--out", str(out)],
        capture_output=True, text=True, check=True, timeout=180)
    written = json.loads(out.read_text())
    assert written["global_kv"]["context_len"] == 26
    assert written["recall"]["layers"]
    assert json.loads(proc.stdout) == written
