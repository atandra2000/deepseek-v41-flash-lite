"""A100 session runbook (Tasks 12/13): CPU-tested wiring for the GPU session.

This module is the boundary between CPU development and the rented A100
session. Everything here runs and is tested on CPU with fixture data; the
GPU steps themselves — bring-up, the 200-step bitwise repeat on CUDA, the
measured C0–C6 gates at production dims, and the production run — are
executed later and are never claimed by this code:

    1. pin          record the determinism environment (must be identical
                    across both bitwise-repeat sessions and the ladder)
    2. pin-probes   pin the exactly-512 sha-pinned probe batches (val split)
    3. repeat       200-step bitwise-repeat driver (ladder.bitwise_repeat)
    4. ladder       C0–C6 ladder via GateRunner + LadderRunner; approvals
                    are re-derived and compared against the recorded file
                    so config drift aborts before GPU hours are spent

Production data comes from the manifest-bound PackedDataset; `--fixture`
waives corpus size only, never integrity (same contract as
training.pretrain). Batch/accumulation sizes are explicit CLI arguments —
no production batch configuration is guessed here.
"""

import argparse
import hashlib
import json
import os
import tempfile
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from training.pretrain import seed_everything


def env_snapshot() -> dict:
    """Everything the bitwise-repeat contract depends on, as plain JSON."""
    cuda = torch.cuda.is_available()
    return {"cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "torch_version": str(torch.__version__),
            "device_type": "cuda" if cuda else "cpu",
            "cuda_device_count": torch.cuda.device_count() if cuda else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if cuda else None,
            "torch_version_cuda": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if cuda else None,
            "numpy_version": np.__version__,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED")}


def pinned_env(stage_id=0) -> dict:
    """Seed, then fail loudly unless the determinism environment is pinned.

    PYTHONHASHSEED can only be set by the launching shell; it is recorded
    here and must match across sessions (compared via the saved pin file).
    """
    seed_everything(stage_id)
    snapshot = env_snapshot()
    if (snapshot["cublas_workspace_config"] != ":4096:8"
            or not snapshot["deterministic"] or snapshot["matmul_tf32"]):
        raise RuntimeError("Determinism environment is not pinned; aborting before GPU spend")
    return snapshot


def env_matches(snapshot) -> bool:
    return snapshot == env_snapshot()
def write_json_atomic(record, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".runbook-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(record, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin(out_path, stage_id=0) -> dict:
    snapshot = pinned_env(stage_id)
    write_json_atomic(snapshot, out_path)
    return snapshot

def pin_probes(data_root, out_path, *, config=None, sequence_length=None, fixture=False) -> str:
    """Pin exactly 512 sha-pinned probe batches from the val split."""
    from data.dataset import PackedDataset

    from training.ladder import ProbeSet
    dataset = PackedDataset(data_root, split="val", sequence_length=sequence_length,
                            production=not fixture, config=config)
    try:
        batches = [{"input_ids": s["input_ids"], "labels": s["labels"]}
                   for s in islice(dataset, 512)]
    finally:
        dataset.close()
    return ProbeSet.pin(batches, out_path).verify()


def repeat(trainer_factory, out_path, env_path=None) -> dict:
    """200-step bitwise-repeat driver; refuses to run unless the determinism
    environment still matches the recorded pin (both sessions and the ladder)."""
    from training.ladder import bitwise_repeat

    if env_path is not None and not env_matches(json.loads(Path(env_path).read_text())):
        raise RuntimeError("Environment drifted from the recorded pin; aborting before GPU spend")
    evidence = bitwise_repeat(trainer_factory)
    evidence["env_sha256"] = None if env_path is None else sha256_file(env_path)
    write_json_atomic(evidence, out_path)
    return evidence


def ladder_run(data_root, checkpoint_dir, output, probes_path, *, config=None,
               approvals_path=None, device="cpu", fixture=False, batch_size=1,
               accumulation_steps=1, learning_rate=3e-4, warmup_steps=200,
               checkpoint_activations=False, probe_sequence_length=None):
    """C0–C6 ladder on the manifest-bound corpus (fixture waives size only).

    The val probes must be loaded at the same sequence length they were
    pinned with (`pin-probes`); production runs pin at the stage context.
    """
    from data.dataset import PackedDataset

    from models.config import load_config
    from training.ladder import GateRunner, LadderRunner, ProbeSet, gate_approvals

    cfg = load_config(config)
    approvals = gate_approvals(cfg)
    if approvals_path is not None:
        recorded = json.loads(Path(approvals_path).read_text())
        if recorded != approvals:
            raise ValueError("Approved config hashes drifted from the recorded approvals")
    dataset = PackedDataset(data_root, production=not fixture, config=config)
    val = PackedDataset(data_root, split="val", sequence_length=probe_sequence_length,
                        production=not fixture, config=config)
    probe_batches = [{"input_ids": s["input_ids"], "labels": s["labels"]} for s in islice(val, 512)]
    val.close()
    probes = None
    try:
        probes = ProbeSet.load(probe_batches, probes_path)
        runner = GateRunner(cfg, dataset, checkpoint_dir, device=device, batch_size=batch_size,
                            accumulation_steps=accumulation_steps, learning_rate=learning_rate,
                            warmup_steps=warmup_steps, checkpoint_activations=checkpoint_activations)
        return LadderRunner(probes, approvals, output).run(runner.run_stage)
    finally:
        dataset.close()
        if probes is not None:
            probes.verify()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("pin", help="Write the determinism environment pin")
    p.add_argument("--out", required=True)
    p.add_argument("--stage-id", type=int, default=0)

    p = commands.add_parser("pin-probes", help="Pin the 512-batch probe set from the val split")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-config")
    p.add_argument("--sequence-length", type=int)
    p.add_argument("--fixture", action="store_true")

    p = commands.add_parser("repeat", help="200-step bitwise-repeat driver")
    p.add_argument("--data", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--training-config", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--env", help="Pin file this session runs under (recorded, sha256)")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--fixture", action="store_true")

    p = commands.add_parser("ladder", help="C0–C6 gate ladder")
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--probes", required=True)
    p.add_argument("--output", required=True, help="runs/ladder.jsonl evidence path")
    p.add_argument("--model-config")
    p.add_argument("--approvals", help="Recorded gate_approvals JSON; drift aborts")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--probe-sequence-length", type=int, default=None,
                   help="Sequence length the probes were pinned with")

    args = parser.parse_args()
    if args.command == "pin":
        print(json.dumps(pin(args.out, args.stage_id), sort_keys=True))
    elif args.command == "pin-probes":
        print(json.dumps({"probe_sha256": pin_probes(args.data, args.out, config=args.model_config,
                                                     sequence_length=args.sequence_length,
                                                     fixture=args.fixture)}))
    elif args.command == "repeat":
        from data.dataset import PackedDataset

        from models.config import load_config
        from models.transformer import Transformer
        from training.ladder import SEED
        from training.pretrain import Trainer, TrainingConfig
        training_config = TrainingConfig(**json.loads(Path(args.training_config).read_text()))
        if training_config.max_steps < 200:
            raise ValueError("The bitwise-repeat contract requires max_steps >= 200")

        class Adapter:  # closing the trainer also closes the dataset
            def __init__(self, trainer, data):
                self._trainer, self._data = trainer, data

            def close(self):
                self._trainer.close()
                self._data.close()

            def __getattr__(self, name):
                return getattr(self._trainer, name)

        def factory():
            seed_everything(training_config.stage_id)
            torch.manual_seed(SEED)
            data = PackedDataset(args.data, production=not args.fixture, config=args.model_config)
            model = Transformer(load_config(args.model_config), max_seq_len=data.sequence_length)
            return Adapter(Trainer(model, data, training_config, args.checkpoint_dir, args.device), data)

        print(json.dumps(repeat(factory, args.out, args.env), sort_keys=True))
    else:
        checkpoint = ladder_run(args.data, args.checkpoint_dir, args.output, args.probes,
                                config=args.model_config, approvals_path=args.approvals,
                                device=args.device, fixture=args.fixture,
                                batch_size=args.batch_size, accumulation_steps=args.accumulation_steps,
                                learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
                                probe_sequence_length=args.probe_sequence_length)
        print(json.dumps({"final_checkpoint": str(checkpoint)}))


if __name__ == "__main__":
    main()

