#!/usr/bin/env python3
"""One-batch, exactly-one-step GPU save/reload gate for the HDD3 environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = ROOT.parent / ".envs/gpu-train-stable-hdd3"
OLD_ROOT = "/" + "/".join(("inspire", "hdd", "global_user", "liuxiaohong-25080", "zhenhaowang"))
CONFIG = ROOT / "configs/cure_two_track_reproduction_v1.json"
H5_PATH = ROOT / "data/third_party/cure_h5_smoke_v1/shared_cure.h5"
EMBEDDER = ROOT / "data/third_party/cure_released/OneRestore_embedder.tar"
RESTORER = ROOT / "data/third_party/cure_released/OneRestore_restorer.tar"
VGG = ROOT / "models/torch/hub/checkpoints/vgg16-397923af.pth"
LAUNCHER = ROOT / "smoke/run_migration_env_gpu_optimizer_1step.sh"

EXPECTED_HASHES = {
    CONFIG: "01a428b7eced044cc26e5a6a939010d23eade49627d6e776d19938acef823119",
    H5_PATH: "f394e7fd180b062eaa268d1f16f85b33cdb02a97e65fa9caacd8f3824539ac7d",
    EMBEDDER: "0a852fd2f0dadfb71f0d1970f2cba9d0ee3f851ccfd123941c3511a2d08ae8d2",
    RESTORER: "746365e9d25bc0861b2bc7d7bd63bbb9c978661bee2634fddb0b68def2c81b01",
    VGG: "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def add_cure_path() -> None:
    sys.path.insert(0, str(ROOT / "third_party/cure_official"))


def verify_environment(output: Path) -> dict[str, object]:
    executable = Path(sys.executable).resolve()
    expected = (EXPECTED_ENV / "bin/python").resolve()
    if executable != expected:
        raise RuntimeError(f"wrong Python: {executable} != {expected}")
    if Path(sys.prefix).resolve() != EXPECTED_ENV.resolve() or Path(sys.base_prefix).resolve() != EXPECTED_ENV.resolve():
        raise RuntimeError(f"wrong prefix: prefix={sys.prefix} base_prefix={sys.base_prefix}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"exactly one visible CUDA GPU is required; count={torch.cuda.device_count()}")
    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError(f"wrong working directory: {Path.cwd()}")
    active = list(sys.path) + [f"{key}={value}" for key, value in os.environ.items()]
    if any(OLD_ROOT in value for value in active):
        raise RuntimeError("old project path found in active process environment/sys.path")
    return {
        "pwd": str(Path.cwd()),
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "visible_gpu_count": torch.cuda.device_count(),
        "output": str(output),
    }


def load_real_batch(device: torch.device) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    with h5py.File(H5_PATH, "r") as handle:
        names = tuple(json.loads(handle.attrs["degradation_names"]))
        if len(handle) < 3:
            raise RuntimeError("real HDF5 needs at least three rows")
        fields = ("zero", "gt", "lr", "type1", "type2", "half")
        raw = {
            field: torch.from_numpy(np.stack([handle[str(index)][field][()] for index in range(3)])).to(device)
            for field in fields
        }
    return raw, names


def reload_only(checkpoint: Path, output: Path) -> None:
    add_cure_path()
    from cure.checkpoint import load_model
    from cure.models import OneRestore

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("epoch") != 1:
        raise RuntimeError("saved checkpoint epoch is not exactly one")
    saved = payload.get("state_dict")
    if not isinstance(saved, dict) or len(saved) != 189:
        raise RuntimeError(f"unexpected saved state_dict: {type(saved)!r}, keys={len(saved) if isinstance(saved, dict) else -1}")
    if not all(isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in saved.values()):
        raise RuntimeError("saved state_dict contains invalid/nonfinite values")
    model = OneRestore()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1.0e-6)
    epoch = load_model(model, checkpoint, optimizer=optimizer)
    loaded = model.state_dict()
    if epoch != 1 or loaded.keys() != saved.keys():
        raise RuntimeError("independent reload key/epoch mismatch")
    probe_key = sorted(saved)[0]
    if not torch.equal(loaded[probe_key], saved[probe_key]):
        raise RuntimeError(f"independent reload tensor mismatch: {probe_key}")
    result = {
        "status": "independent_checkpoint_reload_ok",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "epoch": epoch,
        "state_dict_key_count": len(saved),
        "probe_key": probe_key,
        "probe_tensor_sha256": tensor_digest(saved[probe_key]),
        "optimizer_state_entries": len(payload.get("optimizer", {}).get("state", {})),
        "sys_executable": sys.executable,
        "official_foundir_test_accessed": False,
    }
    (output / "reload_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("HDD3_INDEPENDENT_CHECKPOINT_RELOAD_OK", flush=True)


def train_one_step(output: Path) -> None:
    runtime = verify_environment(output)
    add_cure_path()
    from cure.checkpoint import load_model, save_model
    from cure.data import sample_cure_batch
    from cure.embeddings import PromptEncoder, align_intermediate_targets
    from cure.losses import CURELoss
    from cure.models import OneRestore

    config = json.loads(CONFIG.read_text())
    if "FoundIR official sealed test" not in config["dataset"]["prohibit"]:
        raise RuntimeError("protected-test prohibition is absent")
    resolved = {
        "config": str(CONFIG.resolve()),
        "dataset": str(H5_PATH.resolve()),
        "embedder": str(EMBEDDER.resolve()),
        "restorer": str(RESTORER.resolve()),
        "vgg": str(VGG.resolve()),
        "output": str(output.resolve()),
        "temporary_checkpoint": str((output / "temporary_step_000001.pt").resolve()),
    }
    for name, value in resolved.items():
        if name != "output" and name != "temporary_checkpoint" and not Path(value).is_file():
            raise FileNotFoundError(value)
    input_hashes = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"hash mismatch: {path} {actual} != {expected}")
        input_hashes[str(path)] = actual
    print(json.dumps({"runtime": runtime, "resolved_paths": resolved, "input_hashes": input_hashes}, indent=2), flush=True)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(124)
    torch.cuda.manual_seed_all(124)
    randomizer = random.Random(124)
    raw, names = load_real_batch(device)
    encoder = PromptEncoder(EMBEDDER).to(device).eval().requires_grad_(False)
    restorer = OneRestore().to(device)
    load_model(restorer, RESTORER)
    objective = CURELoss(pretrained_vgg=True).to(device)
    trainable = [parameter for parameter in restorer.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("restorer has no trainable parameters")
    optimizer = torch.optim.Adam(trainable, lr=1.0e-6)
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    batch = sample_cure_batch(raw, names)
    prompts = batch["prompts"]
    with torch.no_grad():
        full_embedding = encoder(prompts)
        identity_embedding = encoder.identity(len(prompts))
        ratio_embedding = encoder.ratio(prompts, 0.5)
        component_embeddings, _, swapped = encoder.components(prompts, randomize_order=True, rng=randomizer)
        partial_targets = align_intermediate_targets(batch["type1"], batch["type2"], swapped)
    model_output = restorer(batch["degraded"], full_embedding)
    identity_output = restorer(batch["identity_input"], identity_embedding)
    ratio_output = restorer(batch["degraded"], ratio_embedding)
    ratio_twice_output = restorer(ratio_output, ratio_embedding)
    partial_first = restorer(batch["degraded"], component_embeddings[:, 0])
    partial_second = restorer(batch["degraded"], component_embeddings[:, 1])
    sequential_first = restorer(partial_first, component_embeddings[:, 1])
    sequential_second = restorer(partial_second, component_embeddings[:, 0])
    loss, terms = objective(
        degraded=batch["degraded"], clean=batch["clean"], negatives=batch["negatives"],
        output=model_output, identity_input=batch["identity_input"], identity_output=identity_output,
        partial_outputs=torch.stack((partial_first, partial_second), dim=1),
        partial_targets=partial_targets,
        sequential_outputs=torch.stack((sequential_first, sequential_second), dim=1),
        ratio_output=ratio_output, ratio_target=batch["half"], ratio_twice_output=ratio_twice_output,
    )
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite loss")
    loss.backward()
    finite_gradients = [p for p in trainable if p.grad is not None and torch.isfinite(p.grad).all()]
    if len(finite_gradients) != len(trainable):
        raise RuntimeError(f"missing/nonfinite gradients: finite={len(finite_gradients)} trainable={len(trainable)}")
    nonzero = [p for p in finite_gradients if torch.count_nonzero(p.grad).item() > 0]
    if not nonzero:
        raise RuntimeError("all gradients are zero")
    probe = nonzero[0]
    before = probe.detach().clone()
    optimizer.step()
    optimizer_steps = 1
    if torch.equal(before, probe.detach()):
        raise RuntimeError("optimizer step did not change the gradient-bearing probe tensor")
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    checkpoint = output / "temporary_step_000001.pt"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    save_model(restorer, optimizer, 1, checkpoint)
    checkpoint_hash = sha256(checkpoint)

    reload_log = output / "reload_process.log"
    command = [sys.executable, str(Path(__file__).resolve()), "--reload-only", str(checkpoint), "--output-dir", str(output)]
    completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), text=True, capture_output=True, check=False)
    reload_log.write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0 or "HDD3_INDEPENDENT_CHECKPOINT_RELOAD_OK" not in completed.stdout:
        raise RuntimeError(f"independent checkpoint reload failed: status={completed.returncode}")

    scan_files = [CONFIG, Path(__file__).resolve(), LAUNCHER, output / "command.txt"]
    old_references = []
    for path in scan_files:
        text = path.read_text(errors="replace")
        if OLD_ROOT in text:
            old_references.append(str(path))
    active = list(sys.path) + [f"{key}={value}" for key, value in os.environ.items()] + list(resolved.values())
    if any(OLD_ROOT in value for value in active) or old_references:
        raise RuntimeError(f"old active path dependency found: files={old_references}")

    result = {
        "status": "HDD3_NEW_ENV_GPU_OPTIMIZER_SAVE_RELOAD_OK",
        **runtime,
        "resolved_paths": resolved,
        "input_hashes": input_hashes,
        "loss": float(loss.item()),
        "loss_terms": {name: float(value.item()) for name, value in terms.items()},
        "gradients_finite": True,
        "trainable_parameter_tensor_count": len(trainable),
        "nonzero_gradient_tensor_count": len(nonzero),
        "optimizer_steps": optimizer_steps,
        "real_batches": 1,
        "microbatch": 3,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "forward_backward_optimizer_seconds_including_first_cuda_use": elapsed,
        "checkpoint_sha256": checkpoint_hash,
        "independent_reload_process_status": completed.returncode,
        "old_active_dependency_count": 0,
        "official_foundir_test_accessed": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("HDD3_NEW_ENV_GPU_OPTIMIZER_SAVE_RELOAD_OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reload-only", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.reload_only is not None:
        reload_only(args.reload_only.resolve(), output)
    else:
        train_one_step(output)


if __name__ == "__main__":
    main()
