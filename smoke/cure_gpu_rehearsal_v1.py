#!/usr/bin/env python3
"""No-update full-loss CURE CUDA rehearsal at the planned per-rank batch 3."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import h5py
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/cure_official"))
from cure.data import sample_cure_batch  # noqa: E402
from cure.checkpoint import load_model  # noqa: E402
from cure.embeddings import PromptEncoder, align_intermediate_targets  # noqa: E402
from cure.losses import CURELoss  # noqa: E402
from cure.models import OneRestore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--embedder", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the memory rehearsal")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(124)
    randomizer = random.Random(124)
    with h5py.File(args.h5, "r") as handle:
        names = tuple(json.loads(handle.attrs["degradation_names"]))
        if len(handle) < 3:
            raise RuntimeError("HDF5 needs at least three real rows")
        fields = ("zero", "gt", "lr", "type1", "type2", "half")
        raw = {
            field: torch.from_numpy(__import__("numpy").stack([handle[str(i)][field][()] for i in range(3)])).to(device)
            for field in fields
        }
    encoder = PromptEncoder(args.embedder).to(device).eval().requires_grad_(False)
    restorer = OneRestore().to(device)
    load_model(restorer, args.baseline)
    objective = CURELoss(pretrained_vgg=True).to(device)
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
    output = restorer(batch["degraded"], full_embedding)
    identity_output = restorer(batch["identity_input"], identity_embedding)
    ratio_output = restorer(batch["degraded"], ratio_embedding)
    ratio_twice_output = restorer(ratio_output, ratio_embedding)
    partial_first = restorer(batch["degraded"], component_embeddings[:, 0])
    partial_second = restorer(batch["degraded"], component_embeddings[:, 1])
    sequential_first = restorer(partial_first, component_embeddings[:, 1])
    sequential_second = restorer(partial_second, component_embeddings[:, 0])
    loss, terms = objective(
        degraded=batch["degraded"], clean=batch["clean"], negatives=batch["negatives"],
        output=output, identity_input=batch["identity_input"], identity_output=identity_output,
        partial_outputs=torch.stack((partial_first, partial_second), dim=1),
        partial_targets=partial_targets,
        sequential_outputs=torch.stack((sequential_first, sequential_second), dim=1),
        ratio_output=ratio_output, ratio_target=batch["half"], ratio_twice_output=ratio_twice_output,
    )
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite CURE rehearsal loss")
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    trainable = [parameter for parameter in restorer.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in trainable):
        raise RuntimeError("missing or nonfinite restorer gradients")
    result = {
        "status": "no_update_cuda_rehearsal_pass",
        "loss": float(loss.item()),
        "loss_terms": {name: float(value.item()) for name, value in terms.items()},
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "device": torch.cuda.get_device_name(device),
        "microbatch": 3,
        "full_loss_forward_backward_seconds_including_first_cuda_use": elapsed,
        "optimizer_steps": 0,
        "official_foundir_test_accessed": False,
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
