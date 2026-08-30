#!/usr/bin/env python3
"""Build a resume-safe disk cache of all FLUX Qwen3 restoration contexts."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.flux_prompt_cache import file_sha256, prompt_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import resolve_workspace_path  # noqa: E402


QWEN3_FILES = {
    "model-00001-of-00002.safetensors": "30b1bebf7b81f39f280b09c74f90feed00835a73145955a8d2fb69543b8371fb",
    "model-00002-of-00002.safetensors": "79f82dbc594a0237e24982800665eb0ddbe8c7bbdf3ac49c2a0005682aca15dc",
}


def load_prompts(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {"qwen_text", "type_oracle_text", "generic_text"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise RuntimeError("condition CSV has no complete restoration text interface")
    prompts = sorted({row[key] for row in rows for key in sorted(required)})
    if not prompts or any(not prompt for prompt in prompts):
        raise RuntimeError("condition CSV produced an empty prompt cache")
    return prompts


def cache_path(root: Path, prompt: str) -> Path:
    key = prompt_sha256(prompt)
    return root / key[:2] / f"{key}.safetensors"


def verify_entry(path: Path, prompt: str) -> dict[str, str]:
    sidecar = path.with_suffix(path.suffix + ".json")
    if path.is_file() and not sidecar.exists():
        tensors = load_file(str(path), device="cpu")
        if set(tensors) != {"context", "context_ids"}:
            raise RuntimeError(f"orphan prompt tensor schema changed: {path}")
        digest = file_sha256(path)
        content = (
            json.dumps(
                {"prompt_sha256": prompt_sha256(prompt), "context_sha256": digest},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        with sidecar.open("xb") as handle:
            handle.write(content)
    payload = json.loads(sidecar.read_text())
    if (
        payload.get("prompt_sha256") != prompt_sha256(prompt)
        or payload.get("context_sha256") != file_sha256(path)
    ):
        raise RuntimeError(f"prompt context identity changed: {path}")
    return {"path": path.relative_to(path.parents[1]).as_posix(), "sha256": payload["context_sha256"]}


def publish_context(path: Path, prompt: str, context: torch.Tensor, context_ids: torch.Tensor) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix(path.suffix + ".json").exists():
        return verify_entry(path, prompt)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        save_file(
            {"context": context.cpu().contiguous(), "context_ids": context_ids.cpu().contiguous()},
            str(temporary),
        )
        digest = file_sha256(temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = path.with_suffix(path.suffix + ".json")
    content = (
        json.dumps(
            {"prompt_sha256": prompt_sha256(prompt), "context_sha256": digest},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    with sidecar.open("xb") as handle:
        handle.write(content)
    return {"path": path.relative_to(path.parents[1]).as_posix(), "sha256": digest}


def encode_pending(config: Mapping[str, object], prompts: Sequence[str], root: Path, device: torch.device) -> None:
    from flux2.sampling import batched_prc_txt
    from flux2.util import load_text_encoder

    encoder = load_text_encoder(str(config["model"]["name"]), device=device).eval().requires_grad_(False)
    with torch.inference_mode():
        for offset in range(0, len(prompts), 4):
            chunk = list(prompts[offset : offset + 4])
            contexts = encoder(chunk).to(torch.bfloat16)
            contexts, ids = batched_prc_txt(contexts)
            for index, prompt in enumerate(chunk):
                publish_context(cache_path(root, prompt), prompt, contexts[index], ids[index])
            print(
                f"FOUNDIR_QWEN3_CONTEXT rank={dist.get_rank()} completed={offset+len(chunk)}/{len(prompts)}",
                flush=True,
            )
    del encoder
    gc.collect()
    torch.cuda.empty_cache()


def build_index(
    *, prompts: Sequence[str], root: Path, config_path: Path, conditions: Path
) -> dict[str, object]:
    entries = {
        prompt_sha256(prompt): verify_entry(cache_path(root, prompt), prompt)
        for prompt in prompts
    }
    payload = {
        "schema_version": 1,
        "status": "frozen_flux_qwen3_prompt_context_cache",
        "prompts": len(prompts),
        "entries": entries,
        "config_sha256": file_sha256(config_path),
        "conditions_sha256": file_sha256(conditions),
        "qwen3_weight_sha256": QWEN3_FILES,
        "protected_foundir_test_accessed": False,
    }
    path = root / "index.json"
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError("refusing to overwrite differing prompt-cache index")
    if not path.exists():
        with path.open("xb") as handle:
            handle.write(content)
    return payload


def verify_weights(config: Mapping[str, object]) -> None:
    root = ROOT / str(config["model"]["text_encoder"])
    for name, expected in QWEN3_FILES.items():
        if file_sha256(root / name) != expected:
            raise RuntimeError(f"FLUX Qwen3 weight changed: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    args.config = resolve_workspace_path(ROOT, args.config)
    args.conditions = resolve_workspace_path(ROOT, args.conditions)
    args.output = resolve_workspace_path(ROOT, args.output)
    config = json.loads(args.config.read_text())
    prompts = load_prompts(args.conditions)
    verify_weights(config)
    if args.verify_existing:
        payload = build_index(prompts=prompts, root=args.output, config_path=args.config, conditions=args.conditions)
        print("FOUNDIR_QWEN3_PROMPT_CACHE_VERIFY_PASS " + json.dumps(payload, sort_keys=True))
        return

    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        if world != int(config["distributed"]["required_world_size"]):
            raise RuntimeError("prompt cache requires exactly eight GPUs")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        properties = torch.cuda.get_device_properties(device)
        if properties.total_memory < int(config["distributed"]["minimum_memory_gib"]) * 1024**3:
            raise RuntimeError("prompt-cache GPU has insufficient memory")
        if not any(name in properties.name for name in ("H100", "H200")):
            raise RuntimeError(f"unsupported GPU: {properties.name}")
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            QWEN3_MODEL_PATH=str(ROOT / config["model"]["text_encoder"]),
        )
        assigned = [prompt for index, prompt in enumerate(prompts) if index % world == rank]
        pending = [prompt for prompt in assigned if not cache_path(args.output, prompt).is_file()]
        if pending:
            encode_pending(config, pending, args.output, device)
        for prompt in assigned:
            verify_entry(cache_path(args.output, prompt), prompt)
        dist.barrier()
        if rank == 0:
            payload = build_index(
                prompts=prompts, root=args.output, config_path=args.config, conditions=args.conditions
            )
            print("FOUNDIR_QWEN3_PROMPT_CACHE_PASS " + json.dumps(payload, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
