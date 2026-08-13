"""Fail-fast validation for the offline FLUX.2 Klein 4B GPU smoke test."""
import importlib
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
expected_prefix = ROOT / ".venv/flux2-smoke"
if Path(sys.prefix).resolve() != expected_prefix.resolve():
    raise SystemExit(
        f"WRONG_PYTHON: expected environment {expected_prefix}, got {sys.prefix}"
    )
flux_repo = ROOT / "third_party/flux2"
expected_flux_revision = "50fe5162777813d869182b139e83b10743caef15"
actual_flux_revision = subprocess.run(
    ["git", "-C", str(flux_repo), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if actual_flux_revision != expected_flux_revision:
    raise SystemExit(f"BAD_FLUX2_REVISION {actual_flux_revision}")
patch_check = subprocess.run(
    ["git", "-C", str(flux_repo), "apply", "--reverse", "--check", str(ROOT / "patches/flux2-offline-smoke.patch")],
    capture_output=True,
    text=True,
)
if patch_check.returncode != 0:
    raise SystemExit("FLUX2_OFFLINE_PATCH_NOT_EXACTLY_APPLIED")
checks = [
    "torch", "torchvision", "einops", "transformers", "safetensors",
    "fire", "openai", "accelerate", "PIL", "flux2",
]
for name in checks:
    try:
        importlib.import_module(name)
    except Exception as exc:
        raise SystemExit(f"MISSING_OR_BROKEN_IMPORT {name}: {exc}") from exc

import torch
from safetensors import safe_open

if shutil.disk_usage(ROOT).free < 5 * 1024**3:
    raise SystemExit("Workspace has less than 5 GiB free disk space")

files = {
    ROOT / "models/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors": (
        7_751_105_712,
        "9c5fed22b76baea749d88fc2abe3ad53245e7b21a0d353a762665eea00043b92",
    ),
    ROOT / "models/flux2-klein-base-4b/ae.safetensors": (
        336_211_292,
        "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3",
    ),
    ROOT / "models/qwen3-4b-fp8/model-00001-of-00002.safetensors": (
        4_412_140_424,
        "30b1bebf7b81f39f280b09c74f90feed00835a73145955a8d2fb69543b8371fb",
    ),
    ROOT / "models/qwen3-4b-fp8/model-00002-of-00002.safetensors": (
        777_912_448,
        "79f82dbc594a0237e24982800665eb0ddbe8c7bbdf3ac49c2a0005682aca15dc",
    ),
}
for path, (expected_size, expected_hash) in files.items():
    if not path.is_file() or path.stat().st_size != expected_size:
        raise SystemExit(f"BAD_FILE {path}: {path.stat().st_size if path.exists() else 'missing'}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_hash:
        raise SystemExit(f"BAD_SHA256 {path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        print(f"OK {path.name}: {len(handle.keys())} tensors")

index = ROOT / "models/qwen3-4b-fp8/model.safetensors.index.json"
data = json.loads(index.read_text())
parts = set(data["weight_map"].values())
if parts != {p.name for p in files if "qwen3" in str(p)}:
    raise SystemExit(f"BAD_QWEN_INDEX parts={parts}")

os.environ["KLEIN_4B_BASE_MODEL_PATH"] = str(ROOT / "models/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors")
os.environ["AE_MODEL_PATH"] = str(ROOT / "models/flux2-klein-base-4b/ae.safetensors")
os.environ["QWEN3_MODEL_PATH"] = str(ROOT / "models/qwen3-4b-fp8")
from flux2.openrouter_api_client import OpenRouterAPIClient  # noqa: F401
from flux2.text_encoder import load_qwen3_embedder
if os.environ.get("QWEN3_MODEL_PATH") != str(ROOT / "models/qwen3-4b-fp8"):
    raise SystemExit("QWEN3_MODEL_PATH does not point to the local snapshot")
if "QWEN3_MODEL_PATH" not in inspect.getsource(load_qwen3_embedder):
    raise SystemExit("Local Qwen path support is missing from flux2.text_encoder")

cli_path = ROOT / "third_party/flux2/scripts/cli.py"
cli_spec = importlib.util.spec_from_file_location("flux2_preflight_cli", cli_path)
if cli_spec is None or cli_spec.loader is None:
    raise SystemExit("Unable to load the FLUX.2 CLI")
cli_module = importlib.util.module_from_spec(cli_spec)
sys.modules[cli_spec.name] = cli_module
cli_spec.loader.exec_module(cli_module)
if not hasattr(cli_module, "main"):
    raise SystemExit("FLUX.2 CLI main() is missing")
if "skip_safety_checks" not in inspect.signature(cli_module.main).parameters:
    raise SystemExit("FLUX.2 CLI offline smoke option is missing")

import einops
import transformers
import safetensors
import accelerate
import openai
import numpy
from transformers import AutoConfig, AutoTokenizer

actual_versions = {
    "einops": einops.__version__,
    "transformers": transformers.__version__,
    "safetensors": safetensors.__version__,
    "accelerate": accelerate.__version__,
    "openai": openai.__version__,
    "numpy": numpy.__version__,
    "torchvision": importlib.import_module("torchvision").__version__,
}
expected_versions = {
    "einops": "0.8.1",
    "transformers": "4.56.1",
    "safetensors": "0.4.5",
    "accelerate": "1.12.0",
    "openai": "2.8.1",
    "numpy": "1.26.4",
    "torchvision": "0.22.0",
}
for name, expected in expected_versions.items():
    actual = actual_versions[name]
    # CUDA wheel builds append a local tag such as +cu128. The public package
    # version is still 0.22.0 and the CUDA build tag is validated with PyTorch.
    comparable_actual = actual.split("+", 1)[0] if name == "torchvision" else actual
    if comparable_actual != expected:
        raise SystemExit(f"BAD_VERSION {name}: expected {expected}, got {actual}")
if not torch.__version__.startswith("2.7.0a0+"):
    raise SystemExit(f"BAD_VERSION torch: expected validated NVIDIA 2.7.0a0 build, got {torch.__version__}")

qwen_path = str(ROOT / "models/qwen3-4b-fp8")
qwen_config = AutoConfig.from_pretrained(qwen_path, local_files_only=True)
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
if qwen_config.model_type != "qwen3" or qwen_config.architectures != ["Qwen3ForCausalLM"]:
    raise SystemExit("BAD_QWEN_CONFIG")
if len(qwen_tokenizer) != 151_669:
    raise SystemExit(f"BAD_QWEN_TOKENIZER size={len(qwen_tokenizer)}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
if torch.cuda.get_device_properties(0).total_memory < 40 * 1024**3:
    raise SystemExit("GPU has less than 40 GiB VRAM")

print(f"CUDA OK: {torch.cuda.get_device_name(0)}")
print("FLUX2 GPU PREFLIGHT PASS")
