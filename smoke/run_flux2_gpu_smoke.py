"""One-command, offline, end-to-end FLUX.2 Klein Base 4B smoke test."""
from pathlib import Path
import importlib.util
import os
import runpy
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    KLEIN_4B_BASE_MODEL_PATH=str(
        ROOT / "models/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors"
    ),
    AE_MODEL_PATH=str(ROOT / "models/flux2-klein-base-4b/ae.safetensors"),
    QWEN3_MODEL_PATH=str(ROOT / "models/qwen3-4b-fp8"),
)

# This exits immediately on any dependency, CUDA, model, or local-path problem.
runpy.run_path(str(ROOT / "smoke/flux2_gpu_preflight.py"), run_name="flux2_preflight")

cli_path = ROOT / "third_party/flux2/scripts/cli.py"
spec = importlib.util.spec_from_file_location("flux2_official_cli", cli_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"Unable to import {cli_path}")
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)

output_dir = ROOT / "output"
before = set(output_dir.glob("*.png")) if output_dir.exists() else set()
cli.main(
    model_name="flux.2-klein-base-4b",
    single_eval=True,
    prompt="A clean, well-lit indoor scene with a colorful vase on a wooden table",
    skip_safety_checks=True,
    width=512,
    height=512,
    num_steps=2,
    guidance=4.0,
    seed=1234,
    upsample_prompt_mode="none",
)
after = set(output_dir.glob("*.png"))
created = sorted(after - before)
if len(created) != 1 or created[0].stat().st_size == 0:
    raise SystemExit("SMOKE FAILED: inference did not create exactly one non-empty PNG")
with Image.open(created[0]) as result:
    result.verify()
with Image.open(created[0]) as result:
    if result.size != (512, 512) or result.mode != "RGB":
        raise SystemExit(f"SMOKE FAILED: bad output image {result.size} {result.mode}")
print(f"FLUX2 END-TO-END SMOKE PASS: {created[0]}")
