from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from lowlight_restoration.condition_adapter import LowLightLevelContext
from lowlight_restoration.conditioned_authorization import (
    CONDITIONED_CONFIG_SHA256,
    QWEN_CHECKPOINT_SHA256,
    QWEN_CONFIG_SHA256,
    QWEN_METRICS_SHA256,
    QWEN_MODEL_REVISION,
    validate_conditioned_authorization,
)
from lowlight_restoration.conditioned_checkpoint import (
    ADAPTER_KEY,
    conditioned_state_dict,
    load_conditioned_checkpoint,
    save_conditioned_checkpoint,
    trainable_parameters,
)
from lowlight_restoration.config import ROOT, TrainConfig, load_train_config
from lowlight_restoration.data import PairRecord
from lowlight_restoration.degradation_assessment import canonical_condition_json
from lowlight_restoration.lora import inject_lora
from lowlight_restoration.qwen_predictions import (
    load_frozen_ridge,
    load_prediction_manifest,
    prediction_bundle_sha256,
    validate_prediction_record,
)
from smoke.compare_lowlight_baseline_qwen import compare_split
from smoke.train_flux2_lowlight_qwen_conditioned import save_training_checkpoint
from train_flux2_lowlight import load_checkpoint_record, sha256


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Module()
        self.block.qkv = nn.Linear(4, 12, bias=False)
        self.block.proj = nn.Linear(12, 4, bias=False)


class ConditionedRestorationTests(unittest.TestCase):
    def test_frozen_config_is_authorized_and_fair(self) -> None:
        config = load_train_config(
            ROOT / "configs/lowlight_flux2_qwen_conditioned_full.json"
        )
        self.assertEqual(config.sha256, CONDITIONED_CONFIG_SHA256)
        self.assertEqual(
            validate_conditioned_authorization(config, require_artifacts=False),
            "full_qwen_conditioned",
        )
        baseline = load_train_config(ROOT / "configs/lowlight_flux2_full.json")
        for section in ("model", "data", "lora", "training", "inference"):
            self.assertEqual(config.section(section), baseline.section(section))

    def test_changed_conditioned_config_is_not_authorized(self) -> None:
        source = ROOT / "configs/lowlight_flux2_qwen_conditioned_full.json"
        payload = json.loads(source.read_text())
        payload["training"]["max_steps"] += 1
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "changed.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "config hash has changed"):
                validate_conditioned_authorization(
                    load_train_config(path), require_artifacts=False
                )

    def test_frozen_ridge_checkpoint_is_finite_and_bound(self) -> None:
        ridge = load_frozen_ridge(
            ROOT
            / "checkpoints/qwen_lowlight_ridge_expanded_head/ridge_head.safetensors"
        )
        self.assertEqual(ridge["coefficient"].shape, (7168,))
        self.assertEqual(ridge["feature_mean"].shape, (7168,))
        self.assertEqual(ridge["feature_scale"].shape, (7168,))
        self.assertIsInstance(ridge["intercept"], float)

    def test_prediction_record_is_label_free_and_fail_closed(self) -> None:
        pair = PairRecord(
            pair_id="sample_L4",
            source_id="sample",
            clean_path=Path("clean.png"),
            degraded_path=Path("degraded.png"),
            project_level="L4",
            seed=1,
            split="test",
            clean_sha256="a" * 64,
            degraded_sha256="b" * 64,
        )
        payload = {
            "schema_version": 1,
            "index": 0,
            "pair_id": pair.pair_id,
            "source_id": pair.source_id,
            "split": pair.split,
            "degraded_sha256": pair.degraded_sha256,
            "predicted_level": 3,
            "raw_score": 3.2,
            "raw_output": canonical_condition_json("low_light", 3),
            "conditioned_config_sha256": CONDITIONED_CONFIG_SHA256,
            "qwen_config_sha256": QWEN_CONFIG_SHA256,
            "qwen_metrics_sha256": QWEN_METRICS_SHA256,
            "qwen_checkpoint_sha256": QWEN_CHECKPOINT_SHA256,
            "qwen_model_revision": QWEN_MODEL_REVISION,
        }
        validate_prediction_record(payload, pair, index=0)
        self.assertEqual(
            prediction_bundle_sha256([payload]),
            prediction_bundle_sha256([copy.deepcopy(payload)]),
        )
        leaked = dict(payload, true_level=4)
        with self.assertRaisesRegex(RuntimeError, "leaks the target label"):
            validate_prediction_record(leaked, pair, index=0)
        changed = dict(payload, degraded_sha256="c" * 64)
        with self.assertRaisesRegex(RuntimeError, "degraded_sha256"):
            validate_prediction_record(changed, pair, index=0)

    def test_combined_checkpoint_round_trip_and_trainable_scope(self) -> None:
        torch.manual_seed(13)
        model = TinyModel()
        inject_lora(model, ["block.*"], rank=2, alpha=2)
        adapter = LowLightLevelContext(context_dim=8)
        parameters = trainable_parameters(model, adapter)
        self.assertEqual(sum(parameter.numel() for parameter in parameters), 104)
        with torch.no_grad():
            for parameter in parameters:
                parameter.normal_()
        expected = {
            key: value.clone() for key, value in conditioned_state_dict(model, adapter).items()
        }
        self.assertIn(ADAPTER_KEY, expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conditioned.safetensors"
            save_conditioned_checkpoint(model, adapter, path, {"kind": "test"})
            with torch.no_grad():
                for parameter in parameters:
                    parameter.zero_()
            load_conditioned_checkpoint(model, adapter, path)
        realized = conditioned_state_dict(model, adapter)
        self.assertEqual(set(realized), set(expected))
        for key in expected:
            torch.testing.assert_close(realized[key], expected[key], rtol=0, atol=0)

    def test_prediction_manifest_validates_sidecars_and_all_levels(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output_dir = Path(directory)
            manifest = output_dir / "predictions.json"
            pairs = [
                PairRecord(
                    pair_id=f"source_L{level}",
                    source_id=f"source_{level}",
                    clean_path=Path("clean.png"),
                    degraded_path=Path("degraded.png"),
                    project_level=f"L{level}",
                    seed=level,
                    split="train",
                    clean_sha256="a" * 64,
                    degraded_sha256=f"{level:064x}",
                )
                for level in range(1, 6)
            ]
            predictions = []
            for index, pair in enumerate(pairs):
                payload = {
                    "schema_version": 1,
                    "index": index,
                    "pair_id": pair.pair_id,
                    "source_id": pair.source_id,
                    "split": pair.split,
                    "degraded_sha256": pair.degraded_sha256,
                    "predicted_level": index + 1,
                    "raw_score": float(index + 1),
                    "raw_output": canonical_condition_json("low_light", index + 1),
                    "conditioned_config_sha256": CONDITIONED_CONFIG_SHA256,
                    "qwen_config_sha256": QWEN_CONFIG_SHA256,
                    "qwen_metrics_sha256": QWEN_METRICS_SHA256,
                    "qwen_checkpoint_sha256": QWEN_CHECKPOINT_SHA256,
                    "qwen_model_revision": QWEN_MODEL_REVISION,
                }
                predictions.append(payload)
                sidecar = output_dir / "records" / f"{pair.pair_id}.json"
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.write_text(json.dumps(payload))
            summary = {
                "schema_version": 1,
                "status": "qwen_lowlight_condition_predictions_complete",
                "conditioned_config_sha256": CONDITIONED_CONFIG_SHA256,
                "qwen_config_sha256": QWEN_CONFIG_SHA256,
                "qwen_metrics_sha256": QWEN_METRICS_SHA256,
                "qwen_checkpoint_sha256": QWEN_CHECKPOINT_SHA256,
                "qwen_model_revision": QWEN_MODEL_REVISION,
                "records": 5,
                "split_counts": {"train": 5, "validation": 0, "test": 0},
                "target_labels_exported": False,
                "test_metrics_computed": False,
                "official_foundir_used": False,
                "prediction_bundle_sha256": prediction_bundle_sha256(predictions),
                "predictions": predictions,
            }
            manifest.write_text(json.dumps(summary))
            raw = {
                "condition": {
                    "prediction_manifest": str(manifest.relative_to(ROOT)),
                    "prediction_output_dir": str(output_dir.relative_to(ROOT)),
                    "expected_prediction_records": 5,
                    "expected_prediction_splits": {
                        "train": 5,
                        "validation": 0,
                        "test": 0,
                    },
                }
            }
            config_path = output_dir / "config.json"
            config_path.write_text(json.dumps(raw))
            mapping, realized, digest = load_prediction_manifest(
                TrainConfig(config_path, raw), pairs, verify_sidecars=True
            )
            self.assertEqual(mapping, {pair.pair_id: index + 1 for index, pair in enumerate(pairs)})
            self.assertEqual(realized["prediction_bundle_sha256"], summary["prediction_bundle_sha256"])
            self.assertEqual(len(digest), 64)

    def test_published_conditioned_checkpoint_binds_optimizer_and_prediction(self) -> None:
        torch.manual_seed(29)
        model = TinyModel()
        inject_lora(model, ["block.*"], rank=2, alpha=2)
        adapter = LowLightLevelContext(context_dim=8)
        parameters = trainable_parameters(model, adapter)
        optimizer = torch.optim.AdamW(parameters, lr=1e-4)
        prediction_sha = "d" * 64
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output_dir = Path(directory)
            config_path = output_dir / "source_config.json"
            raw = {
                "run_name": "conditioned_checkpoint_test",
                "model": {"name": "tiny"},
            }
            config_path.write_text(json.dumps(raw))
            config = TrainConfig(config_path, raw)
            checkpoint = save_training_checkpoint(
                model,
                adapter,
                optimizer,
                output_dir,
                1,
                config,
                {"step": 1, "loss": 0.5, "probe_loss": 0.4},
                prediction_sha,
            )
            record = load_checkpoint_record(checkpoint, config)
            self.assertEqual(record["prediction_manifest_sha256"], prediction_sha)
            self.assertEqual(record["qwen_checkpoint_sha256"], QWEN_CHECKPOINT_SHA256)
            self.assertEqual(record["checkpoint_sha256"], sha256(checkpoint))

    def test_paired_comparison_uses_identical_inputs_and_directional_deltas(self) -> None:
        baseline_records = []
        conditioned_records = []
        for index in range(1000):
            level = index % 5 + 1
            common = {
                "index": index,
                "pair_id": f"pair_{index:04d}",
                "project_level": f"L{level}",
                "split": "test",
                "crop_xy": [index % 7, index % 11],
                "seed": 1000 + index,
                "degraded_psnr": 10.0 + level,
                "degraded_ssim": 0.4,
                "degraded_lpips": 0.3,
            }
            baseline_records.append(
                {
                    **common,
                    "restored_psnr": 20.0,
                    "restored_ssim": 0.80,
                    "restored_lpips": 0.10,
                    "restored_path": f"baseline_{index}.png",
                    "restored_sha256": "a" * 64,
                }
            )
            conditioned_records.append(
                {
                    **common,
                    "predicted_level": level,
                    "condition_correct": True,
                    "restored_psnr": 20.5,
                    "restored_ssim": 0.81,
                    "restored_lpips": 0.09,
                    "restored_path": f"conditioned_{index}.png",
                    "restored_sha256": "b" * 64,
                }
            )
        summary, pairs = compare_split(
            {
                "split": "test",
                "samples": 1000,
                "guidance": 1.0,
                "per_image": baseline_records,
            },
            {
                "split": "test",
                "samples": 1000,
                "guidance": 1.0,
                "per_image": conditioned_records,
            },
            "test",
        )
        self.assertEqual(len(pairs), 1000)
        self.assertAlmostEqual(summary["l2_l5"]["mean_psnr_delta"], 0.5)
        self.assertAlmostEqual(summary["l2_l5"]["mean_ssim_delta"], 0.01)
        self.assertAlmostEqual(summary["l2_l5"]["mean_lpips_delta"], -0.01)
        conditioned_records[0]["degraded_psnr"] = 99.0
        with self.assertRaisesRegex(RuntimeError, "input metric differs"):
            compare_split(
                {
                    "split": "test",
                    "samples": 1000,
                    "guidance": 1.0,
                    "per_image": baseline_records,
                },
                {
                    "split": "test",
                    "samples": 1000,
                    "guidance": 1.0,
                    "per_image": conditioned_records,
                },
                "test",
            )


if __name__ == "__main__":
    unittest.main()
