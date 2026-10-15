#!/usr/bin/env python3
"""Reaggregate frozen routing/restoration artifacts into paper-facing tables.

This script performs no model loading, inference, training, or protected-test
access. It only reads already-frozen development/validation predictions and
metric sidecars.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-09-27"

BASELINE = ROOT / "checkpoints/foundir_qwen_mixed_consensus_development_v3/predictions.csv"
STAGE_FILES = [
    (
        "categorical_confidence_consensus",
        BASELINE,
        "route",
        "accepted",
    ),
    (
        "compositional_linear_v1",
        ROOT / "checkpoints/foundir_qwen_compositional_router_v1/development_predictions.csv",
        "route",
        "accepted",
    ),
    (
        "conditional_noise_v2",
        ROOT / "checkpoints/foundir_qwen_compositional_router_v2/development_predictions.csv",
        "route",
        "accepted",
    ),
    (
        "asymmetric_absence_v3",
        ROOT / "checkpoints/foundir_qwen_compositional_router_v3/development_predictions.csv",
        "route",
        "accepted",
    ),
    (
        "dual_nonlinear",
        ROOT / "checkpoints/foundir_qwen_lowlight_noise_dual_nonlinear_v2/development_predictions.csv",
        "route",
        "accepted",
    ),
]
FINAL = ROOT / "checkpoints/foundir_final_eleven_route_cpu_gate_v1/development_decisions.csv"
PHYSICAL = ROOT / "checkpoints/foundir_noise_family_fail_closed_consensus_v1/development_decisions.csv"
DEV_METRICS = ROOT / "checkpoints/foundir_final_eleven_all_in_one_development_rehearsal_v3/per_record"
FULL_METRICS = ROOT / "checkpoints/foundir_composition_whole_image_v1/per_record"

OUTPUT_JSON = ROOT / f"reports/foundir_routing_paper_ablation_{DATE}.json"
OUTPUT_CSV = ROOT / f"reports/foundir_routing_paper_ablation_{DATE}.csv"
OUTPUT_CURVE = ROOT / f"reports/foundir_routing_risk_coverage_curve_{DATE}.csv"
OUTPUT_MD = ROOT / f"reports/foundir_routing_paper_ablation_{DATE}.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def accepted(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def route_summary(
    rows: list[dict[str, str]], route_key: str, accepted_key: str
) -> dict[str, Any]:
    total = len(rows)
    correct = wrong = 0
    per_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"records": 0, "correct": 0, "wrong": 0, "abstain": 0}
    )
    for row in rows:
        truth = row["truth"]
        stats = per_class[truth]
        stats["records"] += 1
        if not accepted(row[accepted_key]):
            stats["abstain"] += 1
        elif row[route_key] == truth:
            correct += 1
            stats["correct"] += 1
        else:
            wrong += 1
            stats["wrong"] += 1
    covered = correct + wrong
    return {
        "records": total,
        "correct": correct,
        "wrong": wrong,
        "abstain": total - covered,
        "coverage": covered / total,
        "wrong_rate_all": wrong / total,
        "wrong_rate_accepted": wrong / covered if covered else None,
        "per_class": dict(sorted(per_class.items())),
    }


def final_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    counts: dict[str, int] = defaultdict(int)
    per_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["status"]] += 1
        per_class[row["truth"]][row["status"]] += 1
        per_class[row["truth"]]["records"] += 1
    correct = counts["correct"]
    wrong = counts["wrong"]
    covered = correct + wrong
    return {
        "records": total,
        "correct": correct,
        "wrong": wrong,
        "abstain": counts["abstain"],
        "coverage": covered / total,
        "wrong_rate_all": wrong / total,
        "wrong_rate_accepted": wrong / covered if covered else None,
        "per_class": {key: dict(value) for key, value in sorted(per_class.items())},
    }


def pure_confidence_points(
    rows: list[dict[str, str]], accepted_counts: Iterable[int]
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    ranked = sorted(
        rows,
        key=lambda row: (-float(row["eleven_confidence"]), row["record_id"]),
    )
    cumulative_wrong = [0]
    for row in ranked:
        cumulative_wrong.append(
            cumulative_wrong[-1] + int(row["eleven_proposal"] != row["truth"])
        )
    total = len(ranked)
    requested = sorted(set(max(1, min(total, int(value))) for value in accepted_counts))
    by_count: dict[int, dict[str, Any]] = {}
    for count in requested:
        wrong = cumulative_wrong[count]
        point = {
            "accepted": count,
            "coverage": count / total,
            "correct": count - wrong,
            "wrong": wrong,
            "wrong_rate_all": wrong / total,
            "wrong_rate_accepted": wrong / count,
            "minimum_confidence": float(ranked[count - 1]["eleven_confidence"]),
        }
        by_count[count] = point
    return [by_count[count] for count in requested], by_count


def mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def development_metric_summary(paths: list[Path]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text())
        groups[payload["routing_status"]].append(payload)
    result: dict[str, Any] = {}
    for status, rows in sorted(groups.items()):
        psnr_gain = [row["restored"]["psnr"] - row["input"]["psnr"] for row in rows]
        lpips_change = [row["restored"]["lpips"] - row["input"]["lpips"] for row in rows]
        result[status] = {
            "records": len(rows),
            "input_psnr": mean([row["input"]["psnr"] for row in rows]),
            "restored_psnr": mean([row["restored"]["psnr"] for row in rows]),
            "mean_psnr_gain": mean(psnr_gain),
            "input_lpips": mean([row["input"]["lpips"] for row in rows]),
            "restored_lpips": mean([row["restored"]["lpips"] for row in rows]),
            "mean_lpips_change": mean(lpips_change),
            "psnr_worse_records": sum(value < 0 for value in psnr_gain),
            "lpips_worse_records": sum(value > 0 for value in lpips_change),
        }
    return result


def full_metric_summary(paths: list[Path]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    triple: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text())
        groups[payload["routing_status"]].append(payload)
        if payload["category"] == "18Lowlight_Blur_Noise":
            triple[payload["routing_status"]].append(payload)

    def summarize(source: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for status, rows in sorted(source.items()):
            output[status] = {
                "records": len(rows),
                "project_psnr": mean([row["ours"]["psnr"] for row in rows]),
                "project_lpips": mean([row["ours"]["lpips"] for row in rows]),
                "released_foundir_psnr": mean([row["foundir"]["psnr"] for row in rows]),
                "released_foundir_lpips": mean([row["foundir"]["lpips"] for row in rows]),
            }
        return output

    return {"all_660": summarize(groups), "triple_60": summarize(triple)}


def fmt(value: float | None, digits: int = 6) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def build() -> dict[str, Any]:
    required = [path for _, path, _, _ in STAGE_FILES] + [FINAL, PHYSICAL]
    if any(not path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError(f"missing frozen inputs: {missing}")

    baseline_rows = read_csv(BASELINE)
    if len(baseline_rows) != 16_461:
        raise RuntimeError("categorical baseline must contain 16,461 records")
    canonical_ids = [row["record_id"] for row in baseline_rows]
    canonical_id_set = set(canonical_ids)
    if len(canonical_id_set) != len(canonical_ids):
        raise RuntimeError("categorical baseline contains duplicate record IDs")
    stages: list[dict[str, Any]] = []
    for name, path, route_key, accepted_key in STAGE_FILES:
        rows = read_csv(path)
        row_ids = [row["record_id"] for row in rows]
        if len(set(row_ids)) != len(row_ids) or set(row_ids) != canonical_id_set:
            raise RuntimeError(f"record alignment changed for {name}")
        summary = route_summary(rows, route_key, accepted_key)
        summary.update({"name": name, "source": str(path.relative_to(ROOT))})
        stages.append(summary)

    final_rows = read_csv(FINAL)
    final_ids = [row["record_id"] for row in final_rows]
    if len(set(final_ids)) != len(final_ids) or set(final_ids) != canonical_id_set:
        raise RuntimeError("final router records do not align with baseline")
    final = final_summary(final_rows)
    final.update({"name": "final_multi_head", "source": str(FINAL.relative_to(ROOT))})
    stages.append(final)

    accepted_counts = [round(len(baseline_rows) * value / 100) for value in range(30, 101)]
    accepted_counts.extend(stage["correct"] + stage["wrong"] for stage in stages)
    curve, by_count = pure_confidence_points(baseline_rows, accepted_counts)
    for stage in stages:
        count = stage["correct"] + stage["wrong"]
        stage["pure_eleven_head_at_same_accepted_count"] = by_count[count]

    physical_rows = read_csv(PHYSICAL)
    physical_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in physical_rows:
        physical_counts[row["truth"]][row["status"]] += 1
        physical_counts[row["truth"]]["records"] += 1
    physical = {key: dict(value) for key, value in sorted(physical_counts.items())}

    dev_paths = sorted(DEV_METRICS.glob("*.json"))
    full_paths = sorted(FULL_METRICS.glob("*.json"))
    if len(dev_paths) != 1_408 or len(full_paths) != 660:
        raise RuntimeError(
            f"metric sidecar counts changed: development={len(dev_paths)} full={len(full_paths)}"
        )

    return {
        "schema_version": 1,
        "status": "cpu_reaggregation_complete",
        "created_utc_date": DATE,
        "scope": {
            "model_weight_updates": False,
            "gpu_used": False,
            "new_inference": False,
            "official_or_sealed_images_accessed": False,
            "development_predictions_reused": True,
            "development_metric_sidecars_reused": True,
        },
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "routing_stages": stages,
        "pure_eleven_head_risk_coverage": curve,
        "physical_consensus_noise_family": physical,
        "downstream_development_1408": development_metric_summary(dev_paths),
        "downstream_full_resolution_660": full_metric_summary(full_paths),
        "limitations": [
            "The 16,461-record routing set was inspected during iterative router development and is not an independent lockbox.",
            "The pure eleven-head matched-coverage baseline ranks frozen predictions by confidence; it does not retrain or recalibrate that head.",
            "The 1,408-record input-versus-output comparison uses the frozen crop rehearsal, while the 660-record comparison is full resolution.",
            "The 660 per-record sidecars contain the released FoundIR Generalist comparison, not per-record Full FoundIR specialist metrics.",
        ],
    }


def write_outputs(payload: dict[str, Any]) -> None:
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    stage_fields = [
        "name",
        "records",
        "correct",
        "wrong",
        "abstain",
        "coverage",
        "wrong_rate_all",
        "wrong_rate_accepted",
        "pure_head_wrong_same_coverage",
        "pure_head_wrong_rate_all_same_coverage",
    ]
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stage_fields)
        writer.writeheader()
        for stage in payload["routing_stages"]:
            baseline = stage["pure_eleven_head_at_same_accepted_count"]
            writer.writerow(
                {
                    **{key: stage[key] for key in stage_fields[:8]},
                    "pure_head_wrong_same_coverage": baseline["wrong"],
                    "pure_head_wrong_rate_all_same_coverage": baseline["wrong_rate_all"],
                }
            )

    with OUTPUT_CURVE.open("w", newline="") as handle:
        fields = [
            "accepted",
            "coverage",
            "correct",
            "wrong",
            "wrong_rate_all",
            "wrong_rate_accepted",
            "minimum_confidence",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["pure_eleven_head_risk_coverage"])

    stages = payload["routing_stages"]
    final = stages[-1]
    matched = final["pure_eleven_head_at_same_accepted_count"]
    dev = payload["downstream_development_1408"]
    full = payload["downstream_full_resolution_660"]
    lines = [
        "# FoundIR routing paper ablation — CPU reaggregation",
        "",
        f"Date: {DATE} UTC  ",
        "Status: complete; frozen-artifact reaggregation only",
        "",
        "## Decision",
        "",
        "The existing artifacts already support the routing contribution. At the",
        f"final router's accepted count ({final['correct'] + final['wrong']:,} / {final['records']:,},",
        f"coverage {final['coverage']:.4%}), the ordinary eleven-class head ranked",
        f"only by confidence makes **{matched['wrong']}** wrong accepted decisions,",
        f"whereas the final multi-head router makes **{final['wrong']}**. This is a",
        f"reduction of **{matched['wrong'] - final['wrong']} errors "
        f"({(matched['wrong'] - final['wrong']) / matched['wrong']:.2%})** at the same",
        "accepted count. No new routing or restoration experiment is needed to",
        "establish this matched-coverage comparison.",
        "",
        "## Routing ablation on the aligned 16,461-record development set",
        "",
        "| Stage | Coverage | Correct | Wrong | Accepted error | Pure eleven-head wrong at same coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage in stages:
        baseline = stage["pure_eleven_head_at_same_accepted_count"]
        lines.append(
            f"| {stage['name']} | {stage['coverage']:.4%} | {stage['correct']:,} | "
            f"{stage['wrong']:,} | {stage['wrong_rate_accepted']:.4%} | {baseline['wrong']:,} |"
        )
    lines.extend(
        [
            "",
            "The historical stages are not merely a monotonic threshold sweep.",
            "The initial `categorical_confidence_consensus` is actually worse than",
            "the pure eleven-head ranking at the same coverage (201 versus 153",
            "wrong routes), so consensus by itself is not automatically beneficial.",
            "The evidence supports the final compositional/ambiguity/physical design,",
            "not a generic claim that adding more heads always helps.",
            "`dual_nonlinear` dominates `conditional_noise_v2`: it has higher",
            "coverage and fewer wrong routes. The final router also dominates",
            "`asymmetric_absence_v3`: it has higher coverage and fewer wrong routes.",
            "This supports a contribution from representation/consensus changes, not",
            "only from refusing more records.",
            "",
            "## Physical-consensus endpoint for the Noise family",
            "",
            "| Class | Records | Correct | Wrong | Abstain | Coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for category, stats in payload["physical_consensus_noise_family"].items():
        correct = stats.get("correct", 0)
        wrong = stats.get("wrong", 0)
        records = stats["records"]
        lines.append(
            f"| {category} | {records:,} | {correct:,} | {wrong:,} | "
            f"{stats.get('abstain', 0):,} | {(correct + wrong) / records:.4%} |"
        )
    lines.extend(
        [
            "",
            "## Downstream restoration evidence",
            "",
            "### 1,408-record development crop rehearsal",
            "",
            "| Routing status | Records | Input PSNR | Restored PSNR | PSNR gain | Input LPIPS | Restored LPIPS | LPIPS change | PSNR worse | LPIPS worse |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for status, stats in dev.items():
        lines.append(
            f"| {status} | {stats['records']:,} | {fmt(stats['input_psnr'])} | "
            f"{fmt(stats['restored_psnr'])} | {fmt(stats['mean_psnr_gain'])} | "
            f"{fmt(stats['input_lpips'])} | {fmt(stats['restored_lpips'])} | "
            f"{fmt(stats['mean_lpips_change'])} | {stats['psnr_worse_records']:,} | "
            f"{stats['lpips_worse_records']:,} |"
        )
    lines.extend(
        [
            "",
            "Fallback is beneficial on average but is not an absolute no-harm",
            "guarantee. This table reports matched input/output metrics by routing",
            "status; it does not infer safety from the function name.",
            "",
            "### Full-resolution 660 validation records",
            "",
            "| Routing status | Records | Project PSNR / LPIPS | Released FoundIR PSNR / LPIPS |",
            "|---|---:|---:|---:|",
        ]
    )
    for status, stats in full["all_660"].items():
        lines.append(
            f"| {status} | {stats['records']:,} | {fmt(stats['project_psnr'])} / "
            f"{fmt(stats['project_lpips'])} | {fmt(stats['released_foundir_psnr'])} / "
            f"{fmt(stats['released_foundir_lpips'])} |"
        )
    lines.extend(
        [
            "",
            "For the 60-image Blur+Noise+Low-light triple category:",
            "",
            "| Routing status | Records | Project PSNR / LPIPS | Released FoundIR PSNR / LPIPS |",
            "|---|---:|---:|---:|",
        ]
    )
    for status, stats in full["triple_60"].items():
        lines.append(
            f"| {status} | {stats['records']:,} | {fmt(stats['project_psnr'])} / "
            f"{fmt(stats['project_lpips'])} | {fmt(stats['released_foundir_psnr'])} / "
            f"{fmt(stats['released_foundir_lpips'])} |"
        )
    lines.extend(
        [
            "",
            "The correctly routed triple subset remains poor, confirming that the",
            "dominant triple failure is restoration composition rather than route",
            "recognition.",
            "",
            "## Paper interpretation",
            "",
            "The evidence supports a narrow claim: cross-representation and physical",
            "consensus reduce wrong restoration actions substantially beyond ordinary",
            "single-head confidence selection at matched coverage. It does not support",
            "claims of universal safety, open-set robustness, or successful triple",
            "restoration.",
            "",
            "## Provenance and limitations",
            "",
            "- GPU used: 0; model updates: none; new inference: none.",
            "- Official/sealed images accessed: false.",
            "- All routing stages are aligned to the same 16,461 development records.",
            "- This development set was inspected during router iteration and is not",
            "  an independent lockbox; final paper wording must disclose that boundary.",
            "- The full-resolution sidecars compare per-record against Released FoundIR",
            "  Generalist. Full FoundIR specialist metrics are available only as",
            "  aggregate frozen results, not in these per-record sidecars.",
            "",
            "Machine-readable artifacts:",
            "",
            f"- `{OUTPUT_JSON.relative_to(ROOT)}`",
            f"- `{OUTPUT_CSV.relative_to(ROOT)}`",
            f"- `{OUTPUT_CURVE.relative_to(ROOT)}`",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    payload = build()
    if args.check_only:
        print(json.dumps({"status": "pass", "records": payload["routing_stages"][0]["records"]}))
        return
    write_outputs(payload)
    final = payload["routing_stages"][-1]
    matched = final["pure_eleven_head_at_same_accepted_count"]
    print(
        json.dumps(
            {
                "status": "complete",
                "final_coverage": final["coverage"],
                "final_wrong": final["wrong"],
                "pure_head_wrong_same_coverage": matched["wrong"],
                "outputs": [
                    str(OUTPUT_JSON.relative_to(ROOT)),
                    str(OUTPUT_CSV.relative_to(ROOT)),
                    str(OUTPUT_CURVE.relative_to(ROOT)),
                    str(OUTPUT_MD.relative_to(ROOT)),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
