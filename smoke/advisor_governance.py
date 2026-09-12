#!/usr/bin/env python3
"""Create advisor gate reports and fail closed while an advisor hold is unresolved."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOVERNANCE_DIR = ROOT / "reports" / "advisor_governance"
SOW_STAGE2_THRESHOLDS = {
    "blur": {"psnr": 25.9, "lpips": 0.18},
    "noise": {"psnr": 38.9, "lpips": 0.10},
    "haze": {"psnr": 22.1, "lpips": 0.19},
    "rain": {"psnr": 34.2, "lpips": 0.08},
    "low_light": {"psnr": 18.9, "lpips": 0.28},
}
RESUMING_DECISIONS = {"resume_same_mainline", "approve_documented_change"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing governance artifact: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _workspace_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else root / path
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _metrics(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = evidence.get("sealed_task_metrics")
    if not isinstance(metrics, dict):
        summary = evidence.get("summary")
        metrics = summary.get("task_metrics") if isinstance(summary, dict) else None
    return metrics if isinstance(metrics, dict) else {}


def _metric_checks(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for task, threshold in SOW_STAGE2_THRESHOLDS.items():
        values = metrics.get(task)
        if not isinstance(values, dict) or "psnr" not in values or "lpips" not in values:
            continue
        psnr = float(values["psnr"])
        lpips = float(values["lpips"])
        checks[task] = {
            "metrics": {"psnr": psnr, "lpips": lpips},
            "sow": threshold,
            "pass": {"psnr": psnr >= threshold["psnr"], "lpips": lpips <= threshold["lpips"]},
        }
    return checks


def infer_outcome(evidence: Mapping[str, Any]) -> str:
    gate = evidence.get("automatic_gate_pass")
    if gate is True:
        return "pass"
    if gate is False:
        return "failure"
    status = str(evidence.get("status", ""))
    if "failure" in status or "failed" in status or "error" in status:
        return "failure"
    return "information"


def validate_decision(
    hold: Mapping[str, Any], decision_path: Path, *, governance_dir: Path
) -> dict[str, Any]:
    decision = _load_json(decision_path)
    required = {"schema_version", "hold_id", "hold_sha256", "decided_by", "decision", "statement", "recorded_at"}
    if not required.issubset(decision):
        raise RuntimeError("advisor_decision.json is missing required recorded-decision fields")
    if decision["schema_version"] != 1:
        raise RuntimeError("unsupported advisor decision schema")
    if decision["hold_id"] != hold.get("hold_id"):
        raise RuntimeError("advisor decision does not resolve the active hold")
    event_path = governance_dir / "holds" / f"{hold.get('hold_id')}.json"
    if not event_path.is_file() or decision["hold_sha256"] != sha256(event_path):
        raise RuntimeError("advisor decision is not bound to the immutable hold event")
    if decision["decided_by"] not in {"advisor", "user_after_advisor_response"}:
        raise RuntimeError("advisor decision must identify advisor authority")
    if decision["decision"] not in RESUMING_DECISIONS:
        raise RuntimeError("advisor decision does not authorize further experiments")
    if not isinstance(decision["statement"], str) or not decision["statement"].strip():
        raise RuntimeError("advisor decision statement is empty")
    if not isinstance(decision["recorded_at"], str) or not decision["recorded_at"].strip():
        raise RuntimeError("advisor decision timestamp is empty")
    return decision


def check_hold(*, governance_dir: Path = DEFAULT_GOVERNANCE_DIR) -> None:
    hold_path = governance_dir / "advisor_hold.json"
    if not hold_path.exists():
        print("ADVISOR_HOLD_CLEAR no_active_hold")
        return
    hold = _load_json(hold_path)
    event_path = governance_dir / "holds" / f"{hold.get('hold_id')}.json"
    if not event_path.is_file() or event_path.read_bytes() != hold_path.read_bytes():
        raise RuntimeError("active advisor hold does not match its immutable event")
    workspace_root = governance_dir.resolve().parents[1]
    report_path = _workspace_path(workspace_root, str(hold.get("report", "")))
    if not report_path.is_file() or sha256(report_path) != hold.get("report_sha256"):
        raise RuntimeError("active advisor hold report is missing or changed")
    decision_path = governance_dir / "advisor_decision.json"
    if not decision_path.exists():
        raise RuntimeError(
            f"ADVISOR_HOLD_ACTIVE {hold.get('hold_id')}: send {hold.get('report')} to the advisor; "
            f"do not launch another experiment until a matching {decision_path} records the decision"
        )
    validate_decision(hold, decision_path, governance_dir=governance_dir)
    print(f"ADVISOR_HOLD_RESOLVED {hold['hold_id']} decision={decision_path}")


def _render_markdown(report: Mapping[str, Any]) -> str:
    diagnostic_warning = ""
    if report["diagnostic_only"]:
        diagnostic_warning = (
            "\n> **DIAGNOSTIC ONLY / 非主线：本结果不是 FLUX.2 主线结果，"
            "不得替代或包装成 FLUX.2/SOW final result。**\n"
        )
    lines = [f"# Advisor 阶段报告：{report['gate']}", ""]
    if diagnostic_warning:
        lines.extend([diagnostic_warning.strip(), ""])
    lines.extend(
        [
        f"- 结论：`{report['outcome']}`",
        f"- Exact model：`{report['model']}`",
        f"- 是否为 FLUX.2 主线结果：`{str(report['mainline_result']).lower()}`",
        f"- Config：`{report['config']['path']}` (`{report['config']['sha256']}`)",
        f"- Checkpoint：`{report['checkpoint']['path']}` (`{report['checkpoint']['sha256']}`)",
        f"- Evaluation split：`{report['evaluation_split']}`",
        f"- Sealed internal 是否已访问：`{str(report['test_access']['sealed_internal_accessed']).lower()}`",
        f"- Official FoundIR test 是否已访问：`{str(report['test_access']['official_foundir_test_accessed']).lower()}`",
        f"- Evidence：`{report['evidence']['path']}` (`{report['evidence']['sha256']}`)",
        "",
        "## Metrics 与 SOW 对照",
        "",
        "| Task | PSNR / target | LPIPS / target | Checks |",
        "|---|---:|---:|---:|",
        ]
    )
    if report["metrics_vs_sow"]:
        for task, values in report["metrics_vs_sow"].items():
            lines.append(
                f"| {task} | {values['metrics']['psnr']:.4f} / >= {values['sow']['psnr']:.2f} | "
                f"{values['metrics']['lpips']:.5f} / <= {values['sow']['lpips']:.2f} | "
                f"{int(values['pass']['psnr']) + int(values['pass']['lpips'])}/2 |"
            )
    else:
        lines.append("| n/a | n/a | n/a | informational gate |")
    lines.extend(
        [
            "",
            "## 结论与待决策项",
            "",
            f"- 原因/状态：{report['root_cause_or_status']}",
            f"- 请求 advisor 决策：{report['advisor_decision_question']}",
            "",
            "在 advisor/user 的明确、记录化决定前，不改变 model family、architecture、split、"
            "evaluation target 或 acceptance protocol。",
        ]
    )
    return "\n".join(line for line in lines if line is not None) + "\n"


def emit_report(
    *,
    root: Path,
    governance_dir: Path,
    gate: str,
    evidence_path: Path,
    config_path: Path,
    model: str,
    evaluation_split: str,
    outcome: str,
    root_cause_or_status: str,
    advisor_decision_question: str,
    diagnostic_only: bool,
    hold_on_failure: bool,
    require_decision: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    evidence_path = _workspace_path(root, evidence_path)
    config_path = _workspace_path(root, config_path)
    evidence = _load_json(evidence_path)
    evidence_sha = sha256(evidence_path)
    config_sha = sha256(config_path)
    evidence_config_sha = evidence.get("config_sha256")
    if evidence_config_sha is not None and evidence_config_sha != config_sha:
        raise RuntimeError("advisor report config does not match evidence config_sha256")
    if outcome == "auto":
        outcome = infer_outcome(evidence)
    if outcome not in {"pass", "failure", "information"}:
        raise ValueError(f"unsupported outcome: {outcome}")
    checkpoint_path_text = str(
        evidence.get("selected_checkpoint")
        or evidence.get("final_checkpoint")
        or evidence.get("checkpoint")
        or "not_applicable"
    )
    checkpoint_sha = str(
        evidence.get("selected_checkpoint_sha256")
        or evidence.get("final_checkpoint_sha256")
        or evidence.get("checkpoint_sha256")
        or "not_applicable"
    )
    official_access = evidence.get("protected_foundir_test_accessed")
    if official_access is not False:
        raise RuntimeError("advisor reporting refuses evidence without explicit official-test protection")
    sealed_access = evaluation_split in {"sealed_internal", "internal_test"} or str(evidence.get("status", "")).startswith("internal_protocol")
    event_key = "\0".join(
        (
            gate,
            evidence_sha,
            outcome,
            str(diagnostic_only),
            model,
            evaluation_split,
            root_cause_or_status,
            advisor_decision_question,
        )
    ).encode()
    event_id = f"{gate}-{hashlib.sha256(event_key).hexdigest()[:16]}"
    report = {
        "schema_version": 1,
        "report_id": event_id,
        "generated_at": utc_now(),
        "gate": gate,
        "outcome": outcome,
        "model": model,
        "mainline_result": not diagnostic_only,
        "diagnostic_only": diagnostic_only,
        "claim_classification": (
            "diagnostic_non_mainline_not_flux2_mainline_result"
            if diagnostic_only
            else "vlm_guided_flux2_mainline_result"
        ),
        "config": {"path": _relative(root, config_path), "sha256": config_sha},
        "checkpoint": {"path": checkpoint_path_text, "sha256": checkpoint_sha},
        "evaluation_split": evaluation_split,
        "metrics_vs_sow": _metric_checks(_metrics(evidence)),
        "test_access": {
            "sealed_internal_accessed": sealed_access,
            "official_foundir_test_accessed": False,
        },
        "evidence": {"path": _relative(root, evidence_path), "sha256": evidence_sha},
        "root_cause_or_status": root_cause_or_status,
        "advisor_decision_question": advisor_decision_question,
    }
    json_path = governance_dir / "reports" / f"{event_id}.json"
    markdown_path = governance_dir / "reports" / f"{event_id}.md"
    if json_path.exists():
        existing_report = _load_json(json_path)
        comparable_existing = {key: value for key, value in existing_report.items() if key != "generated_at"}
        comparable_new = {key: value for key, value in report.items() if key != "generated_at"}
        if comparable_existing != comparable_new:
            raise RuntimeError(f"existing advisor report changed: {json_path}")
        report = existing_report
    else:
        write_immutable(json_path, canonical_json(report))
    write_immutable(markdown_path, _render_markdown(report).encode())
    if (hold_on_failure and outcome == "failure") or require_decision:
        hold = {
            "schema_version": 1,
            "hold_id": event_id,
            "created_at": report["generated_at"],
            "reason": (
                "proposed route/architecture change requires advisor decision before experiments"
                if require_decision and outcome != "failure"
                else "terminal scientific/runtime failure requires advisor decision before further experiments"
            ),
            "report": _relative(root, markdown_path),
            "report_sha256": sha256(markdown_path),
            "evidence": report["evidence"],
            "allowed_while_active": ["read_only_audit", "advisor_report_preparation"],
            "blocked_while_active": ["training", "evaluation", "model_family_change", "architecture_change", "diagnostic_branch"],
        }
        event_path = governance_dir / "holds" / f"{event_id}.json"
        hold_content = canonical_json(hold)
        write_immutable(event_path, hold_content)
        active_path = governance_dir / "advisor_hold.json"
        if active_path.exists() and active_path.read_bytes() != hold_content:
            active = _load_json(active_path)
            decision_path = governance_dir / "advisor_decision.json"
            if not decision_path.exists():
                raise RuntimeError("refusing to replace an unresolved advisor hold")
            validate_decision(active, decision_path, governance_dir=governance_dir)
            temporary = active_path.with_suffix(".json.next")
            temporary.write_bytes(hold_content)
            os.replace(temporary, active_path)
        elif not active_path.exists():
            write_immutable(active_path, hold_content)
    print(f"ADVISOR_REPORT_READY {markdown_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--governance-dir", type=Path, default=DEFAULT_GOVERNANCE_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    emit = subparsers.add_parser("emit")
    emit.add_argument("--gate", required=True)
    emit.add_argument("--evidence", type=Path, required=True)
    emit.add_argument("--config", type=Path, required=True)
    emit.add_argument("--model", required=True)
    emit.add_argument("--split", required=True)
    emit.add_argument("--outcome", default="auto", choices=("auto", "pass", "failure", "information"))
    emit.add_argument("--root-cause-or-status", required=True)
    emit.add_argument("--advisor-decision-question", required=True)
    emit.add_argument("--diagnostic-only", action="store_true")
    emit.add_argument("--hold-on-failure", action="store_true")
    emit.add_argument("--require-decision", action="store_true")
    runtime = subparsers.add_parser("runtime-failure")
    runtime.add_argument("--gate", required=True)
    runtime.add_argument("--evidence", type=Path, required=True)
    runtime.add_argument("--config", type=Path, required=True)
    runtime.add_argument("--model", required=True)
    runtime.add_argument("--status", type=int, required=True)
    runtime.add_argument("--log", required=True)
    runtime.add_argument("--diagnostic-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    governance_dir = _workspace_path(root, args.governance_dir)
    if args.command == "check":
        check_hold(governance_dir=governance_dir)
        return
    if args.command == "runtime-failure" and (governance_dir / "advisor_hold.json").exists():
        try:
            check_hold(governance_dir=governance_dir)
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            return
    if args.command == "runtime-failure":
        config_path = _workspace_path(root, args.config)
        runtime_key = hashlib.sha256(
            f"{args.gate}\0{sha256(config_path)}\0{args.status}\0{args.log}".encode()
        ).hexdigest()[:16]
        runtime_evidence = governance_dir / "runtime_evidence" / f"{args.gate}-{runtime_key}.json"
        write_immutable(
            runtime_evidence,
            canonical_json(
                {
                    "schema_version": 1,
                    "status": "workflow_runtime_failure",
                    "automatic_gate_pass": False,
                    "config_sha256": sha256(config_path),
                    "protected_foundir_test_accessed": False,
                    "runner_exit_status": args.status,
                    "log": args.log,
                }
            ),
        )
        emit_report(
            root=root,
            governance_dir=governance_dir,
            gate=args.gate,
            evidence_path=runtime_evidence,
            config_path=args.config,
            model=args.model,
            evaluation_split="runtime_before_gate_completion",
            outcome="failure",
            root_cause_or_status=f"workflow exit status {args.status}; inspect {args.log}",
            advisor_decision_question="请判断应修复并恢复同一 FLUX.2 mainline，还是停止/修改路线。",
            diagnostic_only=args.diagnostic_only,
            hold_on_failure=True,
            require_decision=False,
        )
        return
    emit_report(
        root=root,
        governance_dir=governance_dir,
        gate=args.gate,
        evidence_path=args.evidence,
        config_path=args.config,
        model=args.model,
        evaluation_split=args.split,
        outcome=args.outcome,
        root_cause_or_status=args.root_cause_or_status,
        advisor_decision_question=args.advisor_decision_question,
        diagnostic_only=args.diagnostic_only,
        hold_on_failure=args.hold_on_failure,
        require_decision=args.require_decision,
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(3) from None
