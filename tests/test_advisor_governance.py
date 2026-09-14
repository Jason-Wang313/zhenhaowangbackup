import hashlib
import json
from pathlib import Path

import pytest

from smoke.advisor_governance import check_hold, emit_report, infer_outcome, sha256


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def fixture_files(root: Path, *, passed: bool = False) -> tuple[Path, Path]:
    config = root / "configs/main.json"
    write_json(config, {"model": "flux2"})
    evidence = root / "results/final.json"
    write_json(
        evidence,
        {
            "status": "internal_protocol_pass" if passed else "internal_protocol_scientific_failure",
            "automatic_gate_pass": passed,
            "config_sha256": sha256(config),
            "selected_checkpoint": "checkpoints/step.safetensors",
            "selected_checkpoint_sha256": "abc",
            "protected_foundir_test_accessed": False,
            "sealed_task_metrics": {"blur": {"psnr": 20.0, "lpips": 0.25}},
        },
    )
    return config, evidence


def emit(root: Path, *, diagnostic: bool = False) -> dict:
    config, evidence = fixture_files(root)
    return emit_report(
        root=root,
        governance_dir=root / "reports/advisor_governance",
        gate="sealed_internal_final",
        evidence_path=evidence,
        config_path=config,
        model="released FoundIR routing" if diagnostic else "VLM-guided FLUX.2 Klein 4B",
        evaluation_split="sealed_internal",
        outcome="auto",
        root_cause_or_status="scientific threshold failure",
        advisor_decision_question="choose next route",
        diagnostic_only=diagnostic,
        hold_on_failure=True,
    )


def test_failure_creates_immutable_report_and_hold(tmp_path: Path) -> None:
    report = emit(tmp_path)
    governance = tmp_path / "reports/advisor_governance"
    hold = json.loads((governance / "advisor_hold.json").read_text())
    assert hold["hold_id"] == report["report_id"]
    assert (governance / "holds" / f"{hold['hold_id']}.json").is_file()
    assert (governance / "reports" / f"{hold['hold_id']}.md").is_file()
    assert report["metrics_vs_sow"]["blur"]["pass"] == {"psnr": False, "lpips": False}


def test_unresolved_hold_blocks_and_matching_recorded_decision_allows(tmp_path: Path) -> None:
    emit(tmp_path)
    governance = tmp_path / "reports/advisor_governance"
    with pytest.raises(RuntimeError, match="ADVISOR_HOLD_ACTIVE"):
        check_hold(governance_dir=governance)
    hold = json.loads((governance / "advisor_hold.json").read_text())
    write_json(
        governance / "advisor_decision.json",
        {
            "schema_version": 1,
            "hold_id": hold["hold_id"],
            "hold_sha256": sha256(governance / "holds" / f"{hold['hold_id']}.json"),
            "decided_by": "advisor",
            "decision": "resume_same_mainline",
            "statement": "Keep FLUX.2 and run the documented capacity correction.",
            "recorded_at": "2026-09-11T00:00:00Z",
        },
    )
    check_hold(governance_dir=governance)


def test_diagnostic_is_never_labeled_as_flux_mainline_result(tmp_path: Path) -> None:
    report = emit(tmp_path, diagnostic=True)
    assert report["mainline_result"] is False
    assert report["claim_classification"] == "diagnostic_non_mainline_not_flux2_mainline_result"
    markdown = (
        tmp_path
        / "reports/advisor_governance/reports"
        / f"{report['report_id']}.md"
    ).read_text()
    assert "DIAGNOSTIC ONLY / 非主线" in markdown
    assert "不得替代或包装成 FLUX.2/SOW final result" in markdown


def test_decision_must_match_exact_immutable_hold(tmp_path: Path) -> None:
    emit(tmp_path)
    governance = tmp_path / "reports/advisor_governance"
    hold = json.loads((governance / "advisor_hold.json").read_text())
    write_json(
        governance / "advisor_decision.json",
        {
            "schema_version": 1,
            "hold_id": hold["hold_id"],
            "hold_sha256": hashlib.sha256(b"wrong").hexdigest(),
            "decided_by": "advisor",
            "decision": "resume_same_mainline",
            "statement": "resume",
            "recorded_at": "2026-09-11T00:00:00Z",
        },
    )
    with pytest.raises(RuntimeError, match="immutable hold event"):
        check_hold(governance_dir=governance)


def test_every_foundir_shell_entrypoint_checks_the_hold() -> None:
    root = Path(__file__).resolve().parents[1]
    entrypoints = sorted((root / "smoke").glob("launch_foundir_*.sh"))
    entrypoints += sorted((root / "smoke").glob("run_foundir_*.sh"))
    assert entrypoints
    missing = [path.name for path in entrypoints if "advisor_governance.py\" check" not in path.read_text()]
    assert missing == []


def test_proposed_architecture_change_creates_hold_before_experiment(tmp_path: Path) -> None:
    config, evidence = fixture_files(tmp_path, passed=True)
    report = emit_report(
        root=tmp_path,
        governance_dir=tmp_path / "reports/advisor_governance",
        gate="proposed_architecture_change",
        evidence_path=evidence,
        config_path=config,
        model="proposed replacement model",
        evaluation_split="not_run",
        outcome="information",
        root_cause_or_status="proposal only",
        advisor_decision_question="approve or reject",
        diagnostic_only=True,
        hold_on_failure=False,
        require_decision=True,
    )
    hold = json.loads(
        (tmp_path / "reports/advisor_governance/advisor_hold.json").read_text()
    )
    assert hold["hold_id"] == report["report_id"]
    assert "proposed route/architecture change" in hold["reason"]


def test_metric_gate_failure_takes_precedence_over_non_failure_status() -> None:
    assert (
        infer_outcome(
            {
                "status": "zimage_blur_teacher_tiny_overfit_evaluation_complete",
                "automatic_metric_gate_pass": False,
                "human_visual_review_required": True,
                "human_visual_review_complete": False,
            }
        )
        == "failure"
    )
    assert (
        infer_outcome(
            {
                "automatic_gate_pass": True,
                "automatic_metric_gate_pass": False,
            }
        )
        == "failure"
    )


def test_passing_metric_gate_awaits_required_human_visual_review() -> None:
    evidence = {
        "automatic_metric_gate_pass": True,
        "human_visual_review_required": True,
        "human_visual_review_complete": False,
    }
    assert infer_outcome(evidence) == "information"
    evidence["human_visual_review_complete"] = True
    assert infer_outcome(evidence) == "pass"


@pytest.mark.parametrize(
    "access_flags",
    [
        {"protected_foundir_test_accessed": False},
        {"official_foundir_test_accessed": False},
        {
            "protected_foundir_test_accessed": False,
            "official_foundir_test_accessed": False,
        },
    ],
)
def test_report_accepts_each_explicit_official_test_protection_name(
    tmp_path: Path, access_flags: dict
) -> None:
    config, evidence = fixture_files(tmp_path)
    payload = json.loads(evidence.read_text())
    payload.pop("protected_foundir_test_accessed")
    payload.update(access_flags)
    write_json(evidence, payload)
    report = emit_report(
        root=tmp_path,
        governance_dir=tmp_path / "reports/advisor_governance",
        gate="protection_alias",
        evidence_path=evidence,
        config_path=config,
        model="diagnostic model",
        evaluation_split="development",
        outcome="auto",
        root_cause_or_status="test evidence",
        advisor_decision_question="none",
        diagnostic_only=True,
        hold_on_failure=False,
    )
    assert report["test_access"]["official_foundir_test_accessed"] is False


@pytest.mark.parametrize(
    "access_flags",
    [
        {},
        {"official_foundir_test_accessed": True},
        {
            "protected_foundir_test_accessed": False,
            "official_foundir_test_accessed": True,
        },
        {
            "protected_foundir_test_accessed": True,
            "official_foundir_test_accessed": False,
        },
    ],
)
def test_report_rejects_missing_or_conflicting_official_test_protection(
    tmp_path: Path, access_flags: dict
) -> None:
    config, evidence = fixture_files(tmp_path)
    payload = json.loads(evidence.read_text())
    payload.pop("protected_foundir_test_accessed")
    payload.update(access_flags)
    write_json(evidence, payload)
    with pytest.raises(RuntimeError, match="official-test protection"):
        emit_report(
            root=tmp_path,
            governance_dir=tmp_path / "reports/advisor_governance",
            gate="protection_alias",
            evidence_path=evidence,
            config_path=config,
            model="diagnostic model",
            evaluation_split="development",
            outcome="auto",
            root_cause_or_status="test evidence",
            advisor_decision_question="none",
            diagnostic_only=True,
            hold_on_failure=False,
        )
