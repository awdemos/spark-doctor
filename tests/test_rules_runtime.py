import json
from pathlib import Path

from spark_doctor.models import ScanReport
from spark_doctor.rules import run_rules

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> ScanReport:
    data = json.loads((FIXTURES / name).read_text())
    return ScanReport.model_validate(data)


def test_hook_installed_no_named_runtime_no_finding():
    report = _load("docker_hook_no_named_runtime.json")
    findings = run_rules(report)
    assert not any(f.rule_id == "runtime.docker_unhealthy" for f in findings)


def test_cdi_present_no_named_runtime_no_finding():
    report = _load("docker_cdi_no_named_runtime.json")
    findings = run_rules(report)
    assert not any(f.rule_id == "runtime.docker_unhealthy" for f in findings)


def test_nothing_present_triggers_warning():
    report = _load("docker_no_runtime_no_hook_no_cdi.json")
    findings = run_rules(report)
    runtime = [f for f in findings if f.rule_id == "runtime.docker_unhealthy"]
    assert runtime, "expected runtime.docker_unhealthy finding"
    assert runtime[0].severity == "warning"
    assert "NVIDIA runtime is not registered with Docker." in runtime[0].evidence
