import json
from pathlib import Path

import pytest

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
    assert runtime[0].fix_commands == [
        "sudo nvidia-ctk runtime configure --runtime=docker",
        "sudo systemctl restart docker",
    ]
    assert any("interrupt workloads" in action for action in runtime[0].recommended_actions)


@pytest.mark.parametrize(
    "overrides",
    [
        {"nvidia_ctk_installed": False},
        {"daemon_reachable": False, "socket_accessible": False},
        {"docker_installed": False, "daemon_reachable": False},
    ],
)
def test_missing_prerequisites_do_not_suggest_commands(overrides):
    report = _load("docker_no_runtime_no_hook_no_cdi.json")
    report.docker.update(overrides)
    runtime = [f for f in run_rules(report) if f.rule_id == "runtime.docker_unhealthy"]
    assert runtime
    assert runtime[0].fix_commands == []


@pytest.mark.parametrize(
    "gpu_path", ["nvidia_runtime_available", "nvidia_hook_installed", "cdi_specs_present"]
)
def test_working_gpu_path_does_not_suggest_registration(gpu_path):
    report = _load("docker_no_runtime_no_hook_no_cdi.json")
    report.docker[gpu_path] = True
    assert not any(f.rule_id == "runtime.docker_unhealthy" for f in run_rules(report))
