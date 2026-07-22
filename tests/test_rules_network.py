import json
from pathlib import Path

from spark_doctor.models import ScanReport
from spark_doctor.rules import run_rules

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> ScanReport:
    data = json.loads((FIXTURES / name).read_text())
    return ScanReport.model_validate(data)


def _nic_findings(report: ScanReport) -> list:
    return [f for f in run_rules(report) if f.rule_id == "network.nic_link_below_1g"]


def test_realtek_below_1g_flags_with_defect_note():
    findings = _nic_findings(_load("nic_link_below_1g_realtek.json"))
    # Only the operstate=up interface is flagged, not the down one.
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "warning"
    assert "enp1s0" in f.evidence[0]
    assert "autonegotiation defect" in f.source_note
    assert "r8169" in f.source_note


def test_generic_below_1g_flags_without_defect_note():
    findings = _nic_findings(_load("nic_link_below_1g_generic.json"))
    # Only the 100 Mb/s interface is flagged; the 2500 Mb/s one is not.
    assert len(findings) == 1
    f = findings[0]
    assert "eth0" in f.evidence[0]
    assert "autonegotiation defect" not in f.source_note


def test_healthy_interfaces_no_finding():
    assert _nic_findings(_load("healthy_minimal.json")) == []
