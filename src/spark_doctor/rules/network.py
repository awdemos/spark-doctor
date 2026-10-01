from __future__ import annotations

from ..models import Finding, ScanReport
from .engine import Rule


def _eval_nic_link_speed(report: ScanReport) -> list[Finding]:
    net = report.network if isinstance(report.network, dict) else {}
    interfaces = net.get("interfaces")
    if not isinstance(interfaces, list):
        return []

    findings: list[Finding] = []
    for iface in interfaces:
        if not isinstance(iface, dict):
            continue
        if iface.get("operstate") != "up":
            continue
        speed = iface.get("speed_mbps")
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not 0 < speed < 1000:
            continue

        name = str(iface.get("name", "?"))
        driver = str(iface.get("driver") or "")
        realtek = any(x in driver.lower() for x in ("r8127", "r8125", "r8169"))

        source_note = (
            "A NIC in state 'up' negotiating below 1 Gb/s usually indicates a cabling "
            "or autonegotiation fault."
        )
        if realtek:
            source_note = (
                "For Realtek drivers (r8127, r8125, r8169), verify the controller model "
                "and supported link modes before diagnosing an autonegotiation defect. "
                "The driver name alone does not establish the rated link speed."
            )

        findings.append(
            Finding(
                rule_id="network.nic_link_below_1g",
                title="NIC negotiated below 1 Gb/s",
                severity="warning",
                confidence="medium",
                evidence=[
                    f"{name}: link {int(speed)} Mb/s (operstate=up, "
                    f"driver={driver or 'unknown'})"
                ],
                explanation=(
                    "This interface is up but negotiated a link speed below 1 Gb/s. "
                    "Model pulls, container image pulls, and any network-bound serving "
                    "traffic will be throttled."
                ),
                recommended_actions=[
                    "Check the cable and switch port; reseat or replace the cable.",
                    "Inspect `ethtool <iface>` for supported and advertised link modes; "
                    "confirm the NIC and switch support the desired speed before changing settings.",
                ],
                source_note=source_note,
            )
        )
    return findings


rule_network_nic_link_below_1g = Rule(
    id="network.nic_link_below_1g",
    title="NIC link speed below 1 Gb/s",
    fn=_eval_nic_link_speed,
)
