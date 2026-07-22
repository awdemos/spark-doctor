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
        if not isinstance(speed, (int, float)) or speed >= 1000:
            continue

        name = str(iface.get("name", "?"))
        driver = str(iface.get("driver") or "")
        realtek_25g = any(x in driver.lower() for x in ("r8127", "r8125"))

        source_note = (
            "A NIC in state 'up' negotiating below 1 Gb/s usually indicates a cabling "
            "or autonegotiation fault."
        )
        if realtek_25g:
            source_note = (
                "Realtek RTL8127/RTL8125 2.5GbE controllers have a known "
                "autonegotiation defect that can drop the link well below its rated "
                "speed; try forcing the speed with ethtool or a different port/cable."
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
                    "Force the link speed with `ethtool -s <iface> speed 2500 autoneg on` "
                    "(or the rated speed) if autonegotiation is failing.",
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
