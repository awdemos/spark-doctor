from __future__ import annotations

import glob
import os
from typing import Any

from ..models import CollectorStatus
from ..shell import read_text, run


def collect_network() -> tuple[dict[str, Any], CollectorStatus]:
    status = CollectorStatus(name="network", ok=True)
    out: dict[str, Any] = {"interfaces": []}

    addr = run(["ip", "-br", "addr"], timeout=5)
    if addr.ok:
        out["ip_br_addr"] = addr.stdout
    else:
        status.errors.append(f"ip -br addr: {addr.error}: {addr.stderr.strip()[:200]}")
    link = run(["ip", "-br", "link"], timeout=5)
    if link.ok:
        out["ip_br_link"] = link.stdout
    else:
        status.errors.append(f"ip -br link: {link.error}: {link.stderr.strip()[:200]}")

    interfaces: list[dict[str, Any]] = []
    for path in sorted(glob.glob("/sys/class/net/*")):
        name = os.path.basename(path)
        if name == "lo":
            continue
        entry: dict[str, Any] = {"name": name}
        speed = read_text(os.path.join(path, "speed"))
        if speed and speed.strip() and speed.strip() != "-1":
            try:
                entry["speed_mbps"] = int(speed.strip())
            except ValueError:
                status.errors.append(f"{name}: invalid link speed")
        operstate = read_text(os.path.join(path, "operstate"))
        if operstate:
            entry["operstate"] = operstate.strip()
        else:
            status.errors.append(f"{name}: cannot read operstate")
        # detect mellanox/connectx by driver path
        driver_link = os.path.join(path, "device", "driver")
        try:
            driver = os.path.basename(os.readlink(driver_link))
            entry["driver"] = driver
            if "mlx" in driver.lower():
                entry["connectx_like"] = True
        except OSError:
            pass
        # Wireless drivers need not implement the wired ethtool speed attribute.
        if (
            entry.get("operstate") == "up" and entry.get("driver")
            and "speed_mbps" not in entry
            and not os.path.isdir(os.path.join(path, "wireless"))
        ):
            status.errors.append(f"{name}: link speed unavailable for active physical interface")
        interfaces.append(entry)
    out["interfaces"] = interfaces

    rdma = run(["rdma", "link"], timeout=5)
    if rdma.ok:
        out["rdma_link"] = rdma.stdout
    elif rdma.error != "command_not_found" or any(i.get("connectx_like") for i in interfaces):
        status.errors.append(f"rdma link: {rdma.error}: {rdma.stderr.strip()[:200]}")
    ib = run(["ibstat"], timeout=5)
    if ib.ok:
        out["ibstat"] = ib.stdout
    # ibstat (infiniband-diags) is not installed on DGX OS by default, and `rdma link` already
    # reports ConnectX port state, so a missing ibstat only matters when rdma failed too.
    elif ib.error != "command_not_found" or (
        not rdma.ok and any(i.get("connectx_like") for i in interfaces)
    ):
        status.errors.append(f"ibstat: {ib.error}: {ib.stderr.strip()[:200]}")

    status.ok = bool(interfaces or addr.ok or link.ok) or not status.errors
    return out, status
