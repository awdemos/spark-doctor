from __future__ import annotations

from typing import Any

from ..models import CollectorStatus
from ..shell import run


def collect_logs() -> tuple[dict[str, Any], CollectorStatus]:
    status = CollectorStatus(name="logs", ok=True)
    out: dict[str, Any] = {}

    dmesg = run(["dmesg", "-T"], timeout=8)
    if dmesg.stdout:
        out["dmesg_tail"] = "\n".join(dmesg.stdout.splitlines()[-200:])
    if not dmesg.ok:
        status.errors.append(f"dmesg: {dmesg.error}: {dmesg.stderr.strip()[:200]}")

    jctl = run(["journalctl", "-b", "--no-pager", "-n", "300"], timeout=10)
    if jctl.stdout:
        out["journal_tail"] = "\n".join(jctl.stdout.splitlines()[-300:])
    if not jctl.ok:
        status.errors.append(f"journalctl: {jctl.error}: {jctl.stderr.strip()[:200]}")

    status.ok = bool(out) or not status.errors
    return out, status
