from __future__ import annotations

import json
import os
import re
from typing import Any

from ..models import CollectorStatus
from ..shell import run, which

CDI_DIRS = ("/etc/cdi", "/var/run/cdi")


def _cdi_specs_present() -> bool:
    for d in CDI_DIRS:
        try:
            if os.path.isdir(d) and any(os.scandir(d)):
                return True
        except OSError:
            continue
    return False


def _cdi_list_device_count() -> int:
    """`nvidia-ctk cdi list` enumerates generated CDI device specs; catches specs
    the /etc/cdi dir scan misses. Best-effort: 0 if the tool is absent/unparseable."""
    cdi = run(["nvidia-ctk", "cdi", "list"], timeout=5)
    if cdi.error == "command_not_found":
        return 0
    m = re.search(r"Found\s+(\d+)\s+CDI device", f"{cdi.stdout}\n{cdi.stderr}")
    return int(m.group(1)) if m else 0


def _finalize_gpu_readiness(out: dict[str, Any]) -> None:
    """GPU-in-Docker is ready if ANY GPU-injection path exists: a registered
    nvidia runtime, a usable CDI spec, or the OCI prestart hook (`--gpus all`)."""
    out["gpu_docker_ready"] = bool(
        out.get("nvidia_runtime_available")
        or out.get("cdi_specs_present")
        or out.get("nvidia_hook_installed")
    )


def collect_docker() -> tuple[dict[str, Any], CollectorStatus]:
    status = CollectorStatus(name="docker", ok=True)
    out: dict[str, Any] = {
        "docker_installed": which("docker") is not None,
        "nvidia_container_runtime_installed": which("nvidia-container-runtime") is not None,
        "nvidia_ctk_installed": which("nvidia-ctk") is not None,
        "nvidia_hook_installed": which("nvidia-container-runtime-hook") is not None,
        "cdi_specs_present": _cdi_specs_present(),
        "cdi_device_count": 0,
        "daemon_reachable": False,
        "socket_accessible": False,
        "nvidia_runtime_available": False,
        "gpu_container_running": False,
        "gpu_docker_ready": False,
        "runtimes": [],
        "containers": [],
    }

    # `nvidia-ctk cdi list` reports generated device specs even when the /etc/cdi
    # dir scan misses them; either signal means a usable CDI spec exists.
    if out["nvidia_ctk_installed"]:
        out["cdi_device_count"] = _cdi_list_device_count()
        if out["cdi_device_count"] > 0:
            out["cdi_specs_present"] = True

    if not out["docker_installed"]:
        status.errors.append("docker not installed")
        _finalize_gpu_readiness(out)
        return out, status

    ver = run(["docker", "version", "--format", "{{json .}}"], timeout=5)
    if ver.ok:
        out["daemon_reachable"] = True
        out["socket_accessible"] = True
        try:
            out["version"] = json.loads(ver.stdout)
        except json.JSONDecodeError:
            out["version_raw"] = ver.stdout.strip()
    else:
        status.errors.append(f"docker version: {ver.stderr.strip()[:200] or ver.error}")

    info = run(["docker", "info", "--format", "{{json .}}"], timeout=5)
    if info.ok:
        try:
            d = json.loads(info.stdout)
            runtimes = list((d.get("Runtimes") or {}).keys())
            out["runtimes"] = runtimes
            out["nvidia_runtime_available"] = any("nvidia" in r.lower() for r in runtimes)
            out["default_runtime"] = d.get("DefaultRuntime")
            out["server_version"] = d.get("ServerVersion")
        except json.JSONDecodeError:
            pass

    ps = run(["docker", "ps", "--format", "{{json .}}"], timeout=5)
    if ps.ok and ps.stdout.strip():
        containers: list[dict[str, Any]] = []
        for line in ps.stdout.strip().splitlines():
            try:
                containers.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        out["containers"] = containers

        for c in containers:
            cid = c.get("ID")
            if not cid:
                continue
            inspect = run(
                ["docker", "inspect", "--format", "{{json .HostConfig.DeviceRequests}}", cid],
                timeout=5,
            )
            if inspect.ok and inspect.stdout.strip() not in ("", "null"):
                try:
                    requests = json.loads(inspect.stdout)
                except json.JSONDecodeError:
                    requests = None
                if requests:
                    out["gpu_container_running"] = True
                    break

    _finalize_gpu_readiness(out)

    if status.errors and not out["daemon_reachable"]:
        status.ok = False
    return out, status
