from __future__ import annotations

import math

from ..models import CollectorStatus, MemorySnapshot
from ..shell import read_text


def _parse_meminfo(text: str) -> dict[str, int]:
    data: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        v = v.strip()
        parts = v.split()
        if not parts:
            continue
        try:
            value = int(parts[0])
            if value >= 0:
                data[k.strip()] = value
        except ValueError:
            continue
    return data


def _parse_psi(text: str, errors: list[str] | None = None) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        scope = parts[0]
        values: dict[str, float] = {}
        for p in parts[1:]:
            if "=" in p:
                k, _, v = p.partition("=")
                try:
                    value = float(v)
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("invalid pressure value")
                    values[k] = value
                except ValueError:
                    if errors is not None:
                        errors.append(f"PSI {scope}.{k}: invalid value")
                    continue
        if values:
            out[scope] = values
    return out


def collect_memory() -> tuple[MemorySnapshot, CollectorStatus]:
    status = CollectorStatus(name="memory", ok=True)
    snap = MemorySnapshot()

    mi = read_text("/proc/meminfo")
    if mi:
        d = _parse_meminfo(mi)
        snap.mem_total_kb = d.get("MemTotal")
        snap.mem_available_kb = d.get("MemAvailable")
        snap.swap_total_kb = d.get("SwapTotal")
        snap.swap_free_kb = d.get("SwapFree")
        for field in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            if field not in d or d[field] < 0:
                status.errors.append(f"/proc/meminfo: missing or invalid {field}")
    else:
        status.errors.append("missing:/proc/meminfo")

    psi_mem = read_text("/proc/pressure/memory")
    if psi_mem:
        snap.psi_memory = _parse_psi(psi_mem, status.errors)
    if any("avg10" not in snap.psi_memory.get(scope, {}) for scope in ("some", "full")):
        status.errors.append("missing or invalid:/proc/pressure/memory")

    psi_io = read_text("/proc/pressure/io")
    if psi_io:
        snap.psi_io = _parse_psi(psi_io, status.errors)
    if any("avg10" not in snap.psi_io.get(scope, {}) for scope in ("some", "full")):
        status.errors.append("missing or invalid:/proc/pressure/io")

    if status.errors and snap.mem_total_kb is None:
        status.ok = False
    return snap, status
