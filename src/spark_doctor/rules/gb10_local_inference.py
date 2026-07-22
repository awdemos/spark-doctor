from __future__ import annotations

import re
from typing import Any

from ..models import Finding, ScanReport
from .engine import Rule

# GB10-specific local-inference checks accumulated from real serving sessions.


def _is_gb10(report: ScanReport) -> bool:
    name = str(report.gpu.get("name", "")) if isinstance(report.gpu, dict) else ""
    if "GB10" in name:
        return True
    gpus = report.gpu.get("gpus") if isinstance(report.gpu, dict) else None
    if isinstance(gpus, list):
        return any("GB10" in str(g.get("name", "")) for g in gpus if isinstance(g, dict))
    return False


def _python_info(report: ScanReport) -> dict[str, Any]:
    env = report.cuda_env if isinstance(report.cuda_env, dict) else {}
    py = env.get("python")
    return py if isinstance(py, dict) else {}


# ---------------------------------------------------------------------------
# vLLM 0.22.1 `--reasoning-parser nemotron_v3` discards primed reasoning.
# ---------------------------------------------------------------------------
def _vllm_version(report: ScanReport) -> str | None:
    py = _python_info(report)
    pkgs = py.get("packages")
    if isinstance(pkgs, dict):
        v = pkgs.get("vllm")
        if isinstance(v, str):
            return v
    return None


def _eval_nemotron_v3_parser(report: ScanReport) -> list[Finding]:
    procs = []
    for p in report.processes:
        args = p.args or ""
        if "--reasoning-parser" in args and re.search(
            r"--reasoning-parser[=\s]+nemotron_v3", args
        ):
            procs.append(p)
    if not procs:
        return []

    version = _vllm_version(report)
    if version != "0.22.1":
        # Heuristic is only known to apply to 0.22.1; skip otherwise.
        return []

    evidence = [f"vLLM version {version} detected"]
    for p in procs[:3]:
        evidence.append(f"pid {p.pid}: process cmdline contains --reasoning-parser nemotron_v3")

    return [
        Finding(
            rule_id="backend.nemotron_v3_discards_primed_reasoning",
            title="vLLM nemotron_v3 reasoning-parser silently discards primed reasoning",
            severity="warning",
            confidence="medium",
            evidence=evidence,
            explanation=(
                "vLLM 0.22.1's `--reasoning-parser nemotron_v3` discards reasoning that was "
                "primed in the prompt (an opening `<think>` in the template/prompt). If your "
                "chat template opens a reasoning block, that content is dropped from the "
                "parsed output. This is a heuristic: the template-priming half is not "
                "observable from the scan."
            ),
            recommended_actions=[
                "Verify whether your chat template primes an opening `<think>` tag.",
                "If it does, avoid nemotron_v3 on 0.22.1 or upgrade vLLM once fixed.",
            ],
            source_note="Observed on vLLM 0.22.1; heuristic (prompt-priming not directly observable).",
        )
    ]


# ---------------------------------------------------------------------------
# P5: aarch64 + Blackwell prebuilt-wheel gap (flash_attn / bitsandbytes).
# ---------------------------------------------------------------------------
def _eval_aarch64_wheel_gap(report: ScanReport) -> list[Finding]:
    arch = ""
    if isinstance(report.os, dict):
        arch = str(report.os.get("arch", "")).lower()
    if arch not in ("aarch64", "arm64") or not _is_gb10(report):
        return []

    py = _python_info(report)
    optional = py.get("optional_gpu_packages")
    if not isinstance(optional, dict):
        return []

    evidence: list[str] = []
    for name, entry in optional.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("import_ok") is False:
            err = str(entry.get("import_error", "")).strip()
            evidence.append(f"{name}: not importable ({err[:120]})")
        elif entry.get("import_ok") is True and entry.get("cuda_build") is False:
            evidence.append(
                f"{name}: imported but reports no CUDA build (CPU-only) "
                f"[{entry.get('version', '?')}]"
            )
    if not evidence:
        return []

    return [
        Finding(
            rule_id="cuda.aarch64_prebuilt_wheel_gap",
            title="GPU acceleration packages missing or CPU-only on aarch64 + GB10",
            severity="info",
            confidence="low",
            evidence=evidence,
            explanation=(
                "On aarch64 + Blackwell (sm_121) there is often no prebuilt wheel for "
                "flash-attn / bitsandbytes, so `pip install` either fails to find one or "
                "installs a CPU-only build. This is informational — many local-inference "
                "workflows (vLLM, Ollama, llama.cpp) do not need these packages."
            ),
            recommended_actions=[
                "If a workload needs them, build from source against CUDA 13 / sm_121, "
                "or use an NGC container that bundles a matching build.",
                "Otherwise ignore — these are not required for most serving setups.",
            ],
            source_note="aarch64 + sm_121 frequently lacks prebuilt flash-attn/bitsandbytes wheels.",
        )
    ]


# One registered rule per check: an exception in one raises a single
# rule.error.<id> without silently skipping the others.
rule_gb10_nemotron_v3_parser = Rule(
    id="gb10.nemotron_v3_parser",
    title="vLLM nemotron_v3 reasoning-parser discards primed reasoning",
    fn=_eval_nemotron_v3_parser,
)

rule_gb10_aarch64_wheel_gap = Rule(
    id="gb10.aarch64_wheel_gap",
    title="aarch64 + GB10 prebuilt-wheel gap (flash-attn / bitsandbytes)",
    fn=_eval_aarch64_wheel_gap,
)
