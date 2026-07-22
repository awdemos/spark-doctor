from __future__ import annotations

import json
import re
from typing import Any

from ..models import CollectorStatus
from ..shell import run

# Runs inside the user's `python3`, not spark-doctor's venv deps: stdlib only,
# torch imported opportunistically. Must print a single JSON line and never exit nonzero.
_PY_PROBE = r"""
import json, platform, sys
info = {"python_version": platform.python_version(), "executable": sys.executable}
try:
    from importlib.metadata import PackageNotFoundError, version
    pkgs = {}
    for pkg in ("torch", "vllm", "triton", "flash-attn",
                "nvidia-cuda-runtime-cu12", "nvidia-cuda-runtime-cu13"):
        try:
            pkgs[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
        except Exception:
            pass
    info["packages"] = pkgs
except Exception:
    pass
try:
    import torch
    info["torch_import_ok"] = True
    info["torch_version"] = getattr(torch, "__version__", None)
    info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    try:
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception as e:
        info["torch_cuda_available_error"] = f"{type(e).__name__}: {e}"
    try:
        info["torch_arch_list"] = list(torch.cuda.get_arch_list())
    except Exception as e:
        info["torch_arch_list_error"] = f"{type(e).__name__}: {e}"
except Exception as e:
    info["torch_import_ok"] = False
    info["torch_import_error"] = f"{type(e).__name__}: {e}"

print(json.dumps(info))
"""

# flash-attn / bitsandbytes can SEGFAULT on import (bad .so against a mismatched
# CUDA), which would take down the whole probe. Run them in a SEPARATE subprocess
# so a crash never loses the torch info the core probe already collected.
_OPTIONAL_PKG_PROBE = r"""
import json
optional = {}
# flash_attn: no CPU build exists, so a successful import implies a CUDA build.
fa = {}
try:
    import flash_attn
    fa["import_ok"] = True
    fa["version"] = getattr(flash_attn, "__version__", None)
    fa["cuda_build"] = True
except Exception as e:
    fa["import_ok"] = False
    fa["import_error"] = f"{type(e).__name__}: {e}"
optional["flash_attn"] = fa
# bitsandbytes: ships a CPU-only fallback build; detect CUDA support best-effort.
bnb_info = {}
try:
    import bitsandbytes as bnb
    bnb_info["import_ok"] = True
    bnb_info["version"] = getattr(bnb, "__version__", None)
    cuda_flag = getattr(bnb, "COMPILED_WITH_CUDA", None)
    if cuda_flag is not None:
        bnb_info["cuda_build"] = bool(cuda_flag)
except Exception as e:
    bnb_info["import_ok"] = False
    bnb_info["import_error"] = f"{type(e).__name__}: {e}"
optional["bitsandbytes"] = bnb_info
print(json.dumps({"optional_gpu_packages": optional}))
"""


def parse_smi_cuda_version(text: str) -> str | None:
    m = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", text)
    return m.group(1) if m else None


def parse_nvcc_release(text: str) -> str | None:
    m = re.search(r"release\s+([0-9]+(?:\.[0-9]+)?)", text)
    return m.group(1) if m else None


def parse_ldconfig_libcudart(text: str) -> list[str]:
    sonames = {
        m.group(0)
        for line in text.splitlines()
        if (m := re.search(r"libcudart\.so\.[0-9]+", line))
    }
    return sorted(sonames)


def collect_cuda_env(python_executable: str = "python3") -> tuple[dict[str, Any], CollectorStatus]:
    status = CollectorStatus(name="cuda_env", ok=True)
    out: dict[str, Any] = {}

    smi = run(["nvidia-smi"], timeout=8)
    if smi.ok:
        out["driver_cuda_version"] = parse_smi_cuda_version(smi.stdout)
    elif smi.error != "command_not_found":
        status.errors.append(f"nvidia-smi: {smi.error}")

    nvcc = run(["nvcc", "--version"], timeout=8)
    if nvcc.ok:
        out["nvcc_release"] = parse_nvcc_release(nvcc.stdout)

    ld = run(["ldconfig", "-p"], timeout=8)
    if ld.ok:
        out["libcudart_sonames"] = parse_ldconfig_libcudart(ld.stdout)

    # First torch import on this platform can take tens of seconds.
    probe = run([python_executable, "-c", _PY_PROBE], timeout=60)
    if probe.ok and probe.stdout.strip():
        try:
            out["python"] = json.loads(probe.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            status.errors.append("python probe: unparseable output")
    elif probe.error == "command_not_found":
        status.errors.append(f"python probe: {python_executable} not found")
    elif probe.error:
        status.errors.append(f"python probe: {probe.error}")

    # Isolated probe: importing flash-attn/bitsandbytes can segfault. Run it in a
    # separate process so a crash here can't lose the torch info above. Only merge
    # if the core probe produced a python section to attach to.
    if isinstance(out.get("python"), dict):
        opt = run([python_executable, "-c", _OPTIONAL_PKG_PROBE], timeout=60)
        if opt.ok and opt.stdout.strip():
            try:
                parsed = json.loads(opt.stdout.strip().splitlines()[-1])
                if isinstance(parsed.get("optional_gpu_packages"), dict):
                    out["python"]["optional_gpu_packages"] = parsed["optional_gpu_packages"]
            except (json.JSONDecodeError, IndexError):
                status.errors.append("optional-package probe: unparseable output")
        elif opt.error and opt.error != "command_not_found":
            status.errors.append(f"optional-package probe: {opt.error}")

    status.ok = bool(out) or not status.errors
    return out, status
