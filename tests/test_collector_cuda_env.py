from __future__ import annotations

import json

import pytest

from spark_doctor.collectors import cuda_env
from spark_doctor.shell import ShellResult


@pytest.mark.parametrize(
    ("stdout", "error"),
    [
        ("", "nonzero_exit"),
        ("", "timeout"),
        ("", "command_not_found"),
        ("", None),
        ("not json", None),
        ("null", None),
        ("[]", None),
        ('{"optional_gpu_packages": []}', None),
    ],
)
def test_optional_probe_failure_preserves_core_data(
    monkeypatch: pytest.MonkeyPatch, stdout: str, error: str | None
) -> None:
    core = {"torch_import_ok": True, "torch_cuda_version": "13.0"}
    probes: list[str] = []

    def fake_run(args: list[str], *, timeout: float) -> ShellResult:
        if args[0] == "test-python":
            assert timeout == 60
            probes.append(args[2])
            if args[2] == cuda_env._PY_PROBE:
                return ShellResult("python", True, 0, json.dumps(core), "")
            assert args[2] == cuda_env._OPTIONAL_PKG_PROBE
            return ShellResult("python", error is None, -11 if error else 0, stdout, "", error)
        return ShellResult(args[0], False, None, "", "", "command_not_found")

    monkeypatch.setattr(cuda_env, "run", fake_run)
    data, status = cuda_env.collect_cuda_env("test-python")
    assert data["python"] == core
    assert status.ok
    assert any("optional-package probe:" in e for e in status.errors)
    assert probes == [cuda_env._PY_PROBE, cuda_env._OPTIONAL_PKG_PROBE]


def test_optional_probe_merges_package_results(monkeypatch: pytest.MonkeyPatch) -> None:
    optional = {"flash_attn": {"import_ok": False, "import_error": "not installed"}}

    def fake_run(args: list[str], *, timeout: float) -> ShellResult:
        if args[0] != "test-python":
            return ShellResult(args[0], False, None, "", "", "command_not_found")
        result = (
            {"torch_import_ok": True}
            if args[2] == cuda_env._PY_PROBE
            else {"optional_gpu_packages": optional}
        )
        return ShellResult("python", True, 0, "import notice\n" + json.dumps(result), "")

    monkeypatch.setattr(cuda_env, "run", fake_run)
    data, status = cuda_env.collect_cuda_env("test-python")
    assert data["python"]["torch_import_ok"] is True
    assert data["python"]["optional_gpu_packages"] == optional
    assert status.errors == []
