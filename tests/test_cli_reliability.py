from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from spark_doctor import cli
from spark_doctor.models import CollectorStatus, Finding, MemorySnapshot, ProcessInfo, ScanReport

runner = CliRunner()


@pytest.fixture
def collectors(monkeypatch: pytest.MonkeyPatch) -> None:
    results = {
        "os": {"arch": "aarch64"}, "firmware": {}, "cuda_env": {},
        "docker": {}, "network": {}, "processes": [], "logs": {},
        "memory": MemorySnapshot(mem_total_kb=128_000_000, mem_available_kb=90_000_000),
    }
    for name, value in results.items():
        monkeypatch.setattr(cli, f"collect_{name}", lambda value=value, name=name, **kw: (
            value, CollectorStatus(name=name, ok=True)
        ))
    monkeypatch.setattr(cli, "collect_gpu", lambda **kw: (
        {"available": True}, [], CollectorStatus(name="gpu", ok=True)
    ))


@pytest.mark.parametrize("ok", [True, False])
def test_scan_incomplete_exits_three(collectors: None, monkeypatch: pytest.MonkeyPatch, ok: bool) -> None:
    monkeypatch.setattr(cli, "collect_gpu", lambda **kw: (
        {}, [], CollectorStatus(name="gpu", ok=ok, errors=["sampling timed out"])
    ))
    result = runner.invoke(cli.app, ["scan", "--no-save"])
    assert result.exit_code == 3
    assert "Incomplete" in result.stdout
    assert "sampling timed out" in result.stdout
    assert "Overall: OK" not in result.stdout


def test_scan_collector_exception_does_not_abort(collectors: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable() -> None:
        raise PermissionError("denied")
    monkeypatch.setattr(cli, "collect_memory", unavailable)
    result = runner.invoke(cli.app, ["scan", "--no-save"])
    assert result.exit_code == 3
    assert "Incomplete" in result.stdout
    assert "PermissionError" in result.stdout


def test_rule_error_is_incomplete() -> None:
    report = ScanReport(findings=[Finding(rule_id="rule.error.example", title="Rule failed", severity="info")])
    assert cli._exit_code_for(report) == 3


def test_incomplete_takes_precedence_over_critical() -> None:
    report = ScanReport(
        collector_statuses=[CollectorStatus(name="gpu", ok=False)],
        findings=[Finding(rule_id="memory", title="Memory pressure", severity="critical")],
    )
    assert cli._exit_code_for(report) == 3


def test_selected_python_reaches_collector_and_console(
    collectors: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def probe(python_executable: str) -> tuple[dict, CollectorStatus]:
        return {"python": {"executable": python_executable, "torch_import_ok": False}}, CollectorStatus(name="cuda_env", ok=True)
    monkeypatch.setattr(cli, "collect_cuda_env", probe)
    result = runner.invoke(cli.app, ["scan", "--no-save", "--python", "/opt/workload/bin/python"])
    assert result.exit_code == 0
    assert "/opt/workload/bin/python" in result.stdout
    assert "Selected Python environment" in result.stdout


@pytest.mark.parametrize("payload", [
    "{", "[]", '{"memory":{"psi_memory":{"full":{"avg10":"bad"}}}}',
    '{"os":{"os_release": "not a mapping"}}', '{"gpu":{"peak":"invalid"}}',
    '{"network":{"interfaces":["invalid"]}}', '{"gpu_samples":[{"gpu_power_draw_watts":NaN}]}',
    '{"docker":{"containers":1}}', '{"docker":{"containers":["invalid"]}}',
    '{"gpu":{"peak":{"gpu_power_draw_watts":"[/bogus]"}}}',
])
@pytest.mark.parametrize("command", ["doctor", "report", "anonymize"])
def test_invalid_scan_has_readable_error(tmp_path: Path, payload: str, command: str) -> None:
    path = tmp_path / "scan.json"
    path.write_text(payload)
    args = [command, str(path), "--out", str(tmp_path / "out.json")] if command == "anonymize" else [command, "--from", str(path)]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 3
    assert "Invalid scan" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_recipe_cli_rejects_nonfinite_memory(value: str) -> None:
    result = runner.invoke(cli.app, [
        "recipe", "check", str(Path(__file__).parent / "fixtures" / "recipe_ok.yaml"),
        "--mem-available-gb", value,
    ])
    assert result.exit_code == 2
    assert "finite" in result.output or "range" in result.output


def test_numeric_psi_string_cannot_hide_low_memory(tmp_path: Path) -> None:
    path = tmp_path / "scan.json"
    path.write_text(json.dumps({"memory": {
        "mem_total_kb": 128_000_000, "mem_available_kb": 1_000_000,
        "psi_memory": {"full": {"avg10": "0.5"}},
    }}))
    result = runner.invoke(cli.app, ["doctor", "--from", str(path)])
    assert result.exit_code == 2
    assert "Unified memory pressure elevated" in result.stdout


def test_invalid_recipe_has_readable_error(tmp_path: Path) -> None:
    path = tmp_path / "recipe.yaml"
    path.write_text("name: broken\nbackend: vllm\nmodel: example\nruntime:\n  tensor_parallel_size: -1\n")
    result = runner.invoke(cli.app, ["recipe", "check", str(path)])
    assert result.exit_code == 2
    assert "Invalid recipe" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command", ["doctor", "report"])
def test_loaded_reports_are_redacted_by_default(tmp_path: Path, command: str) -> None:
    secret = "hf_abcdefghijklmnopqrstuvwxyz123456"
    path = tmp_path / "raw.json"
    path.write_text(ScanReport(anonymized=False, logs={
        "journal_tail": f"No available memory for the cache blocks; {secret}",
    }).model_dump_json())
    result = runner.invoke(cli.app, [command, "--from", str(path)], env={"COLUMNS": "240"})
    assert result.exit_code in (0, 2)
    assert secret not in result.stdout
    assert "<redacted:" in result.stdout
    raw = runner.invoke(cli.app, [command, "--from", str(path), "--include-sensitive-data"], env={"COLUMNS": "240"})
    assert secret in raw.stdout


def test_default_scan_json_removes_argument_secret(
    collectors: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    secret = "canaryServingSecret123456"
    monkeypatch.setattr(cli, "collect_processes", lambda: (
        [ProcessInfo(pid=1, command="vllm", args=f"vllm serve toy --api-key {secret}")],
        CollectorStatus(name="processes", ok=True),
    ))
    output = tmp_path / "scan.json"
    result = runner.invoke(cli.app, ["scan", "--no-save", "--json", str(output)])
    assert result.exit_code == 0
    saved = output.read_text()
    assert secret not in saved
    assert json.loads(saved)["anonymized"] is True
    raw = runner.invoke(cli.app, ["scan", "--no-save", "--json", str(output), "--include-sensitive-data"])
    assert raw.exit_code == 0
    assert secret in output.read_text()
    assert json.loads(output.read_text())["anonymized"] is False


def test_scan_requires_include_flag_to_disable_redaction(collectors: None) -> None:
    result = runner.invoke(cli.app, ["scan", "--no-save", "--no-anonymize"])
    assert result.exit_code == 2
    assert "No such option" in result.output


@pytest.mark.parametrize("command", ["doctor", "report", "anonymize"])
def test_bad_input_never_echoes_secret(tmp_path: Path, command: str) -> None:
    secret = "hf_abcdefghijklmnopqrstuvwxyz123456"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"memory": {"mem_total_kb": secret}}))
    args = [command, str(path), "--out", str(tmp_path / "redacted.json")] if command == "anonymize" else [command, "--from", str(path)]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 3
    assert "Invalid scan" in result.output
    assert secret not in result.output


def test_report_network_identifiers_need_explicit_flag(tmp_path: Path) -> None:
    path = tmp_path / "raw.json"
    path.write_text(ScanReport(anonymized=False, reproduction_notes="server 192.168.1.22").model_dump_json())
    normal = runner.invoke(cli.app, ["report", "--from", str(path)])
    assert "192.168.1.22" not in normal.stdout
    included = runner.invoke(cli.app, ["report", "--from", str(path), "--include-network-identifiers"])
    assert "192.168.1.22" in included.stdout


def test_failed_python_probe_still_shows_selected_interpreter(
    collectors: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "collect_cuda_env", lambda python_executable: (
        {"python_executable": python_executable},
        CollectorStatus(name="cuda_env", ok=False, errors=["python probe: not found"]),
    ))
    result = runner.invoke(cli.app, ["scan", "--no-save", "--python", "/missing/python"])
    assert result.exit_code == 3
    assert "/missing/python" in result.stdout
    assert "was not assessed" in result.stdout


def test_cpu_only_torch_doctor_exits_warning() -> None:
    result = runner.invoke(cli.app, [
        "doctor", "--from", str(Path(__file__).parent / "fixtures" / "cuda_torch_cpu_only.json"),
    ])
    assert result.exit_code == 1
    assert "CPU-only PyTorch" in result.stdout
    assert "separate GPU container" in result.stdout


@pytest.mark.parametrize("flags,keep_network,keep_hardware", [
    ([], False, False), (["--include-network-identifiers"], True, False),
    (["--include-sensitive-data"], True, True),
])
def test_scan_identifier_privacy_flags_are_independent(
    collectors: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    flags: list[str], keep_network: bool, keep_hardware: bool,
) -> None:
    address = "2001:db8::1"
    serial = "TEST-SERIAL-1234"
    uuid = "GPU-00000000-1111-2222-3333-444444444444"
    monkeypatch.setattr(cli, "collect_network", lambda: (
        {"ip_br_addr": f"eth0 UP {address}/64"}, CollectorStatus(name="network", ok=True),
    ))
    monkeypatch.setattr(cli, "collect_firmware", lambda **kw: (
        {"fwupdmgr": f"│ │ Serial Number: {serial}"}, CollectorStatus(name="firmware", ok=True),
    ))
    monkeypatch.setattr(cli, "collect_gpu", lambda **kw: (
        {"gpus": [{"uuid": uuid}]}, [], CollectorStatus(name="gpu", ok=True),
    ))
    path = tmp_path / "scan.json"
    result = runner.invoke(cli.app, ["scan", "--no-save", "--json", str(path), *flags])
    assert result.exit_code == 0
    output = path.read_text()
    assert (address in output) is keep_network
    assert (serial in output) is keep_hardware
    assert (uuid in output) is keep_hardware
    assert json.loads(output)["anonymized"] is not keep_hardware


@pytest.mark.parametrize("format", ["markdown", "forum", "github"])
@pytest.mark.parametrize("flags,keep_network,keep_hardware", [
    ([], False, False), (["--include-network-identifiers"], True, False),
    (["--include-sensitive-data"], True, True),
])
def test_imported_report_exports_respect_identifier_privacy_flags(
    tmp_path: Path, format: str, flags: list[str], keep_network: bool, keep_hardware: bool,
) -> None:
    address = "2001:db8::1"
    serial = "TEST-SERIAL-1234"
    uuid = "GPU-00000000-1111-2222-3333-444444444444"
    path = tmp_path / "raw.json"
    path.write_text(ScanReport(anonymized=False, reproduction_notes=(
        f"peer {address}: connection refused\npeer {address}:51234\n"
        f"Serial Number: {serial}\nselected {uuid}"
    )).model_dump_json())
    result = runner.invoke(cli.app, ["report", "--from", str(path), "--format", format, *flags])
    assert result.exit_code == 0
    assert (address in result.stdout) is keep_network
    assert (serial in result.stdout) is keep_hardware
    assert (uuid in result.stdout) is keep_hardware


KV_OOM_LOG = "ValueError: No available memory for the cache blocks."


def _logs(monkeypatch: pytest.MonkeyPatch, *, ok: bool = True, calls: list[str] | None = None) -> None:
    def fake() -> tuple[dict[str, str], CollectorStatus]:
        if calls is not None:
            calls.append("logs")
        errors = [] if ok else ["dmesg: nonzero_exit: Operation not permitted"]
        return ({"journal_tail": KV_OOM_LOG} if ok else {}), CollectorStatus(name="logs", ok=ok, errors=errors)
    monkeypatch.setattr(cli, "collect_logs", fake)


def test_default_scan_runs_log_rules_without_shipping_logs(
    collectors: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _logs(monkeypatch)
    output = tmp_path / "scan.json"
    result = runner.invoke(cli.app, ["scan", "--no-save", "--json", str(output)])
    saved = json.loads(output.read_text())
    assert result.exit_code == 2
    assert any(f["rule_id"] == "backend.kv_cache_oom" for f in saved["findings"])
    assert saved["logs"] == {}


def test_include_logs_keeps_log_text(collectors: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _logs(monkeypatch)
    output = tmp_path / "scan.json"
    runner.invoke(cli.app, ["scan", "--no-save", "--include-logs", "--json", str(output)])
    assert json.loads(output.read_text())["logs"] == {"journal_tail": KV_OOM_LOG}


def test_no_logs_skips_reading(collectors: None, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _logs(monkeypatch, calls=calls)
    result = runner.invoke(cli.app, ["scan", "--no-save", "--no-logs", "--include-logs"])
    assert calls == []
    assert result.exit_code == 0


def test_unreadable_logs_only_fail_scan_when_requested(collectors: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _logs(monkeypatch, ok=False)
    default = runner.invoke(cli.app, ["scan", "--no-save"], env={"COLUMNS": "240"})
    assert default.exit_code == 0
    assert "Operation not permitted" in default.stdout
    requested = runner.invoke(cli.app, ["scan", "--no-save", "--include-logs"])
    assert requested.exit_code == 3
