from spark_doctor.models import ProcessInfo, ScanReport
from spark_doctor.rules.backend import rule_backend_multiple_heavy_models


def _proc(pid: int, backend: str | None, rss_gb: float) -> ProcessInfo:
    return ProcessInfo(pid=pid, command="python3", rss_kb=int(rss_gb * 1024 * 1024), detected_backend=backend)


def test_two_heavy_backends_warn() -> None:
    report = ScanReport(processes=[_proc(1, "vllm", 40), _proc(2, "ollama", 20)])
    finding = rule_backend_multiple_heavy_models.fn(report)[0]
    assert finding.severity == "warning"
    assert any("Total heavy backend RSS: 60.0 GB" in line for line in finding.evidence)


def test_multiple_processes_of_one_backend_do_not_count_twice() -> None:
    report = ScanReport(processes=[_proc(1, "vllm", 40), _proc(2, "vllm", 30)])
    assert rule_backend_multiple_heavy_models.fn(report) == []


def test_small_or_non_heavy_processes_are_ignored() -> None:
    report = ScanReport(
        processes=[_proc(1, "vllm", 40), _proc(2, "ollama", 0.5), _proc(3, "open-webui", 4)]
    )
    assert rule_backend_multiple_heavy_models.fn(report) == []
