import json
from pathlib import Path

from spark_doctor.collectors.cuda_env import (
    parse_ldconfig_libcudart,
    parse_nvcc_release,
    parse_smi_cuda_version,
)
from spark_doctor.models import ScanReport
from spark_doctor.rules import run_rules

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> ScanReport:
    data = json.loads((FIXTURES / name).read_text())
    return ScanReport.model_validate(data)


def _cuda_findings(report: ScanReport) -> dict:
    return {f.rule_id: f for f in run_rules(report) if f.rule_id.startswith("cuda.")}


def test_cu12_wheel_on_cuda13_is_critical():
    ids = _cuda_findings(_load("cuda_wheel_mismatch.json"))
    assert "cuda.torch_cu12_wheel" in ids
    assert ids["cuda.torch_cu12_wheel"].severity == "critical"


def test_cu12_wheel_fixture_also_flags_nvcc_and_arch():
    ids = _cuda_findings(_load("cuda_wheel_mismatch.json"))
    assert "cuda.nvcc_toolkit_mismatch" in ids
    assert ids["cuda.nvcc_toolkit_mismatch"].severity == "warning"
    assert "cuda.sm121_not_in_arch_list" in ids


def test_libcudart_import_failure_is_critical():
    ids = _cuda_findings(_load("cuda_libcudart_failure.json"))
    assert "cuda.libcudart_missing" in ids
    assert ids["cuda.libcudart_missing"].severity == "critical"
    assert "cuda.torch_cu12_wheel" not in ids


def test_healthy_cuda13_env_no_findings():
    assert _cuda_findings(_load("cuda_env_healthy.json")) == {}


def test_missing_cuda_env_stays_silent():
    assert _cuda_findings(_load("healthy_minimal.json")) == {}


def test_parse_smi_cuda_version():
    banner = "| NVIDIA-SMI 580.95.05    Driver Version: 580.95.05    CUDA Version: 13.0 |"
    assert parse_smi_cuda_version(banner) == "13.0"
    assert parse_smi_cuda_version("no match here") is None


def test_parse_nvcc_release():
    text = "Cuda compilation tools, release 13.0, V13.0.48"
    assert parse_nvcc_release(text) == "13.0"


def test_parse_ldconfig_libcudart():
    text = (
        "\tlibcudart.so.13 (libc6,AArch64) => /usr/local/cuda/lib64/libcudart.so.13\n"
        "\tlibcudart.so.13 (libc6,AArch64) => /opt/other/libcudart.so.13\n"
        "\tlibcuda.so.1 (libc6,AArch64) => /usr/lib/libcuda.so.1\n"
    )
    assert parse_ldconfig_libcudart(text) == ["libcudart.so.13"]
