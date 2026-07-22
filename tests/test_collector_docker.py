from spark_doctor.collectors.docker_runtime import _finalize_gpu_readiness


def _ready(**keys) -> bool:
    out: dict = {}
    out.update(keys)
    _finalize_gpu_readiness(out)
    return out["gpu_docker_ready"]


def test_ready_via_named_runtime():
    assert _ready(nvidia_runtime_available=True) is True


def test_ready_via_hook_main_field_name():
    # Reads main's key `nvidia_hook_installed`, not the branch's old name.
    assert _ready(nvidia_hook_installed=True) is True


def test_ready_via_cdi_main_field_name():
    assert _ready(cdi_specs_present=True) is True


def test_not_ready_when_no_path():
    assert _ready(
        nvidia_runtime_available=False,
        nvidia_hook_installed=False,
        cdi_specs_present=False,
    ) is False


def test_not_ready_ignores_stale_branch_keys():
    # Old branch keys must NOT satisfy readiness on their own.
    assert _ready(nvidia_runtime_hook_present=True, cdi_available=True) is False
