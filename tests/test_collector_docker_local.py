from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from spark_doctor import shell
from spark_doctor.collectors import docker_runtime


@pytest.fixture
def docker_boundary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[list[Any], dict[str, Any]]:
    for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH", "XDG_RUNTIME_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path))
    monkeypatch.setattr(docker_runtime, "which", lambda name: "/bin/docker" if name == "docker" else None)
    monkeypatch.setattr(shell, "which", lambda name: "/bin/" + name)
    monkeypatch.setattr(docker_runtime, "_cdi_specs_present", lambda: False)
    calls = []
    state = {"context_host": "unix:///run/user/1234/docker.sock", "errors": {}}

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert isinstance(args, list), "Collector must not invoke a shell"
        calls.append((args, kwargs))
        if args[1:3] == ["context", "inspect"]:
            return subprocess.CompletedProcess(args, 0, json.dumps(state["context_host"]), "")
        operation = next((op for op in ("version", "info", "ps", "inspect") if op in args), None)
        if operation in state["errors"]:
            payload = state["errors"][operation]
            if isinstance(payload, Exception):
                raise payload
            return subprocess.CompletedProcess(args, 0, payload, "")
        output = {"version": '{"Server":{"Version":"1"}}', "info": '{"Runtimes":{"nvidia":{}}}', "ps": '{"ID":"local-container"}', "inspect": '[]'}[operation]
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(shell.subprocess, "run", run)
    return calls, state


@pytest.mark.parametrize("endpoint", ["tcp://remote:2375", "ssh://user@remote", "unix://remote/path", "unix://relative", "http://remote"])
def test_remote_host_is_refused_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch, docker_boundary: tuple[list[Any], dict[str, Any]], endpoint: str
) -> None:
    calls, _ = docker_boundary
    monkeypatch.setenv("DOCKER_HOST", endpoint)
    data, status = docker_runtime.collect_docker()
    assert calls == []
    assert not status.ok
    assert status.errors
    assert data["daemon_reachable"] is False
    assert data["daemon_check_skipped"] is True


@pytest.mark.parametrize("configured", [False, True])
def test_remote_context_only_reads_offline_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    docker_boundary: tuple[list[Any], dict[str, Any]], configured: bool
) -> None:
    calls, state = docker_boundary
    state["context_host"] = "ssh://remote"
    if configured:
        (tmp_path / "config.json").write_text('{"currentContext":"remote"}')
    else:
        monkeypatch.setenv("DOCKER_CONTEXT", "remote")
    _, status = docker_runtime.collect_docker()
    assert len(calls) == 1
    assert calls[0][0][1:3] == ["context", "inspect"]
    assert not status.ok and status.errors


@pytest.mark.parametrize("source", ["default", "host", "context"])
def test_local_endpoint_forced_on_every_daemon_command(
    monkeypatch: pytest.MonkeyPatch, docker_boundary: tuple[list[Any], dict[str, Any]], source: str
) -> None:
    calls, _ = docker_boundary
    endpoint = "unix:///var/run/docker.sock"
    if source == "host":
        endpoint = "unix:///run/user/1234/docker.sock"
        monkeypatch.setenv("DOCKER_HOST", endpoint)
    elif source == "context":
        endpoint = "unix:///run/user/1234/docker.sock"
        monkeypatch.setenv("DOCKER_CONTEXT", "rootless")
        monkeypatch.setenv("DOCKER_HOST", "ssh://ignored-remote")
    for key in ("DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        monkeypatch.setenv(key, "ignored-transport-state")
    data, status = docker_runtime.collect_docker()
    assert status.ok and not status.errors
    assert data["endpoint"] == endpoint
    assert data["gpu_docker_ready"] is True
    daemon_calls = [call for call in calls if call[0][1:3] != ["context", "inspect"]]
    assert len(daemon_calls) == 4
    for args, kwargs in daemon_calls:
        assert args[:3] == ["docker", "--host", endpoint]
        assert not any(key in kwargs["env"] for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"))


@pytest.mark.parametrize(("operation", "payload"), [("info", "[]"), ("ps", "null"), ("inspect", "invalid"), ("version", "not json"), ("info", subprocess.TimeoutExpired("docker", 5))])
def test_failed_or_malformed_docker_probe_is_recorded(
    docker_boundary: tuple[list[Any], dict[str, Any]], operation: str, payload: str | Exception
) -> None:
    _, state = docker_boundary
    state["errors"][operation] = payload
    _, status = docker_runtime.collect_docker()
    assert any(operation in error for error in status.errors)
