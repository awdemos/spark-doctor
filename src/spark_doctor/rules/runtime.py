from __future__ import annotations

from ..models import Finding, ScanReport
from .engine import Rule


def _evaluate(report: ScanReport) -> list[Finding]:
    d = report.docker or {}
    # If no docker data collected at all, skip
    if not d:
        return []

    problems: list[str] = []
    fix_commands: list[str] = []
    if not d.get("docker_installed"):
        problems.append("Docker is not installed or not on PATH.")
        fix_commands.extend(
            [
                "sudo apt-get update",
                "sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin",
                "sudo systemctl enable --now docker",
            ]
        )
    if d.get("docker_installed") and not d.get("daemon_reachable"):
        problems.append("Docker daemon is not reachable (socket or permissions).")
        fix_commands.extend(
            [
                "sudo systemctl daemon-reload",
                "sudo systemctl enable --now docker",
            ]
        )
    if d.get("docker_installed") and d.get("daemon_reachable") is False and d.get("socket_accessible") is False:
        problems.append("Current user cannot access the Docker socket.")
        fix_commands.extend(
            [
                "sudo usermod -aG docker $USER",
                "newgrp docker  # or log out and back in",
            ]
        )
    if not (d.get("nvidia_container_runtime_installed") or d.get("nvidia_ctk_installed")):
        problems.append("nvidia-container-runtime / nvidia-ctk not found.")
        fix_commands.extend(
            [
                "sudo apt-get update",
                "sudo apt-get install -y nvidia-container-toolkit",
            ]
        )
    if d.get("daemon_reachable") and not d.get("nvidia_runtime_available"):
        problems.append("NVIDIA runtime is not registered with Docker.")
        fix_commands.extend(
            [
                "sudo nvidia-ctk runtime configure --runtime=docker",
                "sudo systemctl daemon-reload",
                "sudo systemctl restart docker",
            ]
        )

    if not problems:
        return []

    return [
        Finding(
            rule_id="runtime.docker_unhealthy",
            title="Docker or NVIDIA container runtime not ready",
            severity="warning",
            confidence="high",
            evidence=problems,
            explanation=(
                "One or more Docker or NVIDIA container runtime pieces are missing or "
                "misconfigured. Recipes that depend on GPU containers will not run until "
                "this is fixed."
            ),
            recommended_actions=[
                "Install Docker if missing and start the daemon.",
                "Add your user to the `docker` group and re-login if socket access is denied.",
                "Install `nvidia-container-toolkit` and register the NVIDIA runtime.",
                "Re-run `spark-doctor scan` once the runtime is installed.",
            ],
            fix_commands=fix_commands,
            source_note="Standard NVIDIA container runtime setup.",
        )
    ]


rule_runtime_docker_unhealthy = Rule(
    id="runtime.docker_unhealthy",
    title="Docker or NVIDIA container runtime not ready",
    fn=_evaluate,
)
