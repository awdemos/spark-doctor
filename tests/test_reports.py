import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from spark_doctor.models import ScanReport
from spark_doctor.reports import render_console, render_forum, render_github, render_markdown
from spark_doctor.rules import run_rules

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> ScanReport:
    data = json.loads((FIXTURES / name).read_text())
    r = ScanReport.model_validate(data)
    r.findings = run_rules(r)
    return r


def test_markdown_contains_finding_title():
    r = _load("power_limited_14w.json")
    md = render_markdown(r)
    assert "Spark Doctor Report" in md
    assert "Possible GPU low-power state" in md
    assert "Next steps" in md


def test_forum_contains_title_and_tldr():
    r = _load("power_limited_14w.json")
    txt = render_forum(r)
    assert "Title suggestion" in txt
    assert "TL;DR" in txt
    assert "CRITICAL" in txt


def test_github_has_env_section():
    r = _load("power_limited_14w.json")
    txt = render_github(r)
    assert "Environment" in txt
    assert "Steps to reproduce" in txt


@pytest.mark.parametrize("render", [render_markdown, render_forum, render_github])
def test_reports_include_suggested_commands(render):
    report = _load("docker_no_runtime_no_hook_no_cdi.json")
    text = render(report)
    assert "Suggested commands (review before running):" in text
    assert (
        "```bash\nsudo nvidia-ctk runtime configure --runtime=docker\n"
        "sudo systemctl restart docker\n```"
    ) in text
    assert "apt-get" not in text


def test_console_includes_suggested_commands_literally():
    report = _load("docker_no_runtime_no_hook_no_cdi.json")
    report.findings[0].fix_commands.append("printf '[bold]literal[/bold]'")
    output = StringIO()
    render_console(report, Console(file=output, width=120, color_system=None))
    text = output.getvalue()
    assert "Suggested commands (review before running):" in text
    assert "sudo nvidia-ctk runtime configure --runtime=docker" in text
    assert "sudo systemctl restart docker" in text
    assert "printf '[bold]literal[/bold]'" in text


def test_existing_findings_need_no_fix_commands():
    report = _load("power_limited_14w.json")
    assert all(f.fix_commands == [] for f in report.findings)
    assert "Suggested commands" not in render_markdown(report)


def test_fix_commands_round_trip_and_redaction():
    from spark_doctor.privacy import redact_report

    report = _load("docker_no_runtime_no_hook_no_cdi.json")
    command = "tool --token hf_abcdefghijklmnopqrstuvwxyz123456"
    report.findings[0].fix_commands = [command]
    loaded = ScanReport.model_validate_json(report.model_dump_json())
    assert loaded.findings[0].fix_commands == [command]
    redacted = redact_report(loaded)
    assert "hf_abcdefghijklmnopqrstuvwxyz123456" not in redacted.model_dump_json()
    assert "<redacted:" in redacted.findings[0].fix_commands[0]


def test_shell_runner_timeout_and_missing():
    from spark_doctor.shell import run

    # missing command
    r = run(["this_command_does_not_exist_12345"])
    assert not r.ok
    assert r.error == "command_not_found"

    # successful command
    r2 = run(["true"])
    assert r2.ok
    assert r2.returncode == 0

    # timeout
    r3 = run(["sleep", "5"], timeout=0.2)
    assert not r3.ok
    assert r3.error == "timeout"
