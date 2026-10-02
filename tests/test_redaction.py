from __future__ import annotations

import shlex
import subprocess
import sys

import pytest

from spark_doctor.models import ProcessInfo, ScanReport
from spark_doctor.privacy import redact_text, redact_obj
from spark_doctor.privacy import redact_report


IDS = {"user": "zerocool", "host": "sparkbox", "home": "/home/zerocool"}


def test_unterminated_quoted_secret_does_not_stall_redaction():
    code = "from spark_doctor.privacy import redact_text; import sys; print(redact_text(sys.stdin.read()))"
    result = subprocess.run(
        [sys.executable, "-c", code], input="'token=" + "\\a" * 200,
        text=True, capture_output=True, timeout=3, check=True,
    )
    assert "<redacted:secret>" in result.stdout


def test_redacts_home_path():
    text = "loaded /home/zerocool/models/qwen.gguf"
    out = redact_text(text, identifiers=IDS)
    assert "/home/zerocool" not in out
    assert "<redacted:home>" in out


def test_redacts_hf_token():
    text = "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz123456"
    out = redact_text(text, identifiers=IDS)
    assert "hf_abcdefghijklmnopqrstuvwxyz123456" not in out


def test_redacts_bearer_token():
    text = "Authorization: Bearer abcdef1234567890xyz"
    out = redact_text(text, identifiers=IDS)
    assert "abcdef1234567890xyz" not in out
    assert "<redacted:token>" in out


def test_redacts_private_ip_by_default():
    text = "listening on 192.168.1.42"
    out = redact_text(text, identifiers=IDS)
    assert "192.168.1.42" not in out


def test_keeps_private_ip_when_allowed():
    text = "listening on 192.168.1.42"
    out = redact_text(text, include_network_identifiers=True, identifiers=IDS)
    assert "192.168.1.42" in out


def test_redacts_username_and_host():
    text = "user=zerocool host=sparkbox"
    out = redact_text(text, identifiers=IDS)
    assert "zerocool" not in out
    assert "sparkbox" not in out


def test_redact_obj_recursive():
    obj = {"cmd": "curl -H 'Authorization: Bearer secret_xyz_token_value'", "path": "/home/zerocool/x"}
    out = redact_obj(obj, identifiers=IDS)
    assert "secret_xyz_token_value" not in out["cmd"]
    assert "<redacted:home>" in out["path"]


def test_redacts_openai_key():
    text = "key=sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    out = redact_text(text, identifiers=IDS)
    assert "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" not in out


@pytest.mark.parametrize(
    "argument, secret",
    [
        ("--api-key private-value", "private-value"),
        ("--api_key private-value", "private-value"),
        ("--password=x!@:/a,b;c", "x!@:/a,b;c"),
        ("--password 'short secret'", "short secret"),
        ('--api-key="secret with spaces"', "secret with spaces"),
        ("--token abc", "abc"),
        ("--access-token private-value", "private-value"),
        ("PASSWORD=abc", "abc"),
        ('api_key: "a secret"', "a secret"),
        ("'--api-key=whole quoted credential'", "whole quoted credential"),
        ('"PASSWORD=whole quoted credential"', "whole quoted credential"),
    ],
)
def test_redacts_credential_arguments_and_assignments(argument: str, secret: str):
    text = f"serve {argument} --model qwen --max-tokens 4096"
    out = redact_text(text, identifiers=IDS)
    assert secret not in out
    assert "--model qwen --max-tokens 4096" in out
    assert "<redacted:" in out


def test_keeps_options_after_missing_credential_value():
    text = "serve --api-key --model qwen --token --context-length 4096"
    assert redact_text(text, identifiers=IDS) == text


def test_redacts_shell_quoted_password_containing_an_apostrophe():
    text = "serve --password 'first-part'\"'\"'last-part' --model qwen"
    out = redact_text(text, identifiers=IDS)
    assert out == "serve --password <redacted:secret> --model qwen"


@pytest.mark.parametrize("argument", ["'--api-key=whole quoted credential'", '"PASSWORD=whole quoted credential"'])
def test_redacts_entire_quoted_assignment(argument: str):
    out = redact_text(f"serve {argument} --model qwen", identifiers=IDS)
    assert "whole" not in out
    assert "quoted" not in out
    assert "credential" not in out
    assert "--model qwen" in out


@pytest.mark.parametrize("secret", [
    "two words", "first'last", 'a "quoted" value', "x!@:/a,b;c",
    "first last\\", "first\\'last", "<redacted:home>canary-secret",
])
@pytest.mark.parametrize("equals", [False, True])
def test_redacts_credentials_from_preserved_process_arguments(secret: str, equals: bool):
    credential = [f"--api-key={secret}"] if equals else ["--api-key", secret]
    arguments = shlex.join(["serve", *credential, "--model", "a model"])
    out = redact_text(arguments, identifiers=IDS)
    expected = ["--api-key=<redacted:secret>"] if equals else ["--api-key", "<redacted:secret>"]
    assert shlex.split(out) == ["serve", *expected, "--model", "a model"]
    assert redact_text(out, identifiers=IDS) == out


@pytest.mark.parametrize("prefix", ["--api-key=", "PASSWORD=", "token: "])
def test_explicit_assignment_redacts_values_starting_with_dashes(prefix: str):
    text = f"{prefix}--canary-secret --model qwen"
    out = redact_text(text, identifiers=IDS)
    assert "canary-secret" not in out
    assert "--model qwen" in out


@pytest.mark.parametrize("name", ["DB_PASSWORD", "WANDB_API_KEY", "ANTHROPIC_API_KEY", "custom-service-token"])
@pytest.mark.parametrize("form", ["env", "equals", "separate", "structured"])
def test_redacts_prefixed_credential_names(name: str, form: str):
    secret = "canary secret\\'tail"
    if form == "structured":
        out = redact_obj({name: secret, "tokenizer": "qwen", "max_tokens": 4096}, identifiers=IDS)
        assert out == {name: "<redacted:secret>", "tokenizer": "qwen", "max_tokens": 4096}
        return
    credentials = {
        "env": [f"{name}={secret}"],
        "equals": [f"--{name}={secret}"],
        "separate": [f"--{name}", secret],
    }[form]
    args = shlex.join(["serve", *credentials, "--model", "a model", "--max-tokens", "4096"])
    report = redact_report(ScanReport(processes=[ProcessInfo(pid=1, command="serve", args=args)]))
    assert "canary" not in report.model_dump_json()
    assert "tail" not in report.model_dump_json()
    expected = [part.replace(secret, "<redacted:secret>") for part in credentials]
    assert shlex.split(report.processes[0].args) == ["serve", *expected, "--model", "a model", "--max-tokens", "4096"]
    assert redact_report(report).model_dump() == report.model_dump()


def test_redacts_structured_credential_values_without_matching_ordinary_fields():
    obj = {
        "api_key": "short",
        "nested": {"Authorization": "Basic abc", "HF_TOKEN": "arbitrary"},
        "access-token": ["one", "two"],
        "tokenizer": "qwen",
        "max_tokens": 4096,
    }
    out = redact_obj(obj, identifiers=IDS)
    assert out["api_key"] == "<redacted:secret>"
    assert out["nested"] == {
        "Authorization": "<redacted:secret>", "HF_TOKEN": "<redacted:secret>"
    }
    assert out["access-token"] == "<redacted:secret>"
    assert out["tokenizer"] == "qwen"
    assert out["max_tokens"] == 4096
    assert obj["api_key"] == "short"


def test_redaction_is_idempotent_and_preserves_placeholders():
    text = "serve --api-key private-value --token hf_abcdefghijklmnopqrstuvwxyz123456"
    once = redact_text(text, identifiers=IDS)
    assert redact_text(once, identifiers=IDS) == once
    placeholders = "<redacted:home> <redacted:user> <redacted:host> <redacted:token>"
    assert redact_text(
        placeholders, identifiers={"user": "home", "host": "host", "home": "/home/home"}
    ) == placeholders


def test_imported_report_redacts_source_home_and_uname_hostname(monkeypatch):
    monkeypatch.setattr("spark_doctor.privacy.redact._current_identifiers", lambda: IDS)
    report = ScanReport(
        anonymized=False,
        os={"uname": "Linux source-spark 6.11.0 #1 SMP aarch64 GNU/Linux"},
        processes=[ProcessInfo(
            pid=1, command="serve", args="--api-key private-value --model /home/source-owner/qwen"
        )],
        reproduction_notes="source-spark loaded /Users/source-owner/work at 192.168.1.42",
    )
    result = redact_report(report)
    serialized = result.model_dump_json()
    assert "source-spark" not in serialized
    assert "source-owner" not in serialized
    assert "private-value" not in serialized
    assert "192.168.1.42" not in serialized
    assert result.anonymized
    assert result.os["uname"] == "Linux <redacted:host> 6.11.0 #1 SMP aarch64 GNU/Linux"
    assert "private-value" in report.processes[0].args
    assert redact_report(result).model_dump() == result.model_dump()
    with_network = redact_report(report, include_network_identifiers=True)
    assert "192.168.1.42" in with_network.reproduction_notes
    assert "private-value" not in with_network.model_dump_json()
