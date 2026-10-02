from __future__ import annotations

import getpass
import os
import re
import socket
from typing import Any

from ..models import ScanReport


REDACTION_RE = re.compile(r"(<redacted:[a-z_]+>)")
SECRET_NAMES = (
    r"(?:[A-Za-z0-9]+[_-])*(?:api[_-]?key|password|secret|token|(?:access|auth|hf)[_-]token|"
    r"(?:openai|ngc)[_-]api[_-]?key|client[_-]secret|aws[_-]secret[_-]access[_-]key)"
)
SHELL_WORD_PART = r'''(?:"(?:\\.|[^"\\])*"|'[^']*'|\\.|[^\s'"\\])'''
SECRET_RE = re.compile(
    rf"'(?P<single_prefix>(?:--)?{SECRET_NAMES}=)[^']*'{SHELL_WORD_PART}*"
    rf'|"(?P<double_prefix>(?:--)?{SECRET_NAMES}=)(?:\\.|[^"\\])*"{SHELL_WORD_PART}*'
    rf"|(?P<prefix>(?<![\w-])(?:(?:--)?{SECRET_NAMES}['\"]?[ \t]*[:=][ \t]*"
    rf"|--{SECRET_NAMES}[ \t]+(?!--))){SHELL_WORD_PART}+",
    re.IGNORECASE,
)
SECRET_KEY_RE = re.compile(rf"(?:{SECRET_NAMES}|authorization)", re.IGNORECASE)
HOME_PATH_RE = re.compile(r'''(?<![\w])/(?:home|Users)/[^/\s"'<>:;,()]+''')


TOKEN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Hugging Face tokens
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "<redacted:hf_token>"),
    # NGC API keys (base64-ish, long)
    (re.compile(r"\bnvapi-[A-Za-z0-9_\-]{20,}\b"), "<redacted:ngc_token>"),
    # Generic bearer / authorization
    (re.compile(r"(?i)(?<![\w-])(Bearer|Token)\s+(?!--)[A-Za-z0-9_\-\.=]{12,}"), r"\1 <redacted:token>"),
    # AWS-style
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<redacted:aws_key>"),
    # JWT
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), "<redacted:jwt>"),
    # OpenAI sk- keys
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "<redacted:sk_token>"),
    # SSH private key blocks
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
     "<redacted:private_key>"),
]

# Private IPv4 ranges (used when network identifiers are not included)
PRIVATE_IP_RE = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r"|127\.\d{1,3}\.\d{1,3}\.\d{1,3})\b"
)

MAC_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")


def _replace_unredacted(text: str, pattern: re.Pattern[str], replacement: str) -> str:
    # Redaction runs again at export; existing placeholders must survive unchanged.
    return "".join(
        part if index % 2 else pattern.sub(replacement, part)
        for index, part in enumerate(REDACTION_RE.split(text))
    )


def _redact_secret(match: re.Match[str]) -> str:
    for group, quote in (("single_prefix", "'"), ("double_prefix", '"'), ("prefix", "")):
        if (prefix := match.group(group)) is not None:
            return f"{quote}{prefix}<redacted:secret>{quote}"
    return match.group()


def _current_identifiers() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        out["user"] = getpass.getuser()
    except Exception:  # noqa: BLE001
        pass
    try:
        out["host"] = socket.gethostname()
    except Exception:  # noqa: BLE001
        pass
    out["home"] = os.path.expanduser("~")
    return out


def redact_text(
    text: str,
    *,
    include_network_identifiers: bool = False,
    identifiers: dict[str, str] | None = None,
) -> str:
    if not isinstance(text, str) or not text:
        return text
    ids = identifiers if identifiers is not None else _current_identifiers()

    # One pass keeps closing shell quotes and embedded placeholders inside their value.
    text = SECRET_RE.sub(_redact_secret, text)
    for pattern, replacement in TOKEN_PATTERNS:
        text = _replace_unredacted(text, pattern, replacement)

    home = ids.get("home")
    if home and home not in ("/", ""):
        text = _replace_unredacted(text, re.compile(re.escape(home)), "<redacted:home>")
    text = _replace_unredacted(text, HOME_PATH_RE, "<redacted:home>")
    user = ids.get("user")
    if user and len(user) >= 2:
        text = _replace_unredacted(text, re.compile(rf"\b{re.escape(user)}\b"), "<redacted:user>")
    for host in (ids.get("host"), ids.get("source_host")):
        if host and len(host) >= 2:
            text = _replace_unredacted(text, re.compile(rf"\b{re.escape(host)}\b"), "<redacted:host>")

    if not include_network_identifiers:
        text = _replace_unredacted(text, PRIVATE_IP_RE, "<redacted:private_ip>")
        text = _replace_unredacted(text, MAC_RE, "<redacted:mac>")

    return text


def redact_obj(
    obj: Any,
    *,
    include_network_identifiers: bool = False,
    identifiers: dict[str, str] | None = None,
) -> Any:
    if identifiers is None:
        identifiers = _current_identifiers()
    if isinstance(obj, str):
        return redact_text(
            obj,
            include_network_identifiers=include_network_identifiers,
            identifiers=identifiers,
        )
    if isinstance(obj, list):
        return [
            redact_obj(x, include_network_identifiers=include_network_identifiers, identifiers=identifiers)
            for x in obj
        ]
    if isinstance(obj, tuple):
        return tuple(
            redact_obj(x, include_network_identifiers=include_network_identifiers, identifiers=identifiers)
            for x in obj
        )
    if isinstance(obj, dict):
        return {
            k: (
                "<redacted:secret>"
                if isinstance(k, str) and SECRET_KEY_RE.fullmatch(k) and v is not None
                else redact_obj(v, include_network_identifiers=include_network_identifiers, identifiers=identifiers)
            )
            for k, v in obj.items()
        }
    return obj


def redact_report(report: ScanReport, *, include_network_identifiers: bool = False) -> ScanReport:
    """Return a new ScanReport with redacted content."""
    data = report.model_dump(mode="json")
    identifiers = _current_identifiers()
    uname = report.os.get("uname")
    if isinstance(uname, str):
        parts = uname.split()
        if len(parts) >= 3 and parts[0] in {"Linux", "Darwin"} and not REDACTION_RE.fullmatch(parts[1]):
            identifiers["source_host"] = parts[1]
    redacted = redact_obj(
        data, include_network_identifiers=include_network_identifiers, identifiers=identifiers
    )
    redacted["anonymized"] = True
    return ScanReport.model_validate(redacted)
