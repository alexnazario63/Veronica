"""Deterministic boundary sanitizer for documents and untrusted device output."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from knowledge.ingestion.ingestor import detect_prompt_injection, scrub_secrets

_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INJECTION = re.compile(
    r"(?:ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions|"
    r"reveal\s+(?:the\s+)?(?:system\s+)?prompt|"
    r"(?:you\s+are\s+now|act\s+as)\s+(?:an?\s+)?(?:administrator|system|assistant)|"
    r"bypass\s+(?:approval|safety|security)|disable\s+(?:all\s+)?safety|"
    r"execute\s+without\s+authorization|override\s+(?:system|safety)\s+(?:rules|instructions))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SanitizedUntrustedText:
    text: str
    injection_detected: bool
    reasons: tuple[str, ...]
    secrets_redacted: int


def sanitize_untrusted_text(value: str, *, source: str = "document", max_chars: int = 200_000) -> SanitizedUntrustedText:
    """Strip terminal/control payloads, redact credentials and neutralize injection lines.

    Suspicious lines are replaced in place so evidence position remains useful;
    output is still data and must never be promoted to system/developer instructions.
    """
    cleaned = _CONTROL.sub("", _ANSI.sub("", str(value)))[:max_chars]
    cleaned, redacted = scrub_secrets(cleaned)
    legacy_detected, legacy_reasons = detect_prompt_injection(cleaned)
    reasons: list[str] = list(legacy_reasons)
    output: list[str] = []
    for line in cleaned.splitlines():
        if _INJECTION.search(line):
            reasons.append(f"INDIRECT_INSTRUCTION:{source}")
            output.append("[UNTRUSTED_INSTRUCTION_REMOVED]")
        else:
            output.append(line)
    detected = legacy_detected or bool(reasons)
    return SanitizedUntrustedText("\n".join(output), detected, tuple(dict.fromkeys(reasons)), redacted)


def sanitize_device_output(lines: Iterable[str] | str, *, source: str = "device_output") -> SanitizedUntrustedText:
    """Apply the same boundary rules to live banners, descriptions, and protocol data."""
    return sanitize_untrusted_text(lines if isinstance(lines, str) else "\n".join(map(str, lines)), source=source)

