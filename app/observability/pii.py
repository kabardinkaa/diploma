import hashlib
import re


PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "PHONE_RU": re.compile(
        r"(?<!\d)(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?"
        r"\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)"
    ),
    "CARD": re.compile(r"(?<!\d)(?:\d{4}[\s\-]?){3}\d{4}(?!\d)"),
    "INN": re.compile(r"(?<!\d)(?:\d{10}|\d{12})(?!\d)"),
    "PASSPORT": re.compile(r"(?<!\d)\d{2}\s?\d{2}\s?\d{6}(?!\d)"),
}


def redact_pii(text: str) -> str:
    """Replace common personal data patterns with safe placeholders."""
    if not text:
        return text

    for name, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[{name}]", text)

    return text


def prompt_hash(text: str) -> str:
    """Return short stable hash for prompt logging without raw prompt exposure."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]