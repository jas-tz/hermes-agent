"""Centralized secret redaction for JARVIS.

This module provides a single redaction entry point for all secret-bearing
content across the system. It handles API keys, tokens, private keys,
database URLs, passwords, and other credential formats.

All redaction uses the [REDACTED] marker for consistency.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Patterns for secret detection
# =============================================================================

# Private keys
_PRIVATE_KEY_PATTERNS = [
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
    r"-----BEGIN PRIVATE KEY-----",
    r"-----BEGIN ENCRYPTED PRIVATE KEY-----",
]

# Database URLs with credentials
_DATABASE_URL_PATTERNS = [
    r"postgres://[^:\s]+:[^@\s]+@[^\s]+",
    r"postgresql://[^:\s]+:[^@\s]+@[^\s]+",
    r"mysql://[^:\s]+:[^@\s]+@[^\s]+",
    r"mongodb://[^:\s]+:[^@\s]+@[^\s]+",
    r"mongodb\+srv://[^:\s]+:[^@\s]+@[^\s]+",
    r"redis://[^:\s]+:[^@\s]+@[^\s]+",
    r"amqp://[^:\s]+:[^@\s]+@[^\s]+",
    r"amqps://[^:\s]+:[^@\s]+@[^\s]+",
    r"Server=[^;]+;Password=[^;]+;[^;]*",
    r"Data Source=[^;]+;Password=[^;]+;[^;]*",
]

# JWT tokens (three base64url segments separated by dots)
_JWT_PATTERNS = [
    r"eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",
]

# Bearer tokens in headers
_BEARER_PATTERNS = [
    r"Bearer [A-Za-z0-9_\-\.]+",
    r"Basic [A-Za-z0-9+/=]+",
]

# Generic high-entropy strings that look like secrets
_GENERIC_SECRET_PATTERNS = [
    r"(?i)(?:api[_-]?key|token|secret|password)\s*[:=]\s*[\"']?([A-Za-z0-9_\-]{15,})[\"']?",
]

# Full API key prefixes
_FULL_KEY_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{10,}",           # OpenAI / OpenRouter / Anthropic
    r"ghp_[A-Za-z0-9]{10,}",            # GitHub PAT
    r"github_pat_[A-Za-z0-9_]{10,}",    # GitHub PAT (fine-grained)
    r"xox[baprs]-[A-Za-z0-9-]{10,}",    # Slack tokens
    r"AIza[A-Za-z0-9_-]{30,}",          # Google API keys
    r"AKIA[A-Z0-9]{16}",                # AWS Access Key ID
]

# Compile all patterns
_ALL_PATTERNS = (
    _PRIVATE_KEY_PATTERNS
    + _DATABASE_URL_PATTERNS
    + _JWT_PATTERNS
    + _BEARER_PATTERNS
    + _GENERIC_SECRET_PATTERNS
    + _FULL_KEY_PATTERNS
)

_COMPILED_PATTERNS: List[re.Pattern] = [
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in _ALL_PATTERNS
]


# =============================================================================
# Core redaction functions
# =============================================================================

def redact_secrets(text: str, force: bool = False) -> str:
    """Redact secrets from text.

    This is the centralized entry point for secret redaction. It detects
    and redacts API keys, tokens, private keys, database URLs, passwords,
    and other credential formats.

    Args:
        text: The text to redact.
        force: If True, always redact regardless of global setting.

    Returns:
        Redacted text with secrets replaced by [REDACTED].
    """
    if not text:
        return text

    # Apply all patterns
    for pattern in _COMPILED_PATTERNS:
        text = pattern.sub("[REDACTED]", text)

    return text


def contains_secrets(text: str) -> Tuple[bool, List[str]]:
    """Check if text contains potential secrets.

    Args:
        text: The text to check.

    Returns:
        Tuple of (has_secrets, list_of_descriptions).
    """
    if not text:
        return False, []

    findings: List[str] = []
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            findings.append("secret_value")

    return bool(findings), findings


def sanitize_for_memory(content: str) -> Tuple[str, bool]:
    """Sanitize content before storing in memory/Qdrant.

    Args:
        content: The content to sanitize.

    Returns:
        Tuple of (sanitized_content, was_redacted).
    """
    if not content:
        return content, False

    has_secrets, _findings = contains_secrets(content)
    if has_secrets:
        return redact_secrets(content), True

    return content, False


def reject_if_secrets(content: str) -> Optional[str]:
    """Reject content if it contains secrets.

    Args:
        content: The content to check.

    Returns:
        Error message if secrets found, None if safe.
    """
    if not content:
        return None

    has_secrets, findings = contains_secrets(content)
    if has_secrets:
        return (
            "Content appears to contain secrets. "
            "Secrets must not be stored in memory. "
            "Please remove sensitive information before storing."
        )

    return None


# =============================================================================
# Snapshot protection
# =============================================================================

def sanitize_for_snapshot(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize a dictionary for snapshot storage.

    Recursively redacts string values that may contain secrets.
    Environment variables are NOT included in snapshots by default.

    Args:
        data: The dictionary to sanitize.

    Returns:
        Sanitized dictionary safe for snapshot storage.
    """
    if not data:
        return {}

    sanitized: Dict[str, Any] = {}
    for key, value in data.items():
        # Skip sensitive keys entirely
        if _is_sensitive_key(key):
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, str):
            # Only redact if the string actually contains secrets
            has_secrets, _ = contains_secrets(value)
            sanitized[key] = redact_secrets(value) if has_secrets else value
        elif isinstance(value, dict):
            sanitized[key] = sanitize_for_snapshot(value)
        elif isinstance(value, list):
            sanitized[key] = [
                redact_secrets(item) if isinstance(item, str) and contains_secrets(item)[0]
                else sanitize_for_snapshot(item) if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            sanitized[key] = value

    return sanitized


def _is_sensitive_key(key: str) -> bool:
    """Check if a key name suggests it holds a secret value.

    Args:
        key: The key name to check.

    Returns:
        True if the key likely holds a secret.
    """
    if not key:
        return False

    sensitive_patterns = [
        r"(?i)\bpasswords?\b",
        r"(?i)\bsecrets?\b",
        r"(?i)\btokens?\b",
        r"(?i)\bapi[_-]?keys?\b",
        r"(?i)\bapikeys?\b",
        r"(?i)\bprivate[_-]?keys?\b",
        r"(?i)\baccess[_-]?tokens?\b",
        r"(?i)\brefresh[_-]?tokens?\b",
        r"(?i)\bauth[_-]?tokens?\b",
        r"(?i)\bbearer[_-]?tokens?\b",
        r"(?i)\bclient[_-]?secrets?\b",
        r"(?i)\bcredentials?\b",
        r"(?i)\bconnection[_-]?strings?\b",
        r"(?i)\bdatabase[_-]?urls?\b",
        r"(?i)\bdb[_-]?urls?\b",
        r"(?i)\bdsns?\b",
        r"(?i)\baws[_-]?access[_-]?keys?\b",
        r"(?i)\baws[_-]?secrets?\b",
        r"(?i)\bakia\b",
    ]

    for pattern in sensitive_patterns:
        if re.search(pattern, key):
            return True

    return False


# Environment variable names that are always considered sensitive
_SENSITIVE_ENV_KEYS = frozenset({
    "API_KEY", "APIKEY", "API_SECRET", "APP_SECRET", "APP_KEY",
    "ACCESS_TOKEN", "AUTH_TOKEN", "AUTHORIZATION", "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "BASIC_AUTH",
    "BEARER_TOKEN", "CLIENT_SECRET", "CLIENT_TOKEN", "CREDENTIAL",
    "CREDENTIALS", "CSRF_TOKEN", "DATABASE_URL", "DB_PASSWORD",
    "DB_SECRET", "DB_URL", "DB_USER", "EMAIL_PASSWORD", "ENCRYPTION_KEY",
    "GITHUB_TOKEN", "GH_TOKEN", "GIT_TOKEN", "GOOGLE_API_KEY",
    "GOOGLE_SECRET", "JWT_SECRET", "JWT_TOKEN", "MYSQL_PASSWORD",
    "OAUTH_SECRET", "OAUTH_TOKEN", "PASS", "PASSPHRASE", "PASSWORD",
    "PG_PASSWORD", "PG_SECRET", "POSTGRES_PASSWORD", "PRIVATE_KEY",
    "REDIS_PASSWORD", "REDIS_SECRET", "REFRESH_TOKEN", "S3_SECRET",
    "SECRET", "SECRET_KEY", "SECRET_TOKEN", "SESSION_SECRET",
    "SESSION_TOKEN", "SIGNING_KEY", "SIGNING_SECRET", "SLACK_TOKEN",
    "SLACK_SECRET", "SPREADSHEET_TOKEN", "SSH_KEY", "SSL_CERT",
    "SSL_KEY", "TOKEN", "TOKEN_SECRET", "USER_SECRET", "X_API_KEY",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
    "NOUS_API_KEY", "TOGETHER_API_KEY", "FIREWORKS_API_KEY",
    "DEEPINFRA_API_KEY", "GEMINI_API_KEY", "COHERE_API_KEY",
})


def is_sensitive_env_key(key: str) -> bool:
    """Check if an environment variable name suggests it holds a secret.

    Args:
        key: The environment variable name.

    Returns:
        True if the key likely holds a secret.
    """
    if not key:
        return False

    # Direct match
    if key.upper() in _SENSITIVE_ENV_KEYS:
        return True

    # Pattern match
    return _is_sensitive_key(key)
