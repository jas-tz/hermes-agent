"""Document sanitization for untrusted content.

This module provides conservative sanitization of untrusted document content
before it can influence agent instructions or tool decisions.

The sanitizer is intentionally conservative: when in doubt, it strips content
rather than risk allowing injection vectors through.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Patterns that indicate potential injection vectors in untrusted content.
# Each entry: (regex, description, replacement)
_INJECTION_PATTERNS: List[Tuple[str, str, str]] = [
    # Script tags and their contents
    (r"<script[^>]*>.*?</script>", "script tag", "[REMOVED_SCRIPT]"),
    # Event handlers (onload, onclick, etc.)
    (r"\son\w+\s*=\s*[\"'][^\"']*[\"']", "event handler", ""),
    # Data URIs (can contain executable content)
    (r"data:text/html[^,]*,\s*[^\"']*", "data URI", "[REMOVED_DATA_URI]"),
    # JavaScript protocol in URLs
    (r"javascript\s*:\s*[^\"']*", "javascript protocol", "[REMOVED_JS]"),
    # VBScript protocol
    (r"vbscript\s*:\s*[^\"']*", "vbscript protocol", "[REMOVED_VBS]"),
    # iframe tags
    (r"<iframe[^>]*>.*?</iframe>", "iframe", "[REMOVED_IFRAME]"),
    # object/embed tags
    (r"<object[^>]*>.*?</object>", "object tag", "[REMOVED_OBJECT]"),
    (r"<embed[^>]*>", "embed tag", "[REMOVED_EMBED]"),
    # meta refresh
    (r"<meta[^>]*http-equiv\s*=\s*[\"']refresh[\"'][^>]*>", "meta refresh", "[REMOVED_META]"),
    # base tag (can hijack relative URLs)
    (r"<base[^>]*>", "base tag", "[REMOVED_BASE]"),
    # form tags with action
    (r"<form[^>]*action\s*=\s*[\"'][^\"']*[\"'][^>]*>", "form action", "[REMOVED_FORM]"),
    # SVG with script content
    (r"<svg[^>]*>.*?<script.*?</svg>", "svg with script", "[REMOVED_SVG]"),
    # Expression/behavior (IE-specific CSS)
    (r"expression\s*\(", "CSS expression", "[REMOVED_EXPRESSION]"),
    (r"behavior\s*:", "CSS behavior", "[REMOVED_BEHAVIOR]"),
    # @import in CSS (can load external resources)
    (r"@import\s+[\"'][^\"']*[\"']", "CSS import", "[REMOVED_IMPORT]"),
    # binding in CSS (Mozilla-specific)
    (r"binding\s*:", "CSS binding", "[REMOVED_BINDING]"),
]

# Compile patterns for performance
_INJECTION_PATTERNS_COMPILED: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(pattern, re.IGNORECASE | re.DOTALL), desc, replacement)
    for pattern, desc, replacement in _INJECTION_PATTERNS
]

# Maximum content size to sanitize (prevent DoO via large inputs)
MAX_SANITIZE_SIZE = 1_000_000  # 1 MB


def sanitize_untrusted_content(content: str) -> str:
    """Sanitize untrusted document content.
    
    Strips common injection vectors (scripts, event handlers, data URIs, etc.)
    from content that will be presented to the agent.
    
    This is a conservative sanitizer: when in doubt, it strips content.
    It is NOT a complete XSS/injection prevention solution — defense in depth
    should still treat untrusted content as potentially malicious.
    
    Args:
        content: The untrusted content to sanitize.
    
    Returns:
        Sanitized content with injection vectors replaced by markers.
        If sanitization fails, returns the content with a warning marker
        prepended.
    """
    if not content:
        return content
    
    # Size limit check
    if len(content) > MAX_SANITIZE_SIZE:
        content = content[:MAX_SANITIZE_SIZE] + "\n[CONTENT_TRUNCATED]"
    
    try:
        sanitized = content
        for pattern, desc, replacement in _INJECTION_PATTERNS_COMPILED:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized
    except Exception:
        # Fail safe: return content with warning marker
        return "[SANITIZATION_FAILED — TREAT AS UNTRUSTED]\n" + content


def contains_injection_vectors(content: str) -> Tuple[bool, List[str]]:
    """Check if content contains potential injection vectors.
    
    Args:
        content: The content to check.
    
    Returns:
        Tuple of (has_vectors, list_of_descriptions).
    """
    if not content:
        return False, []
    
    findings: List[str] = []
    for pattern, desc, _ in _INJECTION_PATTERNS_COMPILED:
        if pattern.search(content):
            findings.append(desc)
    
    return bool(findings), findings


def sanitize_for_display(content: str) -> str:
    """Sanitize content for display in chat/UI.
    
    More aggressive than sanitize_untrusted_content — also escapes HTML
    entities to prevent rendering attacks.
    
    Args:
        content: The content to sanitize for display.
    
    Returns:
        Sanitized content safe for display.
    """
    if not content:
        return content
    
    # First pass: strip injection vectors
    sanitized = sanitize_untrusted_content(content)
    
    # Second pass: escape HTML entities
    sanitized = sanitized.replace("&", "&amp;")
    sanitized = sanitized.replace("<", "&lt;")
    sanitized = sanitized.replace(">", "&gt;")
    sanitized = sanitized.replace('"', "&quot;")
    sanitized = sanitized.replace("'", "&#x27;")
    
    return sanitized
