"""Filesystem scope restriction for JARVIS.

Provides optional least-privilege filesystem access control.
When configured, tools can only read/write within allowed paths.

Supports:
- Path normalization and resolution
- Symlink detection
- Path traversal prevention
- Configurable scope via security.filesystem_scope
"""
from __future__ import annotations

import os
import pathlib
from typing import Any, Dict, List, Optional

# Default configuration
_DEFAULT_CONFIG = {
    "enabled": False,
    "allowed_paths": [],
    "denied_paths": [],
    "allow_symlinks": True,
    "allow_absolute_paths": True,
}


def get_filesystem_scope_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get the filesystem scope configuration.
    
    Args:
        config: The configuration dict. If None, reads from config.
    
    Returns:
        Filesystem scope configuration dict.
    """
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        except Exception:
            return dict(_DEFAULT_CONFIG)
    
    scope = config.get("security", {}).get("filesystem_scope", {})
    return {
        "enabled": scope.get("enabled", False),
        "allowed_paths": scope.get("allowed_paths", []),
        "denied_paths": scope.get("denied_paths", []),
        "allow_symlinks": scope.get("allow_symlinks", True),
        "allow_absolute_paths": scope.get("allow_absolute_paths", True),
    }


def _resolve_path(path: str) -> str:
    """Resolve a path to its real path (following symlinks).
    
    Args:
        path: The path to resolve.
    
    Returns:
        The resolved absolute path.
    """
    try:
        return str(pathlib.Path(path).expanduser().resolve())
    except Exception:
        return os.path.normpath(os.path.abspath(os.path.expanduser(path)))


def _normalize_path(path: str) -> str:
    """Normalize a path for consistent comparison.
    
    Args:
        path: The path to normalize.
    
    Returns:
        The normalized absolute path.
    """
    try:
        return os.path.normpath(os.path.abspath(os.path.expanduser(path)))
    except Exception:
        return path


def is_symlink(path: str) -> bool:
    """Check if a path is a symlink.
    
    Args:
        path: The path to check.
    
    Returns:
        True if the path is a symlink, False otherwise.
    """
    try:
        return os.path.islink(path)
    except Exception:
        return False


def has_traversal(path: str) -> bool:
    """Check if a path contains traversal components.
    
    Args:
        path: The path to check.
    
    Returns:
        True if the path contains traversal, False otherwise.
    """
    normalized = _normalize_path(path)
    # Check for .. components in the normalized path
    if ".." in pathlib.Path(path).parts:
        return True
    # Check if normalized path differs significantly
    return False


def is_path_within_scope(path: str, allowed_paths: List[str]) -> bool:
    """Check if a path is within the allowed scope.
    
    Args:
        path: The path to check.
        allowed_paths: List of allowed path prefixes.
    
    Returns:
        True if the path is within scope, False otherwise.
    """
    if not allowed_paths:
        return True
    
    try:
        resolved = _resolve_path(path)
        for allowed in allowed_paths:
            allowed_resolved = _resolve_path(allowed)
            if resolved == allowed_resolved or resolved.startswith(allowed_resolved + os.sep):
                return True
    except Exception:
        return False
    
    return False


def is_path_denied(path: str, denied_paths: List[str]) -> bool:
    """Check if a path is in the denied list.
    
    Args:
        path: The path to check.
        denied_paths: List of denied path prefixes.
    
    Returns:
        True if the path is denied, False otherwise.
    """
    if not denied_paths:
        return False
    
    try:
        resolved = _resolve_path(path)
        for denied in denied_paths:
            denied_resolved = _resolve_path(denied)
            if resolved == denied_resolved or resolved.startswith(denied_resolved + os.sep):
                return True
    except Exception:
        return False
    
    return False


def is_path_allowed(target_path: str, config: Optional[Dict[str, Any]] = None) -> tuple[bool, str]:
    """Check if a path is allowed by filesystem scope policy.
    
    This is the main entry point for filesystem path validation.
    
    Args:
        target_path: The path to check.
        config: The configuration dict. If None, reads from config.
    
    Returns:
        Tuple of (is_allowed, reason).
    """
    scope = get_filesystem_scope_config(config)
    
    # If scope is not enabled, allow all
    if not scope.get("enabled", False):
        return True, ""
    
    # Normalize the path
    normalized = _normalize_path(target_path)
    
    # Check for absolute path restriction
    if not scope.get("allow_absolute_paths", True):
        if os.path.isabs(target_path):
            return False, "Absolute paths are not allowed"
    
    # Check for traversal
    if has_traversal(target_path):
        return False, "Path contains traversal components (..)"
    
    # Check symlink
    if not scope.get("allow_symlinks", True):
        if is_symlink(target_path):
            # Check if symlink target is within scope
            try:
                real_path = _resolve_path(target_path)
                if not is_path_within_scope(real_path, scope.get("allowed_paths", [])):
                    return False, "Symlink target is outside allowed scope"
            except Exception:
                return False, "Cannot resolve symlink target"
    
    # Check denied paths
    if is_path_denied(target_path, scope.get("denied_paths", [])):
        return False, "Path is in denied list"
    
    # Check allowed paths (if configured)
    allowed_paths = scope.get("allowed_paths", [])
    if allowed_paths:
        if not is_path_within_scope(target_path, allowed_paths):
            return False, "Path is outside allowed filesystem scope"
    
    return True, ""


def check_path_scope(tool_name: str, path: str, config: Optional[Dict[str, Any]] = None) -> None:
    """Check path scope and raise an error if not allowed.
    
    Args:
        tool_name: The name of the tool making the request.
        path: The path to check.
        config: The configuration dict.
    
    Raises:
        PathScopeError: If the path is not allowed.
    """
    allowed, reason = is_path_allowed(path, config)
    if not allowed:
        raise PathScopeError(f"[{tool_name}] {reason}: {path}")


class PathScopeError(Exception):
    """Raised when a path is outside the allowed filesystem scope."""
    pass


# Legacy functions for backward compatibility with Step 142

def get_allowed_paths(config: Optional[Dict[str, Any]] = None) -> List[str]:
    """Get the configured allowed paths.
    
    Legacy function for backward compatibility.
    """
    scope = get_filesystem_scope_config(config)
    return scope.get("allowed_paths", [])


def check_path_traversal(path: str, base: str = "/") -> bool:
    """Check if a path contains traversal attempts.
    
    Legacy function for backward compatibility with Step 142.
    """
    return has_traversal(path)


def validate_script_paths(code: str, allowed_paths: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validate that filesystem operations in a script stay within allowed paths.
    
    Legacy function for backward compatibility with Step 142.
    """
    if allowed_paths is None:
        allowed_paths = get_allowed_paths()
    
    if not allowed_paths:
        return {"allowed": True, "violations": []}
    
    import re
    
    violations: List[str] = []
    
    # Look for common file operation patterns with string literals
    patterns = [
        # open("path", ...)
        (r'''open\s*\(\s*["']([^"']+)["']''', "open()"),
        # pathlib.Path("path")
        (r'''Path\s*\(\s*["']([^"']+)["']''', "Path()"),
        # os.path operations
        (r'''os\.path\.[a-z]+\s*\(\s*["']([^"']+)["']''', "os.path"),
        # shutil operations
        (r'''shutil\.[a-z]+\s*\(\s*["']([^"']+)["']''', "shutil"),
    ]
    
    for pattern, desc in patterns:
        matches = re.findall(pattern, code)
        for match in matches:
            # Check absolute paths
            if match.startswith("/"):
                if not is_path_within_scope(match, allowed_paths):
                    violations.append(f"{desc}: {match}")
    
    return {
        "allowed": len(violations) == 0,
        "violations": violations,
    }
