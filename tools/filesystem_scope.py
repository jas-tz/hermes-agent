"""Filesystem scope restriction for JARVIS.

Provides optional least-privilege filesystem access control.
When configured, tools can only read/write within allowed paths.

Supports:
- Path normalization and resolution
- Symlink detection
- Path traversal prevention
- Configurable scope via security.filesystem_scope
- Write-only denied paths via security.filesystem_scope.write_denied_paths
  (independent of the read/write scope; see is_write_allowed)
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


# ── Write-only denied paths ─────────────────────────────────────────────────
#
# ``security.filesystem_scope.write_denied_paths`` lists directories (or
# files) that agent file tools must never write to — the path itself and
# every descendant. It is independent of the scope above: it is consulted
# whenever the list is non-empty, whether or not ``enabled`` is set, and it
# never affects reads.
#
# A path is denied when any of the following matches a denied entry:
#   * its lexically normalised absolute form (``..`` collapsed),
#   * its symlink-resolved real path,
# compared component by component and case-insensitively, so a sibling that
# merely shares a string prefix (``vault2`` vs ``vault``) is NOT denied; or
#   * the identity (st_dev, st_ino) of any existing ancestor, which catches
#     aliases string comparison cannot see — e.g. DrvFs case variants and
#     Windows 8.3 short names (``JARVIS~1``) under WSL.
#
# Fail closed: if the list cannot be loaded or a path cannot be evaluated,
# the write is refused.


class WriteBoundaryError(Exception):
    """The write boundary could not be evaluated; the write must be refused."""


def _load_write_boundary_config() -> Dict[str, Any]:
    """Load the config for the write boundary. Unlike
    get_filesystem_scope_config, a failure here propagates (fail closed)."""
    from hermes_cli.config import load_config_readonly
    return load_config_readonly()


def get_write_denied_paths(config: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return the configured write-denied paths as absolute, expanded paths.

    Raises WriteBoundaryError when the configuration is malformed, so callers
    refuse the write rather than silently ignoring a bad entry.
    """
    if config is None:
        config = _load_write_boundary_config()
    security = config.get("security") or {}
    if not isinstance(security, dict):
        raise WriteBoundaryError("security config is not a mapping")
    scope = security.get("filesystem_scope") or {}
    if not isinstance(scope, dict):
        raise WriteBoundaryError("security.filesystem_scope is not a mapping")
    raw = scope.get("write_denied_paths") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise WriteBoundaryError("write_denied_paths must be a list of paths")
    denied: List[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            raise WriteBoundaryError(f"write_denied_paths entry is not a string: {entry!r}")
        entry = entry.strip()
        if not entry:
            continue
        expanded = os.path.expanduser(entry)
        if not os.path.isabs(expanded):
            raise WriteBoundaryError(f"write_denied_paths entry is not absolute: {entry!r}")
        denied.append(expanded)
    return denied


def _path_key(path: str) -> tuple:
    return tuple(part.casefold() for part in pathlib.PurePath(path).parts)


def _path_forms(path: str) -> set:
    """Comparison keys for the lexical and symlink-resolved forms of *path*."""
    return {_path_key(os.path.normpath(path)), _path_key(os.path.realpath(path))}


def _existing_identities(path: str) -> set:
    """(st_dev, st_ino) of *path* and of each of its existing ancestors."""
    identities = set()
    current = os.path.realpath(path)
    while True:
        try:
            st = os.stat(current)
            identities.add((st.st_dev, st.st_ino))
        except (FileNotFoundError, NotADirectoryError):
            pass
        parent = os.path.dirname(current)
        if parent == current:
            return identities
        current = parent


def is_write_allowed(target_path: str, base_dir: Optional[str] = None,
                     config: Optional[Dict[str, Any]] = None) -> tuple[bool, str]:
    """Check *target_path* against ``write_denied_paths``.

    A relative *target_path* is resolved against *base_dir* (the task's
    working directory); *base_dir* must then be absolute. Returns
    ``(is_allowed, reason)``. Never raises: any error denies the write.
    """
    try:
        denied = get_write_denied_paths(config)
        if not denied:
            return True, ""
        if not isinstance(target_path, str) or not target_path.strip():
            raise WriteBoundaryError("empty path")
        expanded = os.path.expanduser(target_path)
        if not os.path.isabs(expanded):
            base = os.path.expanduser(str(base_dir if base_dir is not None else os.getcwd()))
            if not os.path.isabs(base):
                raise WriteBoundaryError(f"base directory is not absolute: {base!r}")
            expanded = os.path.join(base, expanded)
        candidate_keys = _path_forms(expanded)
        candidate_ids = None
        for entry in denied:
            for denied_key in _path_forms(entry):
                if any(key[:len(denied_key)] == denied_key for key in candidate_keys):
                    return False, f"Path is inside a write-denied location ({entry})"
            try:
                st = os.stat(entry)
            except FileNotFoundError:
                continue
            if candidate_ids is None:
                candidate_ids = _existing_identities(expanded)
            if (st.st_dev, st.st_ino) in candidate_ids:
                return False, f"Path is inside a write-denied location ({entry})"
        return True, ""
    except Exception as exc:
        return False, f"Write boundary check failed ({type(exc).__name__}: {exc}); refusing write"
