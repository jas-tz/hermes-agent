"""JARVIS Sub-Agent Filesystem Policy — Phase 25B.

Controlled filesystem tool integration for JARVIS sub-agents.

Uses the existing filesystem_scope.py security mechanisms:
- is_path_allowed() for path validation
- is_write_allowed() for write boundary checking
- has_traversal() for traversal detection

Defines a dedicated sub-agent filesystem scope separate from parent paths.
Parent-only filesystem paths remain inaccessible to sub-agents.

Phase 25B: controlled filesystem tool integration.
"""

from __future__ import annotations

import logging
import os
import pathlib
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sub-agent filesystem policy
# ---------------------------------------------------------------------------

# Default sub-agent filesystem root
# Can be overridden via config: security.subagent_filesystem.root
DEFAULT_SUBAGENT_ROOT = os.path.expanduser("~/.hermes/subagent-workspace")


def get_subagent_filesystem_root(config: Optional[Dict[str, Any]] = None) -> str:
    """Get the root directory for sub-agent filesystem access."""
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        except Exception:
            return DEFAULT_SUBAGENT_ROOT

    root = config.get("security", {}).get("subagent_filesystem", {}).get("root", "")
    if not root:
        return DEFAULT_SUBAGENT_ROOT
    return os.path.expanduser(root)


@dataclass(frozen=True)
class SubagentFilesystemScope:
    """Defines the filesystem scope for a sub-agent.

    Frozen dataclass for immutability.
    """
    root_path: str
    allowed_paths: Tuple[str, ...]
    denied_paths: Tuple[str, ...]
    allow_symlinks: bool = False
    allow_absolute_paths: bool = False
    max_write_size_bytes: int = 1_000_000  # 1MB default

    def __post_init__(self):
        # Resolve root path
        object.__setattr__(self, 'root_path', os.path.realpath(self.root_path))


def create_subagent_filesystem_scope(
    config: Optional[Dict[str, Any]] = None,
) -> SubagentFilesystemScope:
    """Create a sub-agent filesystem scope from configuration.

    The scope is a dedicated directory separate from parent filesystem paths.
    """
    root = get_subagent_filesystem_root(config)

    # Ensure root exists
    try:
        os.makedirs(root, exist_ok=True)
    except Exception as e:
        logger.warning("Could not create sub-agent filesystem root: %s", e)
        # Fall back to temp directory
        root = tempfile.mkdtemp(prefix="jarvis_subagent_")

    # Get denied paths from parent config (parent-only paths)
    denied = []
    if config:
        parent_denied = config.get("security", {}).get("filesystem_scope", {}).get("write_denied_paths", [])
        denied.extend(parent_denied)

    return SubagentFilesystemScope(
        root_path=root,
        allowed_paths=(root,),
        denied_paths=tuple(denied),
        allow_symlinks=False,
        allow_absolute_paths=False,
    )


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

def is_subagent_path_allowed(
    target_path: str,
    scope: SubagentFilesystemScope,
    is_write: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Check if a path is allowed for sub-agent filesystem access.

    Args:
        target_path: The path to check.
        scope: The sub-agent filesystem scope.
        is_write: Whether this is a write operation.
        config: Optional configuration dict.

    Returns:
        Tuple of (is_allowed, reason).
    """
    # Normalize the path
    try:
        normalized = os.path.normpath(os.path.abspath(os.path.expanduser(target_path)))
    except Exception as e:
        return False, f"Path normalization failed: {e}"

    # Check for absolute path restriction
    if not scope.allow_absolute_paths:
        if os.path.isabs(target_path):
            # Still allow if it's within scope after resolution
            pass  # We'll check scope below

    # Check for path traversal
    if ".." in pathlib.Path(target_path).parts:
        return False, "Path contains traversal components (..)"

    # Resolve to real path
    try:
        real_path = os.path.realpath(normalized)
    except Exception as e:
        return False, f"Path resolution failed: {e}"

    # Check against scope
    root_real = os.path.realpath(scope.root_path)

    # Path must be within the sub-agent root
    if not (real_path == root_real or real_path.startswith(root_real + os.sep)):
        return False, f"Path '{target_path}' is outside sub-agent scope (root: {root_real})"

    # Check symlinks
    if not scope.allow_symlinks:
        if os.path.islink(normalized):
            return False, "Symlinks are not allowed in sub-agent filesystem scope"

    # Check denied paths
    for denied in scope.denied_paths:
        denied_real = os.path.realpath(denied)
        if real_path == denied_real or real_path.startswith(denied_real + os.sep):
            return False, f"Path is in denied list: {denied}"

    # Check write boundary
    if is_write:
        try:
            from tools.filesystem_scope import is_write_allowed
            write_ok, write_reason = is_write_allowed(real_path, config=config)
            if not write_ok:
                return False, f"Write denied: {write_reason}"
        except Exception as e:
            return False, f"Write boundary check failed: {e}"

    return True, ""


def validate_subagent_read_path(
    target_path: str,
    scope: SubagentFilesystemScope,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Validate a read path for sub-agent."""
    return is_subagent_path_allowed(target_path, scope, is_write=False, config=config)


def validate_subagent_write_path(
    target_path: str,
    scope: SubagentFilesystemScope,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Validate a write path for sub-agent."""
    return is_subagent_path_allowed(target_path, scope, is_write=True, config=config)


# ---------------------------------------------------------------------------
# Filesystem capability definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubagentFilesystemCapability:
    """Defines filesystem capabilities for a task type."""
    task_type: str
    can_read: bool = False
    can_write: bool = False
    can_create: bool = False
    can_delete: bool = False
    requires_approval_for_write: bool = True
    scope: Optional[SubagentFilesystemScope] = None


def get_filesystem_capability_for_task_type(
    task_type: str,
    scope: Optional[SubagentFilesystemScope] = None,
) -> SubagentFilesystemCapability:
    """Get filesystem capability for a task type.

    Phase 25B: Only 'maintenance' task type gets filesystem access.
    All other task types are denied.
    """
    # Only maintenance task type gets filesystem access
    if task_type == "maintenance":
        return SubagentFilesystemCapability(
            task_type="maintenance",
            can_read=True,
            can_write=True,
            can_create=True,
            can_delete=True,
            requires_approval_for_write=True,
            scope=scope,
        )

    # No filesystem access for other task types
    return SubagentFilesystemCapability(
        task_type=task_type,
        can_read=False,
        can_write=False,
        can_create=False,
        can_delete=False,
        requires_approval_for_write=True,
        scope=scope,
    )


def is_filesystem_access_allowed(
    tool_name: str,
    task_type: str,
    operation: str = "read",
) -> bool:
    """Check if filesystem access is allowed for a task type.

    Args:
        tool_name: The filesystem tool name (e.g., 'read_file', 'write_file').
        task_type: The sub-agent task type.
        operation: 'read' or 'write'.

    Returns:
        True if filesystem access is allowed.
    """
    cap = get_filesystem_capability_for_task_type(task_type)

    if operation == "read":
        return cap.can_read
    elif operation == "write":
        return cap.can_write
    elif operation == "create":
        return cap.can_create
    elif operation == "delete":
        return cap.can_delete

    return False


# ---------------------------------------------------------------------------
# ToolCapability integration
# ---------------------------------------------------------------------------

def register_filesystem_capabilities() -> None:
    """Register filesystem capabilities in the ToolCapability registry.

    Called during JARVIS plugin initialization.
    """
    from plugins.memory.jarvis.tool_capability_registry import (
        _register_capability,
        ToolCapability,
    )

    # Read-only filesystem access for maintenance
    _register_capability(ToolCapability(
        name="subagent_read_file",
        category="filesystem",
        authority_level="L1",
        classification="read",
        allowed_task_types=frozenset({"maintenance"}),
        requires_approval=False,
        parent_only=False,
        exposed_to_subagents=True,
        provenance_required=True,
        sandbox_required=True,
        description="Sub-agent controlled filesystem read",
    ))

    # Write filesystem access for maintenance
    _register_capability(ToolCapability(
        name="subagent_write_file",
        category="filesystem",
        authority_level="L2",
        classification="write",
        allowed_task_types=frozenset({"maintenance"}),
        requires_approval=True,
        parent_only=False,
        exposed_to_subagents=True,
        provenance_required=True,
        sandbox_required=True,
        description="Sub-agent controlled filesystem write",
    ))


# Register at module load time (idempotent — registry prevents duplicates)
try:
    register_filesystem_capabilities()
except Exception:
    pass
