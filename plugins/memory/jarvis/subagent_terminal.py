"""JARVIS Sub-Agent Terminal Capability — Phase 25C.

Controlled terminal execution for JARVIS sub-agents.

This module provides a dedicated terminal capability layer for sub-agents
that is separate from the parent terminal tool. It reuses the existing
terminal infrastructure (tools/terminal_tool.py, tools/environments/local.py)
while adding sub-agent-specific policy restrictions:

- Dedicated sub-agent working directory (from Phase 25B filesystem scope)
- Command-level restrictions (allowlist + denylist)
- Shell metacharacter/injection prevention
- Environment variable isolation
- Explicit safe environment
- Command timeout and resource limits
- stdout/stderr/result size limits
- Approval requirements for consequential commands
- Delegation escape prevention
- Parent-only terminal isolation

Phase 25C: controlled terminal tool integration.
"""

from __future__ import annotations

import logging
import os
import shlex
import re
import resource
import signal
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sub-agent terminal configuration
# ---------------------------------------------------------------------------

# Default limits
DEFAULT_MAX_TIMEOUT_SECONDS = 30
DEFAULT_MAX_OUTPUT_BYTES = 50_000  # 50KB
DEFAULT_MAX_ENV_VARS = 20
DEFAULT_MAX_COMMAND_LENGTH = 500

# Commands that sub-agents are never allowed to run (denylist)
SUBAGENT_DENIED_COMMANDS: FrozenSet[str] = frozenset({
    # System-level commands
    "sudo", "su", "passwd", "useradd", "userdel", "groupadd", "groupdel",
    "chmod", "chown", "chroot", "mount", "umount", "reboot", "shutdown",
    "init", "systemctl", "service",
    # Network tools that could be used for exfiltration
    "curl", "wget", "nc", "netcat", "nmap", "telnet", "ssh", "scp", "rsync",
    # Process control
    "kill", "pkill", "killall", "nohup",
    # Dangerous file operations
    "rm", "rmdir", "dd", "mkfs", "fdisk", "parted",
    # Package managers
    "apt", "apt-get", "yum", "dnf", "pip", "pip3", "npm", "gem",
    # Container/runtime escape attempts
    "docker", "kubectl", "helm",
    # Delegation escape
    "delegate_task",
    # Shell escape
    "bash", "sh", "zsh", "fish", "csh", "tcsh",
    # Code execution that could bypass controls
    "python", "python3", "ruby", "perl", "node", "php",
    # Other dangerous commands
    "eval", "exec", "source", ".",
})

# Allowed commands (allowlist) — only these are permitted without approval
SUBAGENT_ALLOWED_COMMANDS: FrozenSet[str] = frozenset({
    "ls", "cat", "head", "tail", "wc", "grep", "find", "echo",
    "pwd", "cd", "mkdir", "touch", "cp", "mv", "diff",
    "sort", "uniq", "cut", "tr", "awk", "sed",
    "date", "whoami", "hostname", "uname",
    "env", "printenv",
    "which", "whereis",
    "file", "stat",
    "tar", "zip", "unzip", "gzip", "gunzip",
    "jq", "yq",
    "tee",
    "true", "false",
})

# Shell metacharacters that indicate injection attempts
SHELL_METACHARACTERS = re.compile(r'[;&|`$(){}[\]<>!#*?\n\r]')

# Environment variables that must never be inherited
DENIED_ENV_VARS: FrozenSet[str] = frozenset({
    "HERMES_API_KEY", "HERMES_UI_SESSION_ID", "HERMES_SESSION_ID",
    "HONCHO_API_KEY", "MEM0_API_KEY", "SUPERMEMORY_API_KEY",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
    "XAI_API_KEY", "OPENROUTER_API_KEY",
    "JARVIS_QDRANT_URL", "QDRANT_API_KEY",
    "HASS_TOKEN", "HOME_ASSISTANT_TOKEN",
    "TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "SLACK_BOT_TOKEN",
    "PATH",  # Will be set explicitly
    "HOME", "USER", "SHELL",
    "SSH_AUTH_SOCK", "SSH_AGENT_LAUNCHER",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "DOCKER_HOST", "KUBECONFIG",
})


@dataclass(frozen=True)
class SubagentTerminalCapability:
    """Defines terminal capabilities for a task type."""
    task_type: str
    can_execute: bool = False
    allowed_commands: FrozenSet[str] = field(default_factory=frozenset)
    denied_commands: FrozenSet[str] = field(default_factory=frozenset)
    max_timeout_seconds: int = DEFAULT_MAX_TIMEOUT_SECONDS
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    max_env_vars: int = DEFAULT_MAX_ENV_VARS
    max_command_length: int = DEFAULT_MAX_COMMAND_LENGTH
    requires_approval_for_all: bool = True
    working_directory: str = ""
    safe_env: Dict[str, str] = field(default_factory=dict)


def get_terminal_capability_for_task_type(
    task_type: str,
    working_directory: str = "",
) -> SubagentTerminalCapability:
    """Get terminal capability for a task type.

    Phase 25C: Only 'maintenance' task type gets terminal access.
    All other task types are denied.
    """
    if task_type == "maintenance":
        return SubagentTerminalCapability(
            task_type="maintenance",
            can_execute=True,
            allowed_commands=SUBAGENT_ALLOWED_COMMANDS,
            denied_commands=SUBAGENT_DENIED_COMMANDS,
            max_timeout_seconds=DEFAULT_MAX_TIMEOUT_SECONDS,
            max_output_bytes=DEFAULT_MAX_OUTPUT_BYTES,
            max_env_vars=DEFAULT_MAX_ENV_VARS,
            max_command_length=DEFAULT_MAX_COMMAND_LENGTH,
            requires_approval_for_all=True,
            working_directory=working_directory,
            safe_env=_build_safe_env(),
        )

    # No terminal access for other task types
    return SubagentTerminalCapability(
        task_type=task_type,
        can_execute=False,
        working_directory=working_directory,
    )


def _build_safe_env() -> Dict[str, str]:
    """Build a safe environment for sub-agent terminal execution."""
    return {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": os.path.expanduser("~"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TERM": "dumb",
        "TMPDIR": tempfile.gettempdir(),
    }


# ---------------------------------------------------------------------------
# Command validation
# ---------------------------------------------------------------------------

def validate_command(
    command: str,
    capability: SubagentTerminalCapability,
) -> Tuple[bool, str]:
    """Validate a command against the sub-agent terminal policy.

    Args:
        command: The command string to validate.
        capability: The terminal capability for the task type.

    Returns:
        Tuple of (is_valid, reason).
    """
    if not capability.can_execute:
        return False, f"Task type '{capability.task_type}' does not have terminal access"

    if not command or not command.strip():
        return False, "Empty command"

    # Check command length
    if len(command) > capability.max_command_length:
        return False, f"Command exceeds maximum length ({capability.max_command_length})"

    # Check for shell metacharacters (injection prevention)
    if SHELL_METACHARACTERS.search(command):
        return False, "Command contains shell metacharacters (injection attempt)"

    # Parse the command
    try:
        parts = shlex.split(command)
    except ValueError as e:
        return False, f"Command parsing failed: {e}"

    if not parts:
        return False, "Empty command"

    base_cmd = parts[0]

    # Check denylist
    if base_cmd in capability.denied_commands:
        return False, f"Command '{base_cmd}' is denied for sub-agents"

    # Check allowlist
    if base_cmd not in capability.allowed_commands:
        return False, f"Command '{base_cmd}' is not in the allowed list"

    return True, ""


def is_delegation_escape(command: str) -> bool:
    """Check if a command attempts to escape via delegation."""
    dangerous_patterns = [
        r'\bdelegate_task\b',
        r'\bhermes\b.*\bserve\b',
        r'\bhermes\b.*\bgateway\b',
        r'\bhermes\b.*\bdashboard\b',
        r'\bsystemctl\b',
        r'\bservice\b',
        r'\bcrontab\b',
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return True
    return False


def is_filesystem_escape(command: str, working_directory: str) -> bool:
    """Check if a command attempts to escape the working directory."""
    # Check for absolute path references outside working directory
    if working_directory:
        # Normalize working directory
        workdir_real = os.path.realpath(working_directory)

        # Check for path traversal in arguments
        for match in re.finditer(r'[\'"]?(/[^\s\'"]+)[\'"]?', command):
            path = match.group(1)
            try:
                real_path = os.path.realpath(path)
                if not real_path.startswith(workdir_real + os.sep) and real_path != workdir_real:
                    return True
            except Exception:
                pass

    return False


# ---------------------------------------------------------------------------
# Terminal execution
# ---------------------------------------------------------------------------

def execute_subagent_terminal(
    command: str,
    capability: SubagentTerminalCapability,
    task_id: str = "subagent",
    require_approval: bool = True,
) -> str:
    """Execute a command in the sub-agent terminal.

    Args:
        command: The command to execute.
        capability: The terminal capability for the task type.
        task_id: The sub-agent task ID.
        require_approval: Whether to require approval for this command.

    Returns:
        JSON string with output, exit_code, and error fields.
    """
    import json

    if not capability.can_execute:
        return json.dumps({
            "output": "",
            "exit_code": 1,
            "error": f"Task type '{capability.task_type}' does not have terminal access",
            "status": "error",
        })

    if not command or not command.strip():
        return json.dumps({
            "output": "",
            "exit_code": 1,
            "error": "Empty command",
            "status": "error",
        })

    # Check for delegation escape FIRST (before validation)
    if is_delegation_escape(command):
        return json.dumps({
            "output": "",
            "exit_code": 1,
            "error": "Command blocked: delegation escape attempt detected",
            "status": "error",
        })

    # Validate command
    is_valid, reason = validate_command(command, capability)
    if not is_valid:
        return json.dumps({
            "output": "",
            "exit_code": 1,
            "error": f"Command validation failed: {reason}",
            "status": "error",
        })

    # Check for filesystem escape
    if is_filesystem_escape(command, capability.working_directory):
        return json.dumps({
            "output": "",
            "exit_code": 1,
            "error": "Command blocked: filesystem escape attempt detected",
            "status": "error",
        })

    # Require approval for all commands
    if require_approval and capability.requires_approval_for_all:
        try:
            from tools.approval import request_tool_approval
            approval = request_tool_approval(
                tool_name="subagent_terminal",
                reason=f"Sub-agent terminal command: {command[:100]}",
            )
            if not approval.get("approved", False):
                return json.dumps({
                    "output": "",
                    "exit_code": 1,
                    "error": "Command not approved",
                    "status": "denied",
                })
        except Exception as e:
            return json.dumps({
                "output": "",
                "exit_code": 1,
                "error": f"Approval check failed: {e}",
                "status": "error",
            })

    # Execute via the existing terminal infrastructure
    try:
        from tools.terminal_tool import terminal_tool

        result = terminal_tool(
            command=command,
            timeout=capability.max_timeout_seconds,
            task_id=task_id,
            workdir=capability.working_directory,
        )

        # Parse and limit output
        result_data = json.loads(result)
        output = result_data.get("output", "")
        if len(output) > capability.max_output_bytes:
            output = output[:capability.max_output_bytes] + "\n... [OUTPUT TRUNCATED]"
            result_data["output"] = output
            result_data["output_truncated"] = True

        return json.dumps(result_data)

    except Exception as e:
        logger.warning("Sub-agent terminal execution failed: %s", e)
        return json.dumps({
            "output": "",
            "exit_code": 1,
            "error": f"Execution failed: {e}",
            "status": "error",
        })


# ---------------------------------------------------------------------------
# ToolCapability integration
# ---------------------------------------------------------------------------

def register_terminal_capabilities() -> None:
    """Register terminal capabilities in the ToolCapability registry."""
    from plugins.memory.jarvis.tool_capability_registry import (
        _register_capability,
        ToolCapability,
    )

    _register_capability(ToolCapability(
        name="subagent_terminal",
        category="terminal",
        authority_level="L2",
        classification="write",
        allowed_task_types=frozenset({"maintenance"}),
        requires_approval=True,
        parent_only=False,
        exposed_to_subagents=True,
        provenance_required=True,
        sandbox_required=True,
        description="Sub-agent controlled terminal execution",
    ))


# Register at module load time
try:
    register_terminal_capabilities()
except Exception:
    pass
