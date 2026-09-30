"""Tests for Phase 25C — Controlled Terminal Tool Integration."""

import pytest
from unittest.mock import patch

from plugins.memory.jarvis.subagent_terminal import (
    SubagentTerminalCapability,
    SUBAGENT_DENIED_COMMANDS,
    SUBAGENT_ALLOWED_COMMANDS,
    SHELL_METACHARACTERS,
    DENIED_ENV_VARS,
    get_terminal_capability_for_task_type,
    validate_command,
    is_delegation_escape,
    is_filesystem_escape,
    execute_subagent_terminal,
)
from plugins.memory.jarvis.tool_capability_registry import (
    get_capability,
    is_tool_permitted,
    is_tool_parent_only,
)


class TestTaskTypeTerminalAccess:
    """Test task-type-based terminal access control."""

    def test_maintenance_can_execute(self):
        """Maintenance task type can execute terminal commands."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")
        assert cap.can_execute is True

    def test_research_cannot_execute(self):
        """Research task type cannot execute terminal commands."""
        cap = get_terminal_capability_for_task_type("research", "/tmp/work")
        assert cap.can_execute is False

    def test_summarize_cannot_execute(self):
        """Summarize task type cannot execute terminal commands."""
        cap = get_terminal_capability_for_task_type("summarize", "/tmp/work")
        assert cap.can_execute is False

    def test_note_taking_cannot_execute(self):
        """Note-taking task type cannot execute terminal commands."""
        cap = get_terminal_capability_for_task_type("note_taking", "/tmp/work")
        assert cap.can_execute is False

    def test_task_execution_cannot_execute(self):
        """Task execution task type cannot execute terminal commands."""
        cap = get_terminal_capability_for_task_type("task_execution", "/tmp/work")
        assert cap.can_execute is False

    def test_maintenance_requires_approval(self):
        """Maintenance requires approval for all commands."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")
        assert cap.requires_approval_for_all is True

    def test_maintenance_has_safe_env(self):
        """Maintenance uses a safe environment."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")
        assert "PATH" in cap.safe_env
        assert "HOME" in cap.safe_env
        # Should NOT have sensitive API key vars
        sensitive_api_vars = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HERMES_API_KEY",
                             "JARVIS_QDRANT_URL", "AWS_ACCESS_KEY_ID"}
        for denied in sensitive_api_vars:
            assert denied not in cap.safe_env, f"Sensitive env var {denied} found in safe_env"


class TestCommandValidation:
    """Test command validation."""

    @pytest.fixture
    def maintenance_cap(self):
        return get_terminal_capability_for_task_type("maintenance", "/tmp/work")

    def test_allowed_command_passes(self, maintenance_cap):
        """Allowed command passes validation."""
        is_valid, reason = validate_command("ls -la", maintenance_cap)
        assert is_valid is True, f"Reason: {reason}"

    def test_denied_command_fails(self, maintenance_cap):
        """Denied command fails validation."""
        is_valid, reason = validate_command("rm -rf /", maintenance_cap)
        assert is_valid is False
        assert "denied" in reason.lower() or "not in the allowed list" in reason.lower()

    def test_empty_command_fails(self, maintenance_cap):
        """Empty command fails validation."""
        is_valid, reason = validate_command("", maintenance_cap)
        assert is_valid is False

    def test_shell_metacharacter_fails(self, maintenance_cap):
        """Shell metacharacters fail validation (injection prevention)."""
        is_valid, reason = validate_command("ls; cat /etc/passwd", maintenance_cap)
        assert is_valid is False
        assert "metacharacter" in reason.lower()

    def test_backtick_fails(self, maintenance_cap):
        """Backtick injection fails validation."""
        is_valid, reason = validate_command("ls `whoami`", maintenance_cap)
        assert is_valid is False

    def test_pipe_fails(self, maintenance_cap):
        """Pipe injection fails validation."""
        is_valid, reason = validate_command("ls | cat /etc/passwd", maintenance_cap)
        assert is_valid is False

    def test_dollar_fails(self, maintenance_cap):
        """Dollar expansion fails validation."""
        is_valid, reason = validate_command("ls $HOME", maintenance_cap)
        assert is_valid is False

    def test_semicolon_fails(self, maintenance_cap):
        """Semicolon injection fails validation."""
        is_valid, reason = validate_command("ls; rm -rf /", maintenance_cap)
        assert is_valid is False

    def test_long_command_fails(self, maintenance_cap):
        """Overly long command fails validation."""
        is_valid, reason = validate_command("ls " + "a" * 1000, maintenance_cap)
        assert is_valid is False

    def test_non_maintenance_fails(self):
        """Non-maintenance task types fail validation."""
        cap = get_terminal_capability_for_task_type("research", "/tmp/work")
        is_valid, reason = validate_command("ls", cap)
        assert is_valid is False
        assert "does not have terminal access" in reason


class TestDelegationEscapeDetection:
    """Test delegation escape detection."""

    def test_delegate_task_detected(self):
        """delegate_task in command is detected."""
        assert is_delegation_escape("delegate_task --goal test") is True

    def test_hermes_serve_detected(self):
        """hermes serve is detected."""
        assert is_delegation_escape("hermes serve --port 8080") is True

    def test_hermes_gateway_detected(self):
        """hermes gateway is detected."""
        assert is_delegation_escape("hermes gateway restart") is True

    def test_systemctl_detected(self):
        """systemctl is detected."""
        assert is_delegation_escape("systemctl restart hermes") is True

    def test_normal_command_not_detected(self):
        """Normal command is not detected as escape."""
        assert is_delegation_escape("ls -la /tmp") is False

    def test_cat_not_detected(self):
        """cat is not detected as escape."""
        assert is_delegation_escape("cat file.txt") is False


class TestFilesystemEscapeDetection:
    """Test filesystem escape detection."""

    def test_absolute_path_outside_workspace(self):
        """Absolute path outside workspace is detected."""
        assert is_filesystem_escape("ls /etc", "/tmp/work") is True

    def test_traversal_outside_workspace(self):
        """Path traversal outside workspace is detected."""
        assert is_filesystem_escape("ls ../../etc", "/tmp/work") is True

    def test_path_within_workspace(self):
        """Path within workspace is not detected."""
        assert is_filesystem_escape("ls /tmp/work/data", "/tmp/work") is False

    def test_workspace_root(self):
        """Workspace root is not detected."""
        assert is_filesystem_escape("ls /tmp/work", "/tmp/work") is False


class TestTerminalExecution:
    """Test terminal execution."""

    @pytest.fixture
    def maintenance_cap(self):
        return get_terminal_capability_for_task_type("maintenance", "/tmp/work")

    def test_denied_command_not_executed(self, maintenance_cap):
        """Denied command is not executed."""
        result = execute_subagent_terminal("rm -rf /", maintenance_cap, require_approval=False)
        result_dict = __import__("json").loads(result)
        assert result_dict["status"] == "error"
        assert "validation failed" in result_dict["error"].lower()

    def test_injection_not_executed(self, maintenance_cap):
        """Command with injection is not executed."""
        result = execute_subagent_terminal("ls; rm -rf /", maintenance_cap, require_approval=False)
        result_dict = __import__("json").loads(result)
        assert result_dict["status"] == "error"

    def test_unauthorized_task_type_not_executed(self):
        """Unauthorized task type cannot execute."""
        cap = get_terminal_capability_for_task_type("research", "/tmp/work")
        result = execute_subagent_terminal("ls", cap, require_approval=False)
        result_dict = __import__("json").loads(result)
        assert result_dict["status"] == "error"

    def test_delegation_escape_blocked(self, maintenance_cap):
        """Delegation escape attempt is blocked."""
        # crontab triggers delegation escape detection (checked before allowlist)
        result = execute_subagent_terminal("crontab -e", maintenance_cap, require_approval=False)
        result_dict = __import__("json").loads(result)
        assert result_dict["status"] == "error"
        assert "delegation escape" in result_dict["error"].lower()

    def test_approval_denied_blocks_execution(self, maintenance_cap):
        """Approval denial results in zero execution."""
        with patch("tools.approval.request_tool_approval") as mock_approval:
            mock_approval.return_value = {"approved": False}
            result = execute_subagent_terminal("ls", maintenance_cap, require_approval=True)
            result_dict = __import__("json").loads(result)
            assert result_dict["status"] == "denied"

    def test_shell_metacharacter_pattern(self):
        """Shell metacharacter regex catches injection."""
        assert SHELL_METACHARACTERS.search("ls;cat") is not None
        assert SHELL_METACHARACTERS.search("ls|cat") is not None
        assert SHELL_METACHARACTERS.search("ls`cat`") is not None
        assert SHELL_METACHARACTERS.search("ls$HOME") is not None
        assert SHELL_METACHARACTERS.search("ls -la") is None


class TestToolCapabilityIntegration:
    """Test integration with ToolCapability registry."""

    def test_subagent_terminal_registered(self):
        """subagent_terminal is registered in ToolCapability registry."""
        cap = get_capability("subagent_terminal")
        assert cap is not None
        assert cap.category == "terminal"
        assert cap.authority_level == "L2"
        assert cap.classification == "write"
        assert cap.exposed_to_subagents is True
        assert cap.parent_only is False
        assert cap.requires_approval is True
        assert cap.sandbox_required is True

    def test_terminal_still_parent_only(self):
        """Parent terminal tool remains parent-only."""
        assert is_tool_parent_only("terminal") is True
        assert is_tool_permitted("terminal", "maintenance") is False

    def test_subagent_terminal_permitted_for_maintenance(self):
        """Subagent terminal is permitted for maintenance."""
        assert is_tool_permitted("subagent_terminal", "maintenance") is True

    def test_subagent_terminal_not_permitted_for_research(self):
        """Subagent terminal is not permitted for research."""
        assert is_tool_permitted("subagent_terminal", "research") is False


class TestDenylistContent:
    """Test the command denylist."""

    def test_rm_denied(self):
        """rm is denied."""
        assert "rm" in SUBAGENT_DENIED_COMMANDS

    def test_sudo_denied(self):
        """sudo is denied."""
        assert "sudo" in SUBAGENT_DENIED_COMMANDS

    def test_ssh_denied(self):
        """ssh is denied."""
        assert "ssh" in SUBAGENT_DENIED_COMMANDS

    def test_curl_denied(self):
        """curl is denied."""
        assert "curl" in SUBAGENT_DENIED_COMMANDS

    def test_python_denied(self):
        """python is denied."""
        assert "python" in SUBAGENT_DENIED_COMMANDS

    def test_bash_denied(self):
        """bash is denied."""
        assert "bash" in SUBAGENT_DENIED_COMMANDS


class TestAllowlistContent:
    """Test the command allowlist."""

    def test_ls_allowed(self):
        """ls is allowed."""
        assert "ls" in SUBAGENT_ALLOWED_COMMANDS

    def test_cat_allowed(self):
        """cat is allowed."""
        assert "cat" in SUBAGENT_ALLOWED_COMMANDS

    def test_grep_allowed(self):
        """grep is allowed."""
        assert "grep" in SUBAGENT_ALLOWED_COMMANDS

    def test_echo_allowed(self):
        """echo is allowed."""
        assert "echo" in SUBAGENT_ALLOWED_COMMANDS


class TestDeniedEnvVars:
    """Test denied environment variables."""

    def test_api_keys_denied(self):
        """API keys are denied."""
        assert "OPENAI_API_KEY" in DENIED_ENV_VARS
        assert "ANTHROPIC_API_KEY" in DENIED_ENV_VARS

    def test_hermes_secrets_denied(self):
        """Hermes secrets are denied."""
        assert "HERMES_API_KEY" in DENIED_ENV_VARS
        assert "JARVIS_QDRANT_URL" in DENIED_ENV_VARS

    def test_cloud_keys_denied(self):
        """Cloud credentials are denied."""
        assert "AWS_ACCESS_KEY_ID" in DENIED_ENV_VARS
        assert "GOOGLE_APPLICATION_CREDENTIALS" in DENIED_ENV_VARS


class TestChildFailureIsolation:
    """Test child failure isolation."""

    def test_capability_frozen(self):
        """Terminal capability is frozen."""
        cap = get_terminal_capability_for_task_type("maintenance", "/tmp/work")
        with pytest.raises(AttributeError):
            cap.can_execute = False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
