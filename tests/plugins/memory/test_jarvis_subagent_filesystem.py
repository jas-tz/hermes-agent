"""Tests for Phase 25B — Controlled Filesystem Tool Integration."""

import os
import tempfile
import pytest

from plugins.memory.jarvis.subagent_filesystem import (
    SubagentFilesystemScope,
    SubagentFilesystemCapability,
    create_subagent_filesystem_scope,
    is_subagent_path_allowed,
    validate_subagent_read_path,
    validate_subagent_write_path,
    get_subagent_filesystem_root,
    get_filesystem_capability_for_task_type,
    is_filesystem_access_allowed,
)
from plugins.memory.jarvis.tool_capability_registry import (
    get_capability,
    is_tool_permitted,
    is_tool_parent_only,
    get_permitted_tools_for_task_type,
)


@pytest.fixture
def temp_subagent_root(tmp_path):
    """Create a temporary sub-agent filesystem root."""
    root = tmp_path / "subagent-workspace"
    root.mkdir()
    return str(root)


@pytest.fixture
def filesystem_scope(temp_subagent_root):
    """Create a filesystem scope for testing."""
    return SubagentFilesystemScope(
        root_path=temp_subagent_root,
        allowed_paths=(temp_subagent_root,),
        denied_paths=(),
        allow_symlinks=False,
        allow_absolute_paths=False,
    )


class TestFilesystemScopeCreation:
    """Test filesystem scope creation."""

    def test_scope_is_frozen(self, filesystem_scope):
        """Filesystem scope is immutable."""
        with pytest.raises(AttributeError):
            filesystem_scope.root_path = "/tmp"

    def test_scope_resolves_root_path(self, temp_subagent_root):
        """Root path is resolved to real path."""
        # Pass the temporary root explicitly: with no config the function reads
        # the real config and falls back to ~/.hermes/subagent-workspace.
        config = {"security": {"subagent_filesystem": {"root": temp_subagent_root}}}
        scope = create_subagent_filesystem_scope(config)
        assert os.path.isabs(scope.root_path)
        assert scope.root_path == os.path.realpath(temp_subagent_root)

    def test_default_root_uses_hermes_home(self):
        """Default root is under ~/.hermes."""
        root = get_subagent_filesystem_root()
        assert "subagent" in root.lower() or "hermes" in root.lower()


class TestPathValidation:
    """Test path validation within scope."""

    def test_allowed_read_within_scope(self, filesystem_scope, temp_subagent_root):
        """Read within scope is allowed."""
        target = os.path.join(temp_subagent_root, "data", "file.txt")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w") as f:
            f.write("test")

        allowed, reason = validate_subagent_read_path(target, filesystem_scope)
        assert allowed is True, f"Reason: {reason}"

    def test_allowed_write_within_scope(self, filesystem_scope, temp_subagent_root):
        """Write within scope is allowed."""
        target = os.path.join(temp_subagent_root, "output.txt")
        allowed, reason = validate_subagent_write_path(target, filesystem_scope)
        assert allowed is True, f"Reason: {reason}"

    def test_path_traversal_denied(self, filesystem_scope, temp_subagent_root):
        """Path traversal attempts are denied."""
        target = os.path.join(temp_subagent_root, "..", "etc", "passwd")
        allowed, reason = validate_subagent_read_path(target, filesystem_scope)
        assert allowed is False
        assert "traversal" in reason.lower() or "outside" in reason.lower()

    def test_absolute_path_escape_denied(self, filesystem_scope):
        """Absolute path outside scope is denied."""
        allowed, reason = validate_subagent_read_path("/etc/passwd", filesystem_scope)
        assert allowed is False
        assert "outside" in reason.lower()

    def test_symlink_denied(self, filesystem_scope, temp_subagent_root, tmp_path):
        """Symlinks are denied when not allowed."""
        # Create a file outside scope
        outside_file = tmp_path / "secret.txt"
        with open(outside_file, "w") as f:
            f.write("secret")

        # Create symlink inside scope pointing outside
        link_path = os.path.join(temp_subagent_root, "link.txt")
        try:
            os.symlink(str(outside_file), link_path)
            allowed, reason = validate_subagent_read_path(link_path, filesystem_scope)
            assert allowed is False
        except OSError:
            # Symlinks may not be supported on all platforms
            pass


class TestTaskTypeFilesystemAccess:
    """Test task-type-based filesystem access control."""

    def test_maintenance_can_read(self, filesystem_scope):
        """Maintenance task type can read files."""
        cap = get_filesystem_capability_for_task_type("maintenance", filesystem_scope)
        assert cap.can_read is True

    def test_maintenance_can_write(self, filesystem_scope):
        """Maintenance task type can write files."""
        cap = get_filesystem_capability_for_task_type("maintenance", filesystem_scope)
        assert cap.can_write is True

    def test_maintenance_requires_approval(self, filesystem_scope):
        """Maintenance requires approval for writes."""
        cap = get_filesystem_capability_for_task_type("maintenance", filesystem_scope)
        assert cap.requires_approval_for_write is True

    def test_research_cannot_access_filesystem(self, filesystem_scope):
        """Research task type cannot access filesystem."""
        cap = get_filesystem_capability_for_task_type("research", filesystem_scope)
        assert cap.can_read is False
        assert cap.can_write is False

    def test_summarize_cannot_access_filesystem(self, filesystem_scope):
        """Summarize task type cannot access filesystem."""
        cap = get_filesystem_capability_for_task_type("summarize", filesystem_scope)
        assert cap.can_read is False
        assert cap.can_write is False

    def test_note_taking_cannot_access_filesystem(self, filesystem_scope):
        """Note-taking task type cannot access filesystem."""
        cap = get_filesystem_capability_for_task_type("note_taking", filesystem_scope)
        assert cap.can_read is False
        assert cap.can_write is False

    def test_task_execution_cannot_access_filesystem(self, filesystem_scope):
        """Task execution task type cannot access filesystem."""
        cap = get_filesystem_capability_for_task_type("task_execution", filesystem_scope)
        assert cap.can_read is False
        assert cap.can_write is False

    def test_is_filesystem_access_allowed(self):
        """Test the is_filesystem_access_allowed helper."""
        assert is_filesystem_access_allowed("subagent_read_file", "maintenance", "read") is True
        assert is_filesystem_access_allowed("subagent_write_file", "maintenance", "write") is True
        assert is_filesystem_access_allowed("subagent_read_file", "research", "read") is False
        assert is_filesystem_access_allowed("subagent_write_file", "research", "write") is False


class TestToolCapabilityIntegration:
    """Test integration with ToolCapability registry."""

    def test_subagent_read_file_registered(self):
        """subagent_read_file is registered in ToolCapability registry."""
        cap = get_capability("subagent_read_file")
        assert cap is not None
        assert cap.category == "filesystem"
        assert cap.authority_level == "L1"
        assert cap.classification == "read"
        assert cap.exposed_to_subagents is True
        assert cap.parent_only is False

    def test_subagent_write_file_registered(self):
        """subagent_write_file is registered in ToolCapability registry."""
        cap = get_capability("subagent_write_file")
        assert cap is not None
        assert cap.category == "filesystem"
        assert cap.authority_level == "L2"
        assert cap.classification == "write"
        assert cap.exposed_to_subagents is True
        assert cap.parent_only is False
        assert cap.requires_approval is True

    def test_read_file_still_parent_only(self):
        """Parent read_file remains parent-only."""
        assert is_tool_parent_only("read_file") is True
        assert is_tool_permitted("read_file", "maintenance") is False

    def test_write_file_still_parent_only(self):
        """Parent write_file remains parent-only."""
        assert is_tool_parent_only("write_file") is True
        assert is_tool_permitted("write_file", "maintenance") is False

    def test_subagent_filesystem_permitted_for_maintenance(self):
        """Subagent filesystem tools are permitted for maintenance."""
        assert is_tool_permitted("subagent_read_file", "maintenance") is True
        assert is_tool_permitted("subagent_write_file", "maintenance") is True

    def test_subagent_filesystem_not_permitted_for_research(self):
        """Subagent filesystem tools are not permitted for research."""
        assert is_tool_permitted("subagent_read_file", "research") is False
        assert is_tool_permitted("subagent_write_file", "research") is False


class TestForgedToolCalls:
    """Test that forged filesystem tool calls are rejected."""

    def test_forged_filesystem_tool_rejected(self):
        """Forged filesystem tool names are rejected."""
        assert is_tool_permitted("subagent_read_fake", "maintenance") is False
        assert is_tool_permitted("subagent_write_fake", "maintenance") is False

    def test_parent_read_file_forged_as_subagent(self):
        """Parent read_file cannot be accessed via subagent."""
        assert is_tool_permitted("read_file", "maintenance") is False


class TestMultiStepFilesystemOperations:
    """Test multi-step filesystem operations."""

    def test_read_then_write_within_scope(self, filesystem_scope, temp_subagent_root):
        """Multi-step: read then write within scope."""
        # Create initial file
        input_file = os.path.join(temp_subagent_root, "input.txt")
        with open(input_file, "w") as f:
            f.write("initial data")

        output_file = os.path.join(temp_subagent_root, "output.txt")

        # Both should be allowed
        read_ok, _ = validate_subagent_read_path(input_file, filesystem_scope)
        write_ok, _ = validate_subagent_write_path(output_file, filesystem_scope)

        assert read_ok is True
        assert write_ok is True

    def test_sequential_writes_within_scope(self, filesystem_scope, temp_subagent_root):
        """Multiple sequential writes within scope."""
        for i in range(5):
            target = os.path.join(temp_subagent_root, f"file_{i}.txt")
            allowed, _ = validate_subagent_write_path(target, filesystem_scope)
            assert allowed is True


class TestChildFailureIsolation:
    """Test that child filesystem failure doesn't corrupt parent."""

    def test_scope_isolation(self, filesystem_scope):
        """Each child has its own scope."""
        scope2 = SubagentFilesystemScope(
            root_path="/tmp/other",
            allowed_paths=("/tmp/other",),
            denied_paths=(),
        )
        assert filesystem_scope.root_path != scope2.root_path


class TestLifecycleIntegration:
    """Test that lifecycle integration works."""

    def test_scope_immutable_during_lifecycle(self, filesystem_scope):
        """Filesystem scope doesn't change during lifecycle."""
        original_root = filesystem_scope.root_path

        # Simulate lifecycle transitions
        from plugins.memory.jarvis.subagent_types import (
            create_execution_context,
            SubagentLifecycleManager,
        )

        ctx = create_execution_context(
            child_id="sa-test",
            parent_session_id="parent",
            task_goal="test",
            task_type="maintenance",
            delegation_depth=1,
        )
        manager = SubagentLifecycleManager(ctx)

        manager.transition("initializing")
        assert filesystem_scope.root_path == original_root

        manager.transition("running")
        assert filesystem_scope.root_path == original_root

        manager.transition("completed")
        assert filesystem_scope.root_path == original_root


class TestCleanup:
    """Test cleanup after execution."""

    def test_scope_can_be_released(self, filesystem_scope):
        """Scope object can be released after execution."""
        del filesystem_scope


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
