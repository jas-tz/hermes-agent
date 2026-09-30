"""
Phase 31A — JARVIS Sub-Agent Authorization Boundary Certification.

Tests that every sub-agent execution path remains behind the Phase 30
authorization boundary and that no bypass can reach tool dispatch.
"""

import pathlib
import inspect

import plugins.memory.jarvis.__init__ as jarvis_init


def test_dispatch_has_single_call_site():
    source = inspect.getsource(jarvis_init)
    assert source.count("_dispatch_jarvis_tool_call(") == 2
    # One definition + exactly one invocation.


def test_dispatch_definition_exists():
    assert hasattr(jarvis_init, "_dispatch_jarvis_tool_call")
    assert callable(jarvis_init._dispatch_jarvis_tool_call)


def test_provider_has_authorization_boundary():
    provider = jarvis_init.JARVISSubagentProvider
    assert hasattr(provider, "handle_tool_call")
    assert callable(provider.handle_tool_call)


def test_dispatch_is_after_authorization_in_provider():
    source = inspect.getsource(
        jarvis_init.JARVISSubagentProvider.handle_tool_call
    )

    dispatch_pos = source.find("_dispatch_jarvis_tool_call(")
    assert dispatch_pos >= 0

    for required_gate in (
        "is_tool_parent_only",
        "resolve_task_type_tools",
        "is_tool_permitted",
        "requires_approval",
        "validate_command",
        "is_subagent_path_allowed",
        "record_authorization_allowed",
    ):
        assert source.find(required_gate) < dispatch_pos, (
            f"{required_gate} must execute before dispatch"
        )


def test_bypass_gate_is_stripped():
    source = inspect.getsource(
        jarvis_init.JARVISSubagentProvider.handle_tool_call
    )
    assert "_continuity_bypass_gate" in source
    assert "pop(" in source


def test_provenance_isolation_exists_before_dispatch():
    source = inspect.getsource(
        jarvis_init.JARVISSubagentProvider.handle_tool_call
    )

    dispatch_pos = source.find("_dispatch_jarvis_tool_call(")
    provenance_pos = source.find("copy.deepcopy")

    assert provenance_pos >= 0
    assert provenance_pos < dispatch_pos


def test_authorization_recording_precedes_dispatch():
    source = inspect.getsource(
        jarvis_init.JARVISSubagentProvider.handle_tool_call
    )

    dispatch_pos = source.find("_dispatch_jarvis_tool_call(")
    allowed_pos = source.find("record_authorization_allowed")

    assert allowed_pos >= 0
    assert allowed_pos < dispatch_pos


def test_capability_registry_precedes_dispatch():
    source = inspect.getsource(
        jarvis_init.JARVISSubagentProvider.handle_tool_call
    )

    dispatch_pos = source.find("_dispatch_jarvis_tool_call(")

    for symbol in (
        "is_tool_permitted",
        "requires_approval",
    ):
        pos = source.find(symbol)
        assert pos >= 0
        assert pos < dispatch_pos


def test_terminal_policy_precedes_dispatch():
    source = inspect.getsource(
        jarvis_init.JARVISSubagentProvider.handle_tool_call
    )

    dispatch_pos = source.find("_dispatch_jarvis_tool_call(")
    terminal_pos = source.find("validate_command")

    assert terminal_pos >= 0
    assert terminal_pos < dispatch_pos


def test_filesystem_policy_precedes_dispatch():
    source = inspect.getsource(
        jarvis_init.JARVISSubagentProvider.handle_tool_call
    )

    dispatch_pos = source.find("_dispatch_jarvis_tool_call(")
    filesystem_pos = source.find("is_subagent_path_allowed")

    assert filesystem_pos >= 0
    assert filesystem_pos < dispatch_pos


def test_parent_only_policy_precedes_dispatch():
    source = inspect.getsource(
        jarvis_init.JARVISSubagentProvider.handle_tool_call
    )

    dispatch_pos = source.find("_dispatch_jarvis_tool_call(")
    parent_only_pos = source.find("is_tool_parent_only")

    assert parent_only_pos >= 0
    assert parent_only_pos < dispatch_pos


def test_task_type_policy_precedes_dispatch():
    source = inspect.getsource(
        jarvis_init.JARVISSubagentProvider.handle_tool_call
    )

    dispatch_pos = source.find("_dispatch_jarvis_tool_call(")
    task_type_pos = source.find("resolve_task_type_tools")

    assert task_type_pos >= 0
    assert task_type_pos < dispatch_pos


def test_dispatch_is_not_imported_elsewhere():
    root = pathlib.Path("plugins/memory/jarvis")
    matches = []

    for path in root.glob("*.py"):
        text = path.read_text()
        # __init__.py is the source file — skip it
        if path.name == "__init__.py":
            continue
        if "_dispatch_jarvis_tool_call(" in text:
            matches.append(str(path))

    assert matches == []


def test_delegate_tool_does_not_directly_dispatch():
    path = pathlib.Path("tools/delegate_tool.py")
    if path.exists():
        text = path.read_text()
        assert "_dispatch_jarvis_tool_call(" not in text


if __name__ == "__main__":
    print("Phase 31A boundary certification tests loaded.")
