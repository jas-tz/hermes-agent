"""Phase 36A — Capability Base Tests."""

import pytest

from plugins.memory.jarvis.capabilities.base import (
    CapabilityCategory,
    CapabilityInputType,
    CapabilityOutputType,
    CapabilityRequirements,
    CapabilityContext,
    CapabilityResult,
    Capability,
    CapabilityRegistry,
    get_registry,
    reset_registry,
)


@pytest.fixture(autouse=True)
def reset_registry_fixture():
    reset_registry()
    yield
    reset_registry()


# ---------------------------------------------------------------------------
# Mock capabilities for testing
# ---------------------------------------------------------------------------

class _MockCap(Capability):
    def get_name(self) -> str:
        return "mock_cap"

    def get_description(self) -> str:
        return "A mock capability"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_RETRIEVAL

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L1",
            required_tools=frozenset({"hermes_rag_search"}),
            input_type=CapabilityInputType.QUERY,
            output_type=CapabilityOutputType.STRUCTURED,
            requires_approval=False,
            parallelizable=True,
            max_execution_time_seconds=60,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        return CapabilityResult.success_result(
            data={"result": "test"},
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )


class _MockCapL0(Capability):
    def get_name(self) -> str:
        return "mock_l0"

    def get_description(self) -> str:
        return "L0 mock capability"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_RETRIEVAL

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L0",
            required_tools=frozenset({"hermes_rag_search"}),
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        return CapabilityResult.success_result(data={"result": "l0"})


class _MockCapL3(Capability):
    def get_name(self) -> str:
        return "mock_l3"

    def get_description(self) -> str:
        return "L3 mock capability"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_STORAGE

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L3",
            required_tools=frozenset({"hermes_rag_delete", "hermes_rag_store"}),
            requires_approval=True,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        return CapabilityResult.success_result(data={"result": "l3"})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCapabilityRequirements:
    def test_default_values(self):
        reqs = CapabilityRequirements()
        assert reqs.min_authority_level == "L0"
        assert reqs.input_type == CapabilityInputType.TEXT
        assert reqs.requires_approval is False
        assert reqs.parallelizable is True
        assert reqs.max_execution_time_seconds == 300

    def test_custom_values(self):
        reqs = CapabilityRequirements(
            min_authority_level="L3",
            required_tools=frozenset({"hermes_rag_delete"}),
            input_type=CapabilityInputType.QUERY,
            output_type=CapabilityOutputType.STRUCTURED,
            requires_approval=True,
            parallelizable=False,
            max_execution_time_seconds=60,
        )
        assert reqs.min_authority_level == "L3"
        assert reqs.required_tools == frozenset({"hermes_rag_delete"})
        assert reqs.requires_approval is True

    def test_frozen(self):
        reqs = CapabilityRequirements()
        with pytest.raises(AttributeError):
            reqs.min_authority_level = "L3"

    def test_empty_dependencies(self):
        reqs = CapabilityRequirements()
        assert reqs.dependencies == frozenset()


class TestCapabilityContext:
    def test_creation(self):
        ctx = CapabilityContext(
            execution_id="exec-001",
            subagent_id="sa-001",
            parent_session_id="parent-001",
            task_goal="Test goal",
            task_type="research",
            authority_level="L0",
            delegation_depth=1,
        )
        assert ctx.execution_id == "exec-001"
        assert ctx.subagent_id == "sa-001"
        assert ctx.parameters == {}
        assert ctx.dependency_results == {}

    def test_with_parameters(self):
        ctx = CapabilityContext(
            execution_id="exec-002",
            subagent_id="sa-002",
            parent_session_id="parent-002",
            task_goal="Goal",
            task_type="maintenance",
            authority_level="L3",
            delegation_depth=2,
            parameters={"key": "value"},
            dependency_results={"dep": {"result": "data"}},
        )
        assert ctx.parameters == {"key": "value"}
        assert ctx.dependency_results == {"dep": {"result": "data"}}

    def test_frozen(self):
        ctx = CapabilityContext(
            execution_id="exec-003",
            subagent_id="sa-003",
            parent_session_id="parent-003",
            task_goal="Goal",
            task_type="research",
            authority_level="L0",
            delegation_depth=1,
        )
        with pytest.raises(AttributeError):
            ctx.subagent_id = "modified"


class TestCapabilityResult:
    def test_success_result(self):
        result = CapabilityResult.success_result(
            data={"key": "value"},
            message="Success",
            execution_id="exec-001",
            capability_name="test",
            execution_time_seconds=1.5,
        )
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.message == "Success"
        assert result.error is None

    def test_failure_result(self):
        result = CapabilityResult.failure_result(error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_default_values(self):
        result = CapabilityResult(success=True)
        assert result.data is None
        assert result.message == ""
        assert result.error is None
        assert result.composable is True

    def test_to_dict_success(self):
        result = CapabilityResult.success_result(data={"result": "test"}, execution_id="e3")
        d = result.to_dict()
        assert d["success"] is True
        assert d["data"] == {"result": "test"}

    def test_to_dict_failure(self):
        result = CapabilityResult.failure_result(error="error")
        d = result.to_dict()
        assert d["success"] is False
        assert d["error"] == "error"

    def test_frozen(self):
        result = CapabilityResult(success=True)
        with pytest.raises(AttributeError):
            result.success = False


class TestCapability:
    def test_can_execute_with_sufficient_authority(self):
        cap = _MockCap()
        assert cap.can_execute("L3", frozenset({"hermes_rag_search", "hermes_rag_store"})) is True

    def test_can_execute_with_exact_authority(self):
        cap = _MockCap()
        assert cap.can_execute("L1", frozenset({"hermes_rag_search"})) is True

    def test_can_execute_with_insufficient_authority(self):
        cap = _MockCap()
        assert cap.can_execute("L0", frozenset({"hermes_rag_search"})) is False

    def test_can_execute_with_missing_tools(self):
        cap = _MockCap()
        assert cap.can_execute("L3", frozenset({"hermes_rag_store"})) is False

    def test_can_execute_with_empty_tools(self):
        cap = _MockCap()
        assert cap.can_execute("L3", frozenset()) is False

    def test_can_execute_l0(self):
        cap = _MockCapL0()
        assert cap.can_execute("L0", frozenset({"hermes_rag_search"})) is True

    def test_can_execute_l3_all_requirements(self):
        cap = _MockCapL3()
        assert cap.can_execute("L3", frozenset({"hermes_rag_delete", "hermes_rag_store"})) is True

    def test_can_execute_l3_partial_tools(self):
        cap = _MockCapL3()
        assert cap.can_execute("L3", frozenset({"hermes_rag_delete"})) is False

    def test_to_dict(self):
        cap = _MockCap()
        d = cap.to_dict()
        assert d["name"] == "mock_cap"
        assert d["description"] == "A mock capability"
        assert d["category"] == "data_retrieval"
        assert d["requirements"]["min_authority_level"] == "L1"
        assert d["requirements"]["required_tools"] == ["hermes_rag_search"]

    def test_execute(self):
        cap = _MockCap()
        ctx = CapabilityContext(
            execution_id="exec-005",
            subagent_id="sa-005",
            parent_session_id="parent-005",
            task_goal="Goal",
            task_type="research",
            authority_level="L0",
            delegation_depth=1,
        )
        result = cap.execute(ctx)
        assert result.success is True
        assert result.data == {"result": "test"}
        assert result.execution_id == "exec-005"

    def test_validate(self):
        cap = _MockCap()
        ctx = CapabilityContext(
            execution_id="exec-006",
            subagent_id="sa-006",
            parent_session_id="parent-006",
            task_goal="Goal",
            task_type="research",
            authority_level="L0",
            delegation_depth=1,
        )
        valid, reason = cap.validate(ctx)
        assert valid is True
        assert reason == ""


class TestCapabilityRegistry:
    def test_register_and_get(self):
        registry = CapabilityRegistry()
        registry.register(_MockCap)
        assert "mock_cap" in registry.list_capabilities()
        cap_class = registry.get("mock_cap")
        assert cap_class is _MockCap

    def test_register_duplicate_raises(self):
        registry = CapabilityRegistry()
        registry.register(_MockCap)
        with pytest.raises(ValueError):
            registry.register(_MockCap)

    def test_unregister(self):
        registry = CapabilityRegistry()
        registry.register(_MockCap)
        registry.unregister("mock_cap")
        assert "mock_cap" not in registry.list_capabilities()

    def test_unregister_nonexistent(self):
        registry = CapabilityRegistry()
        registry.unregister("nonexistent")

    def test_get_nonexistent(self):
        registry = CapabilityRegistry()
        assert registry.get("nonexistent") is None

    def test_get_capability(self):
        registry = CapabilityRegistry()
        registry.register(_MockCap)
        cap = registry.get_capability("mock_cap")
        assert cap is not None
        assert cap.get_name() == "mock_cap"

    def test_list_capabilities_empty(self):
        registry = CapabilityRegistry()
        assert registry.list_capabilities() == []

    def test_list_capabilities_multiple(self):
        registry = CapabilityRegistry()
        registry.register(_MockCap)
        registry.register(_MockCapL0)
        assert len(registry.list_capabilities()) == 2

    def test_find_by_category(self):
        registry = CapabilityRegistry()
        registry.register(_MockCap)
        registry.register(_MockCapL0)
        registry.register(_MockCapL3)
        retrieval_caps = registry.find_by_category(CapabilityCategory.DATA_RETRIEVAL)
        assert len(retrieval_caps) == 2

    def test_find_by_category_empty(self):
        registry = CapabilityRegistry()
        registry.register(_MockCapL3)
        assert len(registry.find_by_category(CapabilityCategory.DATA_RETRIEVAL)) == 0

    def test_find_executable(self):
        registry = CapabilityRegistry()
        registry.register(_MockCap)
        registry.register(_MockCapL0)
        registry.register(_MockCapL3)
        executable = registry.find_executable("L3", frozenset({"hermes_rag_search", "hermes_rag_delete", "hermes_rag_store"}))
        names = {c.get_name() for c in executable}
        assert "mock_cap" in names
        assert "mock_l0" in names
        assert "mock_l3" in names

    def test_find_executable_filtered_by_authority(self):
        registry = CapabilityRegistry()
        registry.register(_MockCap)
        registry.register(_MockCapL0)
        registry.register(_MockCapL3)
        executable = registry.find_executable("L1", frozenset({"hermes_rag_search"}))
        names = {c.get_name() for c in executable}
        assert "mock_cap" in names
        assert "mock_l0" in names
        assert "mock_l3" not in names

    def test_find_executable_filtered_by_tools(self):
        registry = CapabilityRegistry()
        registry.register(_MockCap)
        registry.register(_MockCapL0)
        registry.register(_MockCapL3)
        executable = registry.find_executable("L3", frozenset({"hermes_rag_search"}))
        names = {c.get_name() for c in executable}
        assert "mock_cap" in names
        assert "mock_l0" in names
        assert "mock_l3" not in names

    def test_to_dict(self):
        registry = CapabilityRegistry()
        registry.register(_MockCap)
        registry.register(_MockCapL0)
        d = registry.to_dict()
        assert len(d["capabilities"]) == 2


class TestGlobalRegistry:
    def test_get_registry(self):
        registry = get_registry()
        assert isinstance(registry, CapabilityRegistry)

    def test_reset_registry(self):
        registry = get_registry()
        registry.register(_MockCap)
        assert len(registry.list_capabilities()) == 1
        reset_registry()
        registry = get_registry()
        assert len(registry.list_capabilities()) == 0


class TestEnums:
    def test_capability_category_values(self):
        assert CapabilityCategory.DATA_RETRIEVAL.value == "data_retrieval"
        assert CapabilityCategory.DATA_STORAGE.value == "data_storage"
        assert CapabilityCategory.EXTERNAL_INTEGRATION.value == "external_integration"

    def test_capability_input_type_values(self):
        assert CapabilityInputType.TEXT.value == "text"
        assert CapabilityInputType.QUERY.value == "query"

    def test_capability_output_type_values(self):
        assert CapabilityOutputType.TEXT.value == "text"
        assert CapabilityOutputType.STRUCTURED.value == "structured"
        assert CapabilityOutputType.BOOLEAN.value == "boolean"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
