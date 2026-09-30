"""JARVIS Capability Registry — Phase 36B.

Concrete capability registry that integrates with the existing JARVIS
tool authorization layer (tool_capability_registry.py).

This registry:
1. Validates capabilities against the existing tool capability registry
2. Ensures capabilities cannot declare tools outside the authorization boundary
3. Provides capability discovery for future orchestration
4. Does NOT modify existing production authorization code
5. Does NOT connect external systems yet

Design principles:
- Capabilities are validated at registration time against the auth layer
- A capability requiring tools not in the tool registry is rejected
- The capability registry is a superset of the tool registry's exposed tools
"""

from __future__ import annotations

import logging
from typing import Any, Dict, FrozenSet, List, Optional, Type

from plugins.memory.jarvis.capabilities.base import (
    Capability,
    CapabilityCategory,
    CapabilityRegistry,
    get_registry as get_base_registry,
    reset_registry as reset_base_registry,
)
from plugins.memory.jarvis.tool_capability_registry import (
    TOOL_CAPABILITY_REGISTRY,
    get_permitted_tools_for_task_type,
    get_capability,
)

logger = logging.getLogger(__name__)


def _is_tool_exposed_to_subagents(tool_name: str) -> bool:
    """Local helper — avoids importing a nonexistent function from tool_capability_registry."""
    cap = TOOL_CAPABILITY_REGISTRY.get(tool_name)
    if cap is None:
        return False
    return cap.exposed_to_subagents and not cap.parent_only


def _get_tool_level(tool_name: str) -> str:
    """Return the authority level string for a registered tool, else 'L3' (fail-closed)."""
    cap = TOOL_CAPABILITY_REGISTRY.get(tool_name)
    if cap is None:
        return "L3"
    return cap.authority_level


class CapabilityRegistrationError(Exception):
    """Raised when a capability fails registration validation."""
    pass


class JARVISCapabilityRegistry(CapabilityRegistry):
    """Capability registry integrated with the JARVIS authorization layer.

    Validates capabilities against the existing tool capability registry
    to ensure the capability layer cannot bypass the authorization boundary.
    """

    def register(self, capability_class: Type[Capability]) -> None:
        """Register a capability with authorization-layer validation.

        Validates:
        1. Capability name is unique
        2. All required tools exist in the JARVIS tool registry
        3. All required tools are exposed to sub-agents
        4. Required tools are consistent with the capability's authority level

        Raises:
            CapabilityRegistrationError: If validation fails.
        """
        temp = capability_class()
        name = temp.get_name()

        # Check uniqueness
        if name in self._capabilities:
            raise CapabilityRegistrationError(
                f"Capability '{name}' already registered"
            )

        # Validate against tool capability registry
        reqs = temp.get_requirements()
        issues = self._validate_against_tool_registry(reqs, name)

        if issues:
            raise CapabilityRegistrationError(
                f"Capability '{name}' failed validation: {'; '.join(issues)}"
            )

        self._capabilities[name] = capability_class
        logger.debug("Registered validated capability: %s", name)

    def _validate_against_tool_registry(
        self, reqs: Any, capability_name: str
    ) -> List[str]:
        """Validate capability requirements against the tool registry."""
        issues = []

        for tool_name in reqs.required_tools:
            # Check tool exists in registry
            if tool_name not in TOOL_CAPABILITY_REGISTRY:
                issues.append(
                    f"Tool '{tool_name}' not found in tool registry"
                )
                continue

            # Check tool is exposed to sub-agents
            if not _is_tool_exposed_to_subagents(tool_name):
                issues.append(
                    f"Tool '{tool_name}' is not exposed to sub-agents"
                )

            # Check authority level consistency
            tool_level = _get_tool_level(tool_name)
            level_order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}

            if level_order.get(tool_level, 0) > level_order.get(reqs.min_authority_level, 0):
                issues.append(
                    f"Tool '{tool_name}' requires level {tool_level} "
                    f"but capability only declares {reqs.min_authority_level}"
                )

        return issues

    def get_capability_for_task_type(
        self, task_type: str
    ) -> List[Capability]:
        """Get all capabilities that can execute for a task type.

        Maps task types to permitted tools using the existing
        tool_capability_registry, then finds matching capabilities.
        """
        permitted_tools = get_permitted_tools_for_task_type(task_type)

        return [
            cap_class()
            for cap_class in self._capabilities.values()
            if cap_class().get_requirements().required_tools.issubset(permitted_tools)
        ]

    def validate_capability_for_execution(
        self, capability_name: str, task_type: str, authority_level: str
    ) -> tuple[bool, str]:
        """Validate that a capability can execute in a specific context.

        Checks both the capability registry and the tool registry
        to ensure authorization consistency.
        """
        cap = self.get_capability(capability_name)
        if cap is None:
            return False, f"Capability '{capability_name}' not found"

        reqs = cap.get_requirements()

        # Check authority level
        level_order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}
        if level_order.get(authority_level, 0) < level_order.get(reqs.min_authority_level, 0):
            return False, (
                f"Authority level {authority_level} insufficient "
                f"(requires {reqs.min_authority_level})"
            )

        # Check tools are permitted for task type
        permitted_tools = get_permitted_tools_for_task_type(task_type)
        if not reqs.required_tools.issubset(permitted_tools):
            missing = reqs.required_tools - permitted_tools
            return False, (
                f"Tools {missing} not permitted for task type '{task_type}'"
            )

        return True, ""


# ---------------------------------------------------------------------------
# Global registry instance
# ---------------------------------------------------------------------------

_jarvis_registry = JARVISCapabilityRegistry()


def get_jarvis_registry() -> JARVISCapabilityRegistry:
    """Get the JARVIS capability registry."""
    return _jarvis_registry


def reset_jarvis_registry() -> None:
    """Reset the JARVIS registry (for testing)."""
    global _jarvis_registry
    _jarvis_registry = JARVISCapabilityRegistry()
