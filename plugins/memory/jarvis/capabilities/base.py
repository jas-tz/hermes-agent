"""JARVIS Capability Base — Phase 36A."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Type

logger = logging.getLogger(__name__)


class CapabilityCategory(Enum):
    DATA_RETRIEVAL = "data_retrieval"
    DATA_STORAGE = "data_storage"
    DATA_TRANSFORMATION = "data_transformation"
    EXTERNAL_INTEGRATION = "external_integration"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    ORCHESTRATION = "orchestration"


class CapabilityInputType(Enum):
    TEXT = "text"
    STRUCTURED = "structured"
    FILE = "file"
    QUERY = "query"
    NONE = "none"


class CapabilityOutputType(Enum):
    TEXT = "text"
    STRUCTURED = "structured"
    FILE = "file"
    BOOLEAN = "boolean"
    NONE = "none"


@dataclass(frozen=True)
class CapabilityRequirements:
    min_authority_level: str = "L0"
    required_tools: FrozenSet[str] = field(default_factory=frozenset)
    input_type: CapabilityInputType = CapabilityInputType.TEXT
    output_type: CapabilityOutputType = CapabilityOutputType.TEXT
    requires_approval: bool = False
    parallelizable: bool = True
    max_execution_time_seconds: int = 300
    dependencies: FrozenSet[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class CapabilityContext:
    execution_id: str
    subagent_id: str
    parent_session_id: str
    task_goal: str
    task_type: str
    authority_level: str
    delegation_depth: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependency_results: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class CapabilityResult:
    success: bool
    data: Any = None
    message: str = ""
    error: Optional[str] = None
    execution_id: str = ""
    capability_name: str = ""
    execution_time_seconds: float = 0.0
    composable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "error": self.error,
            "execution_id": self.execution_id,
            "capability_name": self.capability_name,
            "execution_time_seconds": self.execution_time_seconds,
            "composable": self.composable,
        }

    @classmethod
    def success_result(cls, data=None, message="", execution_id="", capability_name="", execution_time_seconds=0.0):
        return cls(success=True, data=data, message=message, execution_id=execution_id, capability_name=capability_name, execution_time_seconds=execution_time_seconds)

    @classmethod
    def failure_result(cls, error, message="", execution_id="", capability_name="", execution_time_seconds=0.0):
        return cls(success=False, error=error, message=message, execution_id=execution_id, capability_name=capability_name, execution_time_seconds=execution_time_seconds)


class Capability(ABC):

    @abstractmethod
    def get_name(self) -> str:
        ...

    @abstractmethod
    def get_description(self) -> str:
        ...

    @abstractmethod
    def get_category(self) -> CapabilityCategory:
        ...

    @abstractmethod
    def get_requirements(self) -> CapabilityRequirements:
        ...

    @abstractmethod
    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        ...

    @abstractmethod
    def execute(self, context: CapabilityContext) -> CapabilityResult:
        ...

    def can_execute(self, authority_level: str, available_tools: FrozenSet[str]) -> bool:
        level_order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}
        reqs = self.get_requirements()
        if level_order.get(authority_level, 0) < level_order.get(reqs.min_authority_level, 0):
            return False
        if not reqs.required_tools.issubset(available_tools):
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        reqs = self.get_requirements()
        return {
            "name": self.get_name(),
            "description": self.get_description(),
            "category": self.get_category().value,
            "requirements": {
                "min_authority_level": reqs.min_authority_level,
                "required_tools": sorted(reqs.required_tools),
                "input_type": reqs.input_type.value,
                "output_type": reqs.output_type.value,
                "requires_approval": reqs.requires_approval,
                "parallelizable": reqs.parallelizable,
                "max_execution_time_seconds": reqs.max_execution_time_seconds,
                "dependencies": sorted(reqs.dependencies),
            },
        }


class CapabilityRegistry:
    def __init__(self) -> None:
        self._capabilities: Dict[str, Type[Capability]] = {}

    def register(self, capability_class: Type[Capability]) -> None:
        temp = capability_class()
        name = temp.get_name()
        if name in self._capabilities:
            raise ValueError(f"Capability '{name}' already registered")
        self._capabilities[name] = capability_class
        logger.debug("Registered capability: %s", name)

    def unregister(self, name: str) -> None:
        self._capabilities.pop(name, None)

    def get(self, name: str) -> Optional[Type[Capability]]:
        return self._capabilities.get(name)

    def list_capabilities(self) -> List[str]:
        return sorted(self._capabilities.keys())

    def get_capability(self, name: str) -> Optional[Capability]:
        cap_class = self._capabilities.get(name)
        if cap_class is None:
            return None
        return cap_class()

    def find_by_category(self, category: CapabilityCategory) -> List[Capability]:
        return [cap_class() for cap_class in self._capabilities.values() if cap_class().get_category() == category]

    def find_executable(self, authority_level: str, available_tools: FrozenSet[str]) -> List[Capability]:
        return [cap_class() for cap_class in self._capabilities.values() if cap_class().can_execute(authority_level, available_tools)]

    def to_dict(self) -> Dict[str, Any]:
        return {"capabilities": [cap_class().to_dict() for cap_class in self._capabilities.values()]}


_registry = CapabilityRegistry()

def get_registry() -> CapabilityRegistry:
    return _registry

def reset_registry() -> None:
    global _registry
    _registry = CapabilityRegistry()
