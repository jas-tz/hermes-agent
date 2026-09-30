# JARVIS Capability Registry — Phase 36B/37 public surface.

from plugins.memory.jarvis.capabilities.base import (
    Capability,
    CapabilityCategory,
    CapabilityContext,
    CapabilityInputType,
    CapabilityOutputType,
    CapabilityRegistry,
    CapabilityRequirements,
    CapabilityResult,
    get_registry,
    reset_registry,
)
from plugins.memory.jarvis.capabilities.registry import (
    JARVISCapabilityRegistry,
    CapabilityRegistrationError,
    get_jarvis_registry,
    reset_jarvis_registry,
)
from plugins.memory.jarvis.capabilities.concrete_rag import (
    RagSearchCapability,
    RagStoreCapability,
    RagDeleteCapability,
    RagRetrieveLearningsCapability,
)
from plugins.memory.jarvis.capabilities.concrete_continuity import (
    GoalListCapability,
    GoalCreateCapability,
    TaskListCapability,
    TaskCreateCapability,
    DecisionListCapability,
    DecisionRecordCapability,
    GoalCompleteCapability,
    TaskCompleteCapability,
)

__all__ = [
    # base
    "Capability",
    "CapabilityCategory",
    "CapabilityContext",
    "CapabilityInputType",
    "CapabilityOutputType",
    "CapabilityRegistry",
    "CapabilityRequirements",
    "CapabilityResult",
    # registry
    "JARVISCapabilityRegistry",
    "CapabilityRegistrationError",
    "get_registry",
    "reset_registry",
    "get_jarvis_registry",
    "reset_jarvis_registry",
    # concrete RAG
    "RagSearchCapability",
    "RagStoreCapability",
    "RagDeleteCapability",
    "RagRetrieveLearningsCapability",
    # concrete continuity
    "GoalListCapability",
    "GoalCreateCapability",
    "TaskListCapability",
    "TaskCreateCapability",
    "DecisionListCapability",
    "DecisionRecordCapability",
    "GoalCompleteCapability",
    "TaskCompleteCapability",
]


def register_default_capabilities(registry: JARVISCapabilityRegistry) -> None:
    """Register the full Phase 37 default capability set."""
    for cap_cls in [
        RagSearchCapability,
        RagStoreCapability,
        RagDeleteCapability,
        RagRetrieveLearningsCapability,
        GoalListCapability,
        GoalCreateCapability,
        TaskListCapability,
        TaskCreateCapability,
        DecisionListCapability,
        DecisionRecordCapability,
        GoalCompleteCapability,
        TaskCompleteCapability,
    ]:
        registry.register(cap_cls)
