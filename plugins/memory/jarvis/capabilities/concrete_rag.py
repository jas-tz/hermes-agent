"""Concrete JARVIS RAG capabilities — Phase 37.

Each class wraps an existing JARVIS tool call through the AUTHORIZED
_dispatch_jarvis_tool_call path (the same dispatch function used by
JARVISSubagentProvider.handle_tool_call). No new privileges are introduced.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict

from plugins.memory.jarvis.capabilities.base import (
    Capability,
    CapabilityCategory,
    CapabilityContext,
    CapabilityInputType,
    CapabilityOutputType,
    CapabilityRequirements,
    CapabilityResult,
)

logger = logging.getLogger(__name__)


def _dispatch(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke the authorized JARVIS sub-agent dispatch path.

    This is the SAME function used by JARVISSubagentProvider — concrete
    capabilities never bypass the authorization boundary.
    """
    from plugins.memory.jarvis import _dispatch_jarvis_tool_call
    raw = _dispatch_jarvis_tool_call(tool_name, args)
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return {"ok": False, "error": f"Malformed dispatch response: {raw!r}"}


class RagSearchCapability(Capability):
    """Wraps hermes_rag_search — semantic memory lookup."""

    def get_name(self) -> str:
        return "rag_search"

    def get_description(self) -> str:
        return "Search JARVIS semantic memory via RAG"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_RETRIEVAL

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L0",
            required_tools=frozenset({"hermes_rag_search"}),
            input_type=CapabilityInputType.QUERY,
            output_type=CapabilityOutputType.STRUCTURED,
            requires_approval=False,
            parallelizable=True,
            max_execution_time_seconds=30,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        query = context.parameters.get("query", "")
        if not query or not query.strip():
            return False, "query parameter is required"
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        args = {
            "query": context.parameters.get("query", ""),
            "limit": int(context.parameters.get("limit", 5)),
        }
        result = _dispatch("hermes_rag_search", args)
        if result.get("ok"):
            return CapabilityResult.success_result(
                data=result,
                execution_id=context.execution_id,
                capability_name=self.get_name(),
            )
        return CapabilityResult.failure_result(
            error=result.get("error", "Unknown RAG search error"),
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )


class RagStoreCapability(Capability):
    """Wraps hermes_rag_store — store memory with provenance."""

    def get_name(self) -> str:
        return "rag_store"

    def get_description(self) -> str:
        return "Store memory in JARVIS RAG"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_STORAGE

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L1",
            required_tools=frozenset({"hermes_rag_store"}),
            input_type=CapabilityInputType.TEXT,
            output_type=CapabilityOutputType.STRUCTURED,
            requires_approval=False,
            parallelizable=True,
            max_execution_time_seconds=60,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        text = context.parameters.get("text", "")
        if not text or not text.strip():
            return False, "text parameter is required"
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        args = {
            "text": context.parameters.get("text", ""),
            "metadata": context.parameters.get("metadata"),
            "provenance": context.parameters.get("provenance"),
        }
        result = _dispatch("hermes_rag_store", args)
        if result.get("ok"):
            return CapabilityResult.success_result(
                data=result,
                execution_id=context.execution_id,
                capability_name=self.get_name(),
            )
        return CapabilityResult.failure_result(
            error=result.get("error", "Unknown RAG store error"),
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )


class RagDeleteCapability(Capability):
    """Wraps hermes_rag_delete — delete a memory by ID (L3 destructive)."""

    def get_name(self) -> str:
        return "rag_delete"

    def get_description(self) -> str:
        return "Delete a RAG memory by ID"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_STORAGE

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L3",
            required_tools=frozenset({"hermes_rag_delete"}),
            input_type=CapabilityInputType.TEXT,
            output_type=CapabilityOutputType.STRUCTURED,
            requires_approval=True,
            parallelizable=False,
            max_execution_time_seconds=30,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        mid = context.parameters.get("memory_id", "")
        if not mid or not mid.strip():
            return False, "memory_id parameter is required"
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        args = {"memory_id": context.parameters.get("memory_id", "")}
        result = _dispatch("hermes_rag_delete", args)
        if result.get("ok"):
            return CapabilityResult.success_result(
                data=result,
                execution_id=context.execution_id,
                capability_name=self.get_name(),
            )
        return CapabilityResult.failure_result(
            error=result.get("error", "Unknown RAG delete error"),
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )


class RagRetrieveLearningsCapability(Capability):
    """Wraps hermes_rag_retrieve_learnings — cross-session learning lookup."""

    def get_name(self) -> str:
        return "rag_retrieve_learnings"

    def get_description(self) -> str:
        return "Retrieve cross-session learnings"

    def get_category(self) -> CapabilityCategory:
        return CapabilityCategory.DATA_RETRIEVAL

    def get_requirements(self) -> CapabilityRequirements:
        return CapabilityRequirements(
            min_authority_level="L0",
            required_tools=frozenset({"hermes_rag_retrieve_learnings"}),
            input_type=CapabilityInputType.QUERY,
            output_type=CapabilityOutputType.STRUCTURED,
            requires_approval=False,
            parallelizable=True,
            max_execution_time_seconds=30,
        )

    def validate(self, context: CapabilityContext) -> tuple[bool, str]:
        query = context.parameters.get("query", "")
        if not query or not query.strip():
            return False, "query parameter is required"
        return True, ""

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        args = {
            "query": context.parameters.get("query", ""),
            "limit": int(context.parameters.get("limit", 5)),
        }
        result = _dispatch("hermes_rag_retrieve_learnings", args)
        if result.get("ok"):
            return CapabilityResult.success_result(
                data=result,
                execution_id=context.execution_id,
                capability_name=self.get_name(),
            )
        return CapabilityResult.failure_result(
            error=result.get("error", "Unknown retrieve error"),
            execution_id=context.execution_id,
            capability_name=self.get_name(),
        )
