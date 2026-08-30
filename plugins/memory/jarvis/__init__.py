"""Hermes RAG provider — thin adapter over the Jarvis RAG engine.

Provides persistent semantic memory via Qdrant local mode, embedding
with intfloat/multilingual-e5-small. Delegates all operations to
rag_memory.py without duplicating any logic.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


class HermesRAGProvider(MemoryProvider):
    """MemoryProvider adapter that delegates to the Jarvis RAG engine."""

    def __init__(self) -> None:
        self._hermes_home: Optional[str] = None
        self._session_id: Optional[str] = None
        self._original_cwd: Optional[str] = None

    @property
    def name(self) -> str:
        return "jarvis"

    def is_available(self) -> bool:
        """Return True if the RAG engine module exists and can be found.

        This checks for the existence of the rag_memory module file rather than
        requiring all dependencies (torch, sentence-transformers) to be importable
        in the current venv. The actual runtime uses the rag venv which has all
        dependencies installed.
        """
        try:
            rag_path = os.path.expanduser("~/.hermes/rag")
            rag_memory_path = os.path.join(rag_path, "rag_memory.py")
            return os.path.isfile(rag_memory_path)
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = str(kwargs.get("hermes_home", os.path.expanduser("~/.hermes")))

        # rag_memory uses a relative path for Qdrant; we must run from ~/.hermes/rag
        # so that the relative path resolves correctly. Store the original cwd so we
        # don't interfere with the caller's working directory expectations.
        rag_dir = Path(self._hermes_home) / "rag"
        if rag_dir.is_dir():
            self._original_cwd = os.getcwd()
            os.chdir(str(rag_dir))
            logger.debug("HermesRAGProvider: changed cwd to %s", rag_dir)

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        pass

    def system_prompt_block(self) -> str:
        return (
            "# Hermes RAG\n"
            "Semantic memory powered by Qdrant. Use hermes_rag_search to recall"
            " project context and hermes_rag_store to save durable facts.\n"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "hermes_rag_search",
                "description": "Search the Jarvis RAG semantic memory for relevant context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                        "limit": {"type": "integer", "description": "Max results (default 5)."},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "hermes_rag_store",
                "description": "Store a memory in the Jarvis RAG semantic store.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The memory text to store."},
                        "metadata": {"type": "object", "description": "Optional metadata dict."},
                        "provenance": {"type": "object", "description": "Optional provenance metadata (source, context_id, confidence)."},
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "hermes_rag_delete",
                "description": "Delete a memory by its UUID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "The UUID of the memory to delete."},
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "hermes_rag_extract_learnings",
                "description": "Extract cross-session learnings from a completed conversation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "The session identifier."},
                        "messages": {"type": "array", "description": "List of message dicts with 'role' and 'content'."},
                    },
                    "required": ["session_id"],
                },
            },
            {
                "name": "hermes_rag_store_learnings",
                "description": "Extract and store cross-session learnings from a completed session.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "The session identifier."},
                        "messages": {"type": "array", "description": "List of message dicts with 'role' and 'content'."},
                    },
                    "required": ["session_id"],
                },
            },
            {
                "name": "hermes_rag_retrieve_learnings",
                "description": "Retrieve cross-session learnings relevant to a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                        "limit": {"type": "integer", "description": "Max results (default 5)."},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "hermes_rag_rollback_learnings",
                "description": "Delete all cross-session learnings (rollback).",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            from rag_memory import (
                add_memory, search_memory, delete_memory, store_memory_with_provenance,
                extract_session_learnings, store_session_learnings,
                retrieve_cross_session_learnings, rollback_cross_session_learnings,
            )

            if tool_name == "hermes_rag_search":
                query = args.get("query", "")
                limit = int(args.get("limit", 5))
                results = search_memory(query, limit=limit)
                return json.dumps({"ok": True, "results": results})

            if tool_name == "hermes_rag_store":
                text = args.get("text", "")
                metadata = args.get("metadata")
                provenance = args.get("provenance")
                if not text:
                    return json.dumps({"ok": False, "error": "text is required"})
                if provenance:
                    memory_id = store_memory_with_provenance(text, metadata=metadata, provenance=provenance)
                else:
                    memory_id = add_memory(text, metadata=metadata)
                return json.dumps({"ok": True, "memory_id": memory_id})

            if tool_name == "hermes_rag_delete":
                memory_id = args.get("memory_id", "")
                if not memory_id:
                    return json.dumps({"ok": False, "error": "memory_id is required"})
                result = delete_memory(memory_id)
                return json.dumps(result)

            # Cross-session learning tools (Step 150)
            if tool_name == "hermes_rag_extract_learnings":
                session_id = args.get("session_id", "")
                messages = args.get("messages", [])
                if not session_id:
                    return json.dumps({"ok": False, "error": "session_id is required"})
                learnings = extract_session_learnings(messages, session_id)
                return json.dumps({"ok": True, "learnings": learnings, "count": len(learnings)})

            if tool_name == "hermes_rag_store_learnings":
                session_id = args.get("session_id", "")
                messages = args.get("messages", [])
                if not session_id:
                    return json.dumps({"ok": False, "error": "session_id is required"})
                result = store_session_learnings(session_id, messages)
                return json.dumps(result)

            if tool_name == "hermes_rag_retrieve_learnings":
                query = args.get("query", "")
                limit = int(args.get("limit", 5))
                if not query:
                    return json.dumps({"ok": False, "error": "query is required"})
                results = retrieve_cross_session_learnings(query, limit=limit)
                return json.dumps({"ok": True, "results": results, "count": len(results)})

            if tool_name == "hermes_rag_rollback_learnings":
                result = rollback_cross_session_learnings()
                return json.dumps(result)

            return json.dumps({"ok": False, "error": f"unknown tool: {tool_name}"})
        except Exception as exc:
            logger.warning("HermesRAGProvider tool error: %s", exc)
            return json.dumps({"ok": False, "error": str(exc)})

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        pass

    def shutdown(self) -> None:
        if self._original_cwd is not None:
            try:
                os.chdir(self._original_cwd)
            except Exception:
                pass
            self._original_cwd = None


def register(ctx) -> None:
    provider = HermesRAGProvider()
    ctx.register_memory_provider(provider)
