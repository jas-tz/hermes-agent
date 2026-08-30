"""Tests for the HermesRAGProvider memory plugin."""
import json
import sys
import os
from unittest.mock import patch, MagicMock

import pytest

# Ensure the hermes-agent root is on sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# Ensure the rag module is on sys.path (for rag_memory)
rag_path = os.path.expanduser("~/.hermes/rag")
sys.path.insert(0, rag_path)


class TestHermesRAGProviderInterface:
    """Verify the provider implements the MemoryProvider ABC."""

    @pytest.fixture
    def provider(self):
        from plugins.memory.jarvis import HermesRAGProvider
        return HermesRAGProvider()

    @pytest.fixture
    def abc(self):
        from agent.memory_provider import MemoryProvider
        return MemoryProvider

    def test_inherits_from_memory_provider(self, provider, abc):
        assert isinstance(provider, abc)

    def test_name_is_jarvis(self, provider):
        assert provider.name == "jarvis"

    def test_implements_is_available(self, provider):
        assert callable(provider.is_available)

    def test_implements_initialize(self, provider):
        assert callable(provider.initialize)

    def test_implements_get_tool_schemas(self, provider):
        assert callable(provider.get_tool_schemas)

    def test_implements_handle_tool_call(self, provider):
        assert callable(provider.handle_tool_call)

    def test_implements_prefetch(self, provider):
        assert callable(provider.prefetch)

    def test_implements_sync_turn(self, provider):
        assert callable(provider.sync_turn)

    def test_implements_shutdown(self, provider):
        assert callable(provider.shutdown)

    def test_implements_get_config_schema(self, provider):
        assert callable(provider.get_config_schema)

    def test_implements_save_config(self, provider):
        assert callable(provider.save_config)


class TestHermesRAGProviderTools:
    """Verify the provider exposes the expected tool schemas."""

    @pytest.fixture
    def provider(self):
        from plugins.memory.jarvis import HermesRAGProvider
        return HermesRAGProvider()

    def test_has_rag_search_tool(self, provider):
        tools = provider.get_tool_schemas()
        names = [t["name"] for t in tools]
        assert "hermes_rag_search" in names

    def test_has_rag_store_tool(self, provider):
        tools = provider.get_tool_schemas()
        names = [t["name"] for t in tools]
        assert "hermes_rag_store" in names

    def test_has_rag_delete_tool(self, provider):
        tools = provider.get_tool_schemas()
        names = [t["name"] for t in tools]
        assert "hermes_rag_delete" in names

    def test_tool_schemas_have_required_fields(self, provider):
        tools = provider.get_tool_schemas()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool


class TestHermesRAGProviderDelegation:
    """Verify tool calls delegate correctly to rag_memory.py."""

    @pytest.fixture
    def provider(self):
        from plugins.memory.jarvis import HermesRAGProvider
        return HermesRAGProvider()

    def test_store_delegates_to_add_memory(self, provider):
        with patch("rag_memory.add_memory") as mock_add:
            mock_add.return_value = "test-uuid-123"
            result = json.loads(provider.handle_tool_call("hermes_rag_store", {"text": "test memory"}))
            assert result["ok"] is True
            assert result["memory_id"] == "test-uuid-123"
            mock_add.assert_called_once_with("test memory", metadata=None)

    def test_store_with_metadata_delegates(self, provider):
        with patch("rag_memory.add_memory") as mock_add:
            mock_add.return_value = "test-uuid-456"
            result = json.loads(provider.handle_tool_call(
                "hermes_rag_store", {"text": "test", "metadata": {"source": "test"}}
            ))
            assert result["ok"] is True
            mock_add.assert_called_once_with("test", metadata={"source": "test"})

    def test_store_empty_text_returns_error(self, provider):
        result = json.loads(provider.handle_tool_call("hermes_rag_store", {"text": ""}))
        assert result["ok"] is False
        assert "text is required" in result["error"]

    def test_search_delegates_to_search_memory(self, provider):
        with patch("rag_memory.search_memory") as mock_search:
            mock_search.return_value = [{"score": 0.9, "text": "result", "metadata": {}}]
            result = json.loads(provider.handle_tool_call("hermes_rag_search", {"query": "test query"}))
            assert result["ok"] is True
            assert "results" in result
            mock_search.assert_called_once_with("test query", limit=5)

    def test_search_with_limit_delegates(self, provider):
        with patch("rag_memory.search_memory") as mock_search:
            mock_search.return_value = []
            result = json.loads(provider.handle_tool_call("hermes_rag_search", {"query": "test", "limit": 3}))
            assert result["ok"] is True
            mock_search.assert_called_once_with("test", limit=3)

    def test_delete_delegates_to_delete_memory(self, provider):
        with patch("rag_memory.delete_memory") as mock_delete:
            mock_delete.return_value = {"ok": True, "status": "completed"}
            result = json.loads(provider.handle_tool_call("hermes_rag_delete", {"memory_id": "uuid-123"}))
            assert result["ok"] is True
            mock_delete.assert_called_once_with("uuid-123")

    def test_delete_missing_id_returns_error(self, provider):
        result = json.loads(provider.handle_tool_call("hermes_rag_delete", {}))
        assert result["ok"] is False
        assert "memory_id is required" in result["error"]


class TestHermesRAGProviderErrors:
    """Verify errors are propagated clearly."""

    @pytest.fixture
    def provider(self):
        from plugins.memory.jarvis import HermesRAGProvider
        return HermesRAGProvider()

    def test_unknown_tool_returns_error(self, provider):
        result = json.loads(provider.handle_tool_call("unknown_tool", {}))
        assert result["ok"] is False
        assert "Unknown tool" in result["error"]

    def test_exception_in_delegation_returns_error_json(self, provider):
        with patch("rag_memory.search_memory", side_effect=RuntimeError("Qdrant down")):
            result = json.loads(provider.handle_tool_call("hermes_rag_search", {"query": "test"}))
            assert result["ok"] is False
            assert "Qdrant down" in result["error"]

    def test_exception_in_store_returns_error_json(self, provider):
        with patch("rag_memory.add_memory", side_effect=Exception("full disk")):
            result = json.loads(provider.handle_tool_call("hermes_rag_store", {"text": "data"}))
            assert result["ok"] is False
            assert "full disk" in result["error"]


class TestHermesRAGProviderRegistration:
    """Verify the provider can register through the plugin discovery mechanism."""

    def test_register_function_exists(self):
        from plugins.memory.jarvis import register
        assert callable(register)

    def test_register_adds_provider(self):
        from plugins.memory.jarvis import register
        mock_ctx = MagicMock()
        register(mock_ctx)
        mock_ctx.register_memory_provider.assert_called_once()
        provider = mock_ctx.register_memory_provider.call_args[0][0]
        assert provider.name == "jarvis"


class TestHermesRAGProviderDiscovery:
    """Verify the provider is discoverable via the plugin system."""

    def test_discoverable_via_discover_memory_providers(self):
        from plugins.memory import discover_memory_providers
        providers = discover_memory_providers()
        names = [name for name, _, _ in providers]
        assert "jarvis" in names

    def test_loadable_via_load_memory_provider(self):
        from plugins.memory import load_memory_provider
        provider = load_memory_provider("jarvis")
        assert provider is not None
        assert provider.name == "jarvis"

    def test_is_available_returns_bool(self):
        from plugins.memory.jarvis import HermesRAGProvider
        provider = HermesRAGProvider()
        assert isinstance(provider.is_available(), bool)
