"""Tests for ``hermes doctor --provider`` memory provider health check."""
import sys
import os
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.expanduser("~/.hermes/rag"))

from hermes_cli import doctor as doctor_mod


class TestFindRagSitePackages:

    def test_returns_none_when_rag_venv_missing(self):
        with patch.object(doctor_mod.Path, "is_dir", return_value=False):
            assert doctor_mod._find_rag_site_packages() is None

    def test_returns_none_when_no_site_packages_dir(self):
        with patch.object(doctor_mod.Path, "is_dir", return_value=True):
            with patch.object(doctor_mod.Path, "iterdir", return_value=[]):
                result = doctor_mod._find_rag_site_packages()
                assert result is None


class TestRunProviderHealthCheck:

    def test_no_provider_configured_reports_builtin(self):
        with patch("hermes_cli.config.load_config", return_value={"memory": {}}), \
             patch.object(doctor_mod, "check_ok"), \
             patch.object(doctor_mod, "check_fail") as mock_fail, \
             patch.object(doctor_mod, "check_info") as mock_info, \
             patch.object(doctor_mod, "_section"):
            doctor_mod._run_provider_health_check([])
        assert mock_fail.call_count == 0
        assert any("No external" in str(c) for c in mock_info.call_args_list)

    def test_provider_not_discovered_fails(self):
        with patch("hermes_cli.config.load_config", return_value={"memory": {"provider": "jarvis"}}), \
             patch("plugins.memory.discover_memory_providers", return_value=[("holographic", "", True)]), \
             patch.object(doctor_mod, "check_ok"), \
             patch.object(doctor_mod, "check_fail") as mock_fail, \
             patch.object(doctor_mod, "check_info"), \
             patch.object(doctor_mod, "_section"):
            doctor_mod._run_provider_health_check([])
        assert mock_fail.call_count >= 1

    def test_provider_not_available_fails(self):
        mock_provider = MagicMock()
        mock_provider.name = "jarvis"
        mock_provider.is_available.return_value = False
        with patch("hermes_cli.config.load_config", return_value={"memory": {"provider": "jarvis"}}), \
             patch("plugins.memory.discover_memory_providers", return_value=[("jarvis", "", True)]), \
             patch("plugins.memory.load_memory_provider", return_value=mock_provider), \
             patch.object(doctor_mod, "check_ok"), \
             patch.object(doctor_mod, "check_fail") as mock_fail, \
             patch.object(doctor_mod, "check_info"), \
             patch.object(doctor_mod, "_section"):
            doctor_mod._run_provider_health_check([])
        assert mock_fail.call_count >= 1

    def test_config_read_error_reports_and_returns(self):
        with patch("hermes_cli.config.load_config", side_effect=Exception("corrupt")), \
             patch.object(doctor_mod, "check_ok"), \
             patch.object(doctor_mod, "check_fail") as mock_fail, \
             patch.object(doctor_mod, "check_info"), \
             patch.object(doctor_mod, "_section"):
            issues = []
            doctor_mod._run_provider_health_check(issues)
        assert mock_fail.call_count >= 1
        assert len(issues) >= 1

    def test_provider_load_returns_none_fails(self):
        with patch("hermes_cli.config.load_config", return_value={"memory": {"provider": "unknown"}}), \
             patch("plugins.memory.discover_memory_providers", return_value=[("jarvis", "", True)]), \
             patch("plugins.memory.load_memory_provider", return_value=None), \
             patch.object(doctor_mod, "check_ok"), \
             patch.object(doctor_mod, "check_fail") as mock_fail, \
             patch.object(doctor_mod, "check_info"), \
             patch.object(doctor_mod, "_section"):
            doctor_mod._run_provider_health_check([])
        assert mock_fail.call_count >= 1

    def test_load_error_reports_and_returns(self):
        with patch("hermes_cli.config.load_config", return_value={"memory": {"provider": "jarvis"}}), \
             patch("plugins.memory.discover_memory_providers", return_value=[("jarvis", "", True)]), \
             patch("plugins.memory.load_memory_provider", side_effect=Exception("import error")), \
             patch.object(doctor_mod, "check_ok"), \
             patch.object(doctor_mod, "check_fail") as mock_fail, \
             patch.object(doctor_mod, "check_info"), \
             patch.object(doctor_mod, "_section"):
            issues = []
            doctor_mod._run_provider_health_check(issues)
        assert mock_fail.call_count >= 1


class TestDoctorProviderArg:

    def test_provider_arg_triggers_health_check(self):
        mock_provider = MagicMock()
        mock_provider.name = "jarvis"
        mock_provider.is_available.return_value = True
        mock_provider.handle_tool_call.side_effect = [
            '{"ok": true, "memory_id": "test-id"}',
            '{"ok": true, "results": [{"text": "doctor-health-check-marker"}]}',
            '{"ok": true, "status": "completed"}',
        ]
        with patch("hermes_cli.config.load_config", return_value={"memory": {"provider": "jarvis"}}), \
             patch("plugins.memory.discover_memory_providers", return_value=[("jarvis", "", True)]), \
             patch("plugins.memory.load_memory_provider", return_value=mock_provider), \
             patch.object(doctor_mod, "get_hermes_home", return_value="/tmp/fake-hermes"), \
             patch.object(doctor_mod, "check_ok"), \
             patch.object(doctor_mod, "check_fail"), \
             patch.object(doctor_mod, "check_info"), \
             patch.object(doctor_mod, "_section"), \
             patch("hermes_cli.doctor_live.maybe_run_live_checks", return_value=None):
            args = MagicMock()
            args.fix = False
            args.ack = None
            args.provider = True
            doctor_mod.run_doctor(args)
        mock_provider.initialize.assert_called_once()
        mock_provider.shutdown.assert_called_once()
