"""JARVIS_QDRANT_URL through Hermes's own environment mechanism.

Proves the pre-cutover state is inert (a commented line in ~/.hermes/.env
does not switch Jarvis to server mode), that uncommenting it does, and that
once the cutover marker exists a missing URL fails instead of falling back
to the production local store.

Uses throwaway hermes_home directories and a simulated production store;
no Qdrant client is ever constructed here.
"""
import os
import sys
from pathlib import Path

import pytest

# Repository root (not tests/, which has its own hermes_cli test package).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, os.path.expanduser("~/.hermes/rag"))

from hermes_cli.config_defaults import OPTIONAL_ENV_VARS  # noqa: E402
from hermes_cli.env_loader import load_hermes_dotenv  # noqa: E402

QDRANT_ENV_VARS = ("JARVIS_QDRANT_URL", "JARVIS_QDRANT_PATH", "JARVIS_QDRANT_TIMEOUT")

PRE_CUTOVER_BLOCK = (
    "# Jarvis memory: Qdrant Server (localhost only). INACTIVE until production\n"
    "# memory has been migrated to the server.\n"
    "# JARVIS_QDRANT_URL=http://127.0.0.1:6333\n"
)
CUTOVER_LINE = "JARVIS_QDRANT_URL=http://127.0.0.1:6333\n"


@pytest.fixture
def rm(monkeypatch, tmp_path):
    """rag_memory with a simulated production store and marker location."""
    import rag_memory

    # load_hermes_dotenv() writes straight into os.environ; monkeypatch only
    # undoes keys that existed beforehand, so these are removed explicitly.
    loaded_keys = QDRANT_ENV_VARS + ("SOME_OTHER_SETTING",)
    saved = {k: os.environ.pop(k) for k in loaded_keys if k in os.environ}
    prod = tmp_path / "rag" / "qdrant_data"
    monkeypatch.setattr(rag_memory, "PRODUCTION_QDRANT_PATH", prod)
    monkeypatch.setattr(rag_memory, "QDRANT_PATH", prod)
    monkeypatch.setattr(rag_memory, "SERVER_REQUIRED_MARKER", tmp_path / "rag" / "qdrant_server_required")
    yield rag_memory
    for k in loaded_keys:
        os.environ.pop(k, None)
    os.environ.update(saved)


def _hermes_home(tmp_path, body):
    home = tmp_path / "hermes_home"
    home.mkdir()
    (home / ".env").write_text("SOME_OTHER_SETTING=1\n" + body, encoding="utf-8")
    return home


def test_jarvis_qdrant_url_is_registered_as_a_non_secret_setting():
    entry = OPTIONAL_ENV_VARS["JARVIS_QDRANT_URL"]

    assert entry["category"] == "setting"
    assert entry["password"] is False
    assert "loopback" in entry["description"]


def test_commented_pre_cutover_line_leaves_local_mode(rm, tmp_path):
    home = _hermes_home(tmp_path, PRE_CUTOVER_BLOCK)

    load_hermes_dotenv(hermes_home=home)

    assert "JARVIS_QDRANT_URL" not in os.environ
    assert rm.resolve_qdrant_config() == {"mode": "local", "path": str(rm.QDRANT_PATH)}


def test_uncommented_line_switches_to_server_mode(rm, tmp_path):
    home = _hermes_home(tmp_path, CUTOVER_LINE)

    load_hermes_dotenv(hermes_home=home)

    assert os.environ["JARVIS_QDRANT_URL"] == "http://127.0.0.1:6333"
    assert rm.resolve_qdrant_config() == {
        "mode": "server", "url": "http://127.0.0.1:6333", "timeout": 5,
    }


def test_marker_with_line_still_commented_fails_clearly(rm, tmp_path):
    rm.SERVER_REQUIRED_MARKER.parent.mkdir(parents=True)
    rm.SERVER_REQUIRED_MARKER.touch()
    home = _hermes_home(tmp_path, PRE_CUTOVER_BLOCK)

    load_hermes_dotenv(hermes_home=home)

    with pytest.raises(rm.QdrantConfigError, match="JARVIS_QDRANT_URL is not set"):
        rm.resolve_qdrant_config()
    assert not rm.PRODUCTION_QDRANT_PATH.exists()


def test_marker_with_non_loopback_url_in_env_file_is_refused(rm, tmp_path):
    rm.SERVER_REQUIRED_MARKER.parent.mkdir(parents=True)
    rm.SERVER_REQUIRED_MARKER.touch()
    home = _hermes_home(tmp_path, "JARVIS_QDRANT_URL=http://192.168.1.20:6333\n")

    load_hermes_dotenv(hermes_home=home)

    with pytest.raises(rm.QdrantConfigError, match="loopback"):
        rm.resolve_qdrant_config()
    assert not rm.PRODUCTION_QDRANT_PATH.exists()
