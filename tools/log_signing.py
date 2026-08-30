"""HMAC-based log signing for tamper-evident audit records.

Provides integrity protection for audit logs using a hash chain.
Each log entry is signed with HMAC-SHA256, and the signature chains
to the previous entry so any modification breaks the chain.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import pathlib
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default signing key path (relative to hermes config dir)
_DEFAULT_KEY_PATH = pathlib.Path.home() / ".hermes" / ".log_signing_key"


class LogSigner:
    """HMAC-based log signing for tamper detection.

    Uses a hash chain: each entry's signature depends on the previous
    signature, so any insertion, deletion, or modification breaks the chain.
    """

    def __init__(self, key: bytes, chain_id: str = "default") -> None:
        """Initialize the log signer.

        Args:
            key: HMAC signing key (should be 32+ bytes).
            chain_id: Identifier for this log chain.
        """
        if len(key) < 16:
            raise ValueError("Signing key must be at least 16 bytes")
        self._key = key
        self._chain_id = chain_id
        self._chain_hash = hashlib.sha256(
            f"hermes-log-chain:{chain_id}".encode()
        ).digest()

    @classmethod
    def from_config(cls, chain_id: str = "default") -> Optional["LogSigner"]:
        """Create a LogSigner from the configured key.

        Returns None if signing is not configured.
        """
        try:
            from hermes_cli.config import cfg_get
            key_hex = cfg_get("log_signing.key")
            if key_hex:
                key = bytes.fromhex(key_hex)
                return cls(key, chain_id)
        except Exception:
            pass
        return None

    @classmethod
    def generate_key(cls) -> bytes:
        """Generate a new random signing key."""
        return os.urandom(32)

    def sign_entry(self, entry: Dict[str, Any]) -> str:
        """Sign a log entry with the hash chain.

        Args:
            entry: The log entry to sign (must be JSON-serializable).

        Returns:
            Hex-encoded HMAC signature.
        """
        entry_bytes = json.dumps(entry, sort_keys=True, default=str).encode()
        chain_input = self._chain_hash + entry_bytes
        signature = hmac.new(self._key, chain_input, hashlib.sha256).hexdigest()
        self._chain_hash = bytes.fromhex(signature)
        return signature

    def verify_chain(
        self, entries: List[Dict[str, Any]], signatures: List[str]
    ) -> bool:
        """Verify the integrity of a log chain.

        Args:
            entries: List of log entries.
            signatures: List of corresponding signatures.

        Returns:
            True if the chain is valid, False if any entry was modified.
        """
        if len(entries) != len(signatures):
            return False
        chain_hash = hashlib.sha256(
            f"hermes-log-chain:{self._chain_id}".encode()
        ).digest()
        for entry, sig in zip(entries, signatures):
            entry_bytes = json.dumps(entry, sort_keys=True, default=str).encode()
            chain_input = chain_hash + entry_bytes
            expected = hmac.new(self._key, chain_input, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, sig):
                return False
            chain_hash = bytes.fromhex(sig)
        return True


def sign_log_entry(entry: Dict[str, Any]) -> Optional[str]:
    """Sign a log entry using the configured signer.

    Returns None if log signing is not configured.
    """
    signer = LogSigner.from_config()
    if signer is None:
        return None
    try:
        return signer.sign_entry(entry)
    except Exception as e:
        logger.warning("Failed to sign log entry: %s", e)
        return None
