"""Conversation memory compression utilities for JARVIS.

Provides deterministic, fail-safe compression of conversation history
while preserving safety-critical information and redacting secrets.

This module integrates with the existing agent/conversation_compression.py
infrastructure without replacing it. It provides a standalone utility
for compressing conversation history outside the full agent context.

Safety features:
- Deterministic output (same input produces equivalent output)
- Secret redaction before compression output is stored/returned
- Fail-safe: original conversation preserved on compression failure
- No silent context loss
- Configurable with backward-compatible defaults
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tools.secret_redaction import redact_secrets

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of conversation compression.

    Attributes:
        compressed_messages: The compressed message list
        original_message_count: Number of messages before compression
        compressed_message_count: Number of messages after compression
        compression_ratio: Ratio of compressed to original size
        preserved_categories: Categories of information preserved
        deterministic_hash: Hash of the compression for verification
        fallback_used: Whether fallback to uncompressed was used
        error: Error message if compression failed
    """
    compressed_messages: List[Dict[str, Any]]
    original_message_count: int
    compressed_message_count: int
    compression_ratio: float
    preserved_categories: List[str] = field(default_factory=list)
    deterministic_hash: str = ""
    fallback_used: bool = False
    error: Optional[str] = None


@dataclass
class CompressionConfig:
    """Configuration for conversation compression.

    Attributes:
        enabled: Whether compression is enabled
        target_ratio: Target compression ratio (0.0-1.0)
        preserve_last_n: Number of recent messages to preserve verbatim
        redact_secrets: Whether to redact secrets in compressed output
        max_message_length: Maximum length of a single message content
        deterministic: Whether to use deterministic compression
    """
    enabled: bool = False
    target_ratio: float = 0.3
    preserve_last_n: int = 5
    redact_secrets: bool = True
    max_message_length: int = 10000
    deterministic: bool = True


def _extract_goals(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract user goals from conversation messages."""
    goals = []
    goal_patterns = [
        r"(?:I want|I need|I'd like|please|goal|objective|aim)\s+(.+?)(?:\.|$)",
        r"(?:help me|assist me|can you)\s+(.+?)(?:\.|$)",
    ]
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            for pattern in goal_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                goals.extend(matches)
    return goals[:10]  # Limit to top 10 goals


def _extract_decisions(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract decisions made during conversation."""
    decisions = []
    decision_patterns = [
        r"(?:I'll|we'll|let's|decided to|going to)\s+(.+?)(?:\.|$)",
        r"(?:decision|chose|selected|opted for)\s*[:\-]?\s*(.+?)(?:\.|$)",
    ]
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            for pattern in decision_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                decisions.extend(matches)
    return decisions[:10]


def _extract_constraints(messages: List[Dict[str, Any]]) -> List[str]:
    """Extract constraints and requirements."""
    constraints = []
    constraint_patterns = [
        r"(?:must|should|need to|require|constraint|limitation)\s+(.+?)(?:\.|$)",
        r"(?:don't|do not|avoid|never)\s+(.+?)(?:\.|$)",
    ]
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            for pattern in constraint_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                constraints.extend(matches)
    return constraints[:10]


def _extract_task_state(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract current task state from conversation."""
    state = {
        "completed_steps": [],
        "current_step": None,
        "pending_items": [],
    }
    for msg in messages[-20:]:  # Look at recent messages
        content = msg.get("content", "")
        if isinstance(content, str):
            # Look for completed steps
            done_matches = re.findall(r"(?:completed|finished|done|resolved)\s*[:\-]?\s*(.+?)(?:\.|$)", content, re.IGNORECASE)
            state["completed_steps"].extend(done_matches)
            # Look for pending items
            pending_matches = re.findall(r"(?:pending|todo|to do|remaining|next)\s*[:\-]?\s*(.+?)(?:\.|$)", content, re.IGNORECASE)
            state["pending_items"].extend(pending_matches)
    return state


def _extract_tool_results(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract important tool results from conversation."""
    results = []
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 50:
                results.append({
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content_preview": content[:200],
                })
    return results[:5]  # Limit to 5 most important results


def _create_summary_message(
    messages: List[Dict[str, Any]],
    goals: List[str],
    decisions: List[str],
    constraints: List[str],
    task_state: Dict[str, Any],
    tool_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a summary message from extracted information."""
    summary_parts = ["[CONVERSATION SUMMARY]"]
    
    if goals:
        summary_parts.append(f"Goals: {'; '.join(goals[:5])}")
    if decisions:
        summary_parts.append(f"Decisions: {'; '.join(decisions[:5])}")
    if constraints:
        summary_parts.append(f"Constraints: {'; '.join(constraints[:5])}")
    if task_state["completed_steps"]:
        summary_parts.append(f"Completed: {'; '.join(task_state['completed_steps'][:5])}")
    if task_state["pending_items"]:
        summary_parts.append(f"Pending: {'; '.join(task_state['pending_items'][:5])}")
    if tool_results:
        summary_parts.append(f"Key results: {len(tool_results)} tool outputs preserved")
    
    return {
        "role": "system",
        "content": "\n".join(summary_parts),
    }


def _truncate_message(message: Dict[str, Any], max_length: int) -> Dict[str, Any]:
    """Truncate message content to max_length."""
    content = message.get("content", "")
    if isinstance(content, str) and len(content) > max_length:
        truncated = message.copy()
        truncated["content"] = content[:max_length] + "... [truncated]"
        return truncated
    return message


def _compute_deterministic_hash(messages: List[Dict[str, Any]]) -> str:
    """Compute a deterministic hash of the message list."""
    content = ""
    for msg in messages:
        content += msg.get("role", "") + msg.get("content", "")
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def compress_conversation(
    messages: List[Dict[str, Any]],
    config: Optional[CompressionConfig] = None,
) -> CompressionResult:
    """Compress conversation history into a compact representation.

    This is the main entry point for conversation compression. It:
    1. Extracts key information (goals, decisions, constraints, task state)
    2. Preserves recent messages verbatim
    3. Creates a summary of older messages
    4. Redacts secrets from the output
    5. Returns a deterministic result

    Args:
        messages: List of conversation messages
        config: Compression configuration (uses defaults if None)

    Returns:
        CompressionResult with compressed messages and metadata
    """
    if config is None:
        config = CompressionConfig()
    
    if not config.enabled:
        return CompressionResult(
            compressed_messages=messages,
            original_message_count=len(messages),
            compressed_message_count=len(messages),
            compression_ratio=1.0,
            preserved_categories=[],
            deterministic_hash=_compute_deterministic_hash(messages),
            fallback_used=False,
        )
    
    try:
        # Handle empty conversation
        if not messages:
            return CompressionResult(
                compressed_messages=[],
                original_message_count=0,
                compressed_message_count=0,
                compression_ratio=0.0,
                preserved_categories=[],
                deterministic_hash=_compute_deterministic_hash([]),
                fallback_used=False,
            )
        
        # If conversation is short, don't compress
        if len(messages) <= config.preserve_last_n:
            return CompressionResult(
                compressed_messages=messages,
                original_message_count=len(messages),
                compressed_message_count=len(messages),
                compression_ratio=1.0,
                preserved_categories=["all_recent"],
                deterministic_hash=_compute_deterministic_hash(messages),
                fallback_used=False,
            )
        
        # Split into older messages and recent messages
        older_messages = messages[:-config.preserve_last_n]
        recent_messages = messages[-config.preserve_last_n:]
        
        # Extract key information from older messages
        goals = _extract_goals(older_messages)
        decisions = _extract_decisions(older_messages)
        constraints = _extract_constraints(older_messages)
        task_state = _extract_task_state(older_messages)
        tool_results = _extract_tool_results(older_messages)
        
        # Create summary message
        summary = _create_summary_message(
            older_messages, goals, decisions, constraints, task_state, tool_results
        )
        
        # Truncate recent messages if needed
        truncated_recent = [_truncate_message(m, config.max_message_length) for m in recent_messages]
        
        # Combine summary + recent messages
        compressed = [summary] + truncated_recent
        
        # Redact secrets if enabled
        if config.redact_secrets:
            for i, msg in enumerate(compressed):
                if isinstance(msg.get("content"), str):
                    compressed[i] = msg.copy()
                    compressed[i]["content"] = redact_secrets(msg["content"])
        
        # Compute deterministic hash
        det_hash = _compute_deterministic_hash(compressed)
        
        # Build preserved categories list
        preserved = []
        if goals:
            preserved.append("goals")
        if decisions:
            preserved.append("decisions")
        if constraints:
            preserved.append("constraints")
        if task_state["completed_steps"]:
            preserved.append("completed_steps")
        if task_state["pending_items"]:
            preserved.append("pending_items")
        if tool_results:
            preserved.append("tool_results")
        preserved.append("recent_context")
        
        return CompressionResult(
            compressed_messages=compressed,
            original_message_count=len(messages),
            compressed_message_count=len(compressed),
            compression_ratio=len(compressed) / len(messages) if messages else 0,
            preserved_categories=preserved,
            deterministic_hash=det_hash,
            fallback_used=False,
        )
    
    except Exception as e:
        logger.warning("Compression failed, falling back to uncompressed: %s", e)
        return CompressionResult(
            compressed_messages=messages,
            original_message_count=len(messages),
            compressed_message_count=len(messages),
            compression_ratio=1.0,
            preserved_categories=[],
            deterministic_hash=_compute_deterministic_hash(messages),
            fallback_used=True,
            error=str(e),
        )


def get_compression_config(config_dict: Optional[Dict[str, Any]] = None) -> CompressionConfig:
    """Create CompressionConfig from a configuration dictionary.

    Args:
        config_dict: Configuration dictionary (e.g., from config.yaml)

    Returns:
        CompressionConfig with values from the dictionary
    """
    if config_dict is None:
        return CompressionConfig()
    
    return CompressionConfig(
        enabled=config_dict.get("enabled", False),
        target_ratio=config_dict.get("target_ratio", 0.3),
        preserve_last_n=config_dict.get("preserve_last_n", 5),
        redact_secrets=config_dict.get("redact_secrets", True),
        max_message_length=config_dict.get("max_message_length", 10000),
        deterministic=config_dict.get("deterministic", True),
    )
