# CLAUDE.md — Jarvis Integration Development Contract

## Mission

You are taking over an existing Jarvis persistent-memory integration inside the Hermes Agent repository.

This is NOT a greenfield project. Preserve the existing architecture and extend it carefully.

## Repository

Hermes Agent:
- Repository: `~/.hermes/hermes-agent`
- Upstream: `https://github.com/NousResearch/hermes-agent.git`

Jarvis RAG:
- Root: `~/.hermes/rag`
- Engine: `~/.hermes/rag/rag_memory.py`
- JSON bridge: `~/.hermes/rag/rag_bridge.py`
- Qdrant data: `~/.hermes/rag/qdrant_data`
- Embedding model: `intfloat/multilingual-e5-small`

Primary Hermes adapter:
- `plugins/memory/jarvis/__init__.py`

Primary provider tests:
- `tests/plugins/memory/test_jarvis_provider.py`

## Required first step: audit before modifying

Before changing code:

1. Read this file and the repository `AGENTS.md`.
2. Inspect the relevant existing implementation.
3. Search callers and extension points.
4. Inspect existing tests.
5. Determine whether functionality already exists before creating anything new.
6. Verify assumptions against the actual source code.
7. Produce an audit/findings summary before implementation.

The first takeover task must be READ-ONLY unless the user explicitly authorizes implementation.

## Architecture to preserve

The intended architecture is:

User Goal
→ Goal Understanding
→ Memory Recall
→ Task Planning
→ Risk Classification
→ Manual Approval when required
→ Tool Execution
→ Result Verification
→ Task Completion or Recovery
→ Important Outcome → Persistent Memory

Hermes remains the agent runtime/orchestration layer.

Jarvis RAG provides persistent semantic memory.

Integration path:

Hermes
→ MemoryManager
→ HermesRAGProvider
→ rag_memory.py
→ SentenceTransformers
→ Local Qdrant
→ `jarvis_memory`

Jarvis should EXTEND Hermes, not duplicate Hermes capabilities.

Use existing Hermes interfaces, tools, skills, and lifecycle hooks whenever possible.

## Memory lifecycle

Respect the distinction between:

### Prefetch

`prefetch()` retrieves relevant persistent memory before a turn.

### Turn synchronization

`sync_turn()` receives completed user/assistant turn information through Hermes MemoryManager and runs in the background.

Do NOT automatically store every conversation turn as durable cross-session memory without verifying the intended design.

### Session-end learning

`on_session_end()` is the session-level lifecycle hook.

Existing Jarvis RAG functions include:

- `extract_session_learnings()`
- `store_session_learnings()`
- `retrieve_cross_session_learnings()`
- `rollback_cross_session_learnings()`

These mechanisms are intentionally selective and provenance-aware.

Preserve existing secret-redaction and eligibility rules.

## Safety rules

NEVER perform these without explicit user approval:

- `git reset --hard`
- `git clean`
- destructive deletion
- `git restore` of user changes
- force push
- destructive database operations
- production/shared-system changes
- credential or secret changes
- package installation/removal when it materially changes the environment
- Docker/system configuration changes
- external communications
- irreversible actions

Do not expose secrets.

Do not read or print secret values merely for debugging.

Prefer reversible local changes.

Stop when an unexpected error changes the expected scope.

## Git safety

The repository may contain local work that must be preserved.

Before modifying anything:

- inspect `git status`
- inspect relevant `git diff`
- inspect relevant recent commits

Do NOT pull, merge, reset, rebase, clean, or force-push merely to make the repository look clean.

Never discard existing uncommitted Jarvis work.

Do not create commits unless the user explicitly authorizes the commit.

## Implementation rules

Prefer the smallest correct change.

Do not:

- create a parallel Jarvis architecture,
- duplicate Hermes core functionality,
- add speculative infrastructure,
- rewrite working components without evidence,
- weaken existing safety controls,
- bypass approval mechanisms,
- change unrelated files.

When a change is justified:

1. Explain the identified problem.
2. Identify the smallest implementation.
3. Add or update focused tests.
4. Run relevant tests.
5. Inspect the final diff.
6. Run `git diff --check`.
7. Report exactly what changed and what was verified.

## Current known state

The Jarvis provider currently has an uncommitted modification in:

`plugins/memory/jarvis/__init__.py`

That modification includes:

- removal of process-wide `os.chdir()`,
- expanded RAG usage guidance in the system prompt,
- implementation of provider-side `prefetch()`.

The RAG engine currently anchors Qdrant to its own module directory, so process-wide `cwd` changes should not be reintroduced merely to locate Qdrant.

Do not overwrite or discard this work.

The existing Jarvis provider also has lifecycle methods that require careful evaluation, especially:

- `sync_turn()`
- `on_session_end()`

Do not implement these based on assumptions. Trace the Hermes lifecycle and existing Jarvis RAG functions first.

## Testing expectations

Relevant tests include:

`tests/plugins/memory/test_jarvis_provider.py`

Also inspect relevant Hermes memory-provider tests and existing provider implementations when determining lifecycle behavior.

Tests should verify behavior, not merely method existence.

For example, a test that only checks:

`callable(provider.sync_turn)`

does not prove that `sync_turn()` works.

## Operating style

Work step-by-step.

For complex work, divide it into small, verifiable milestones.

Do not silently broaden scope.

When uncertain:

- inspect source,
- inspect tests,
- inspect callers,
- then ask for approval if a consequential choice remains.

Never claim a task is complete without verification.
