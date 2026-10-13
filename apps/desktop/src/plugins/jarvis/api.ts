/**
 * JARVIS memory — typed client for the read-only backend plugin
 * (`plugins/memory/jarvis/dashboard/plugin_api.py`, mounted at
 * `/api/plugins/jarvis`).
 *
 * Every call goes through `ctx.rest`, the authenticated, profile-aware,
 * namespace-scoped plugin door. Only GETs are issued — no `method`, no `body` —
 * and only against the five read routes below. There is deliberately no write,
 * approve, discard, delete, edit or rollback call in this module.
 */
import type { PluginRestOptions } from '@hermes/plugin-sdk'

type Rest = <T>(path: string, opts?: PluginRestOptions) => Promise<T>

let rest: null | Rest = null

/** Bind the plugin's REST door on register; the returned disposer unbinds it. */
export function bindApi(r: Rest): () => void {
  rest = r

  return () => {
    rest = null
  }
}

function get<T>(path: string, params: Record<string, string> = {}): Promise<T> {
  if (!rest) {
    return Promise.reject(new Error('jarvis api not ready'))
  }

  const qs = new URLSearchParams(params).toString()

  return rest<T>(qs ? `${path}?${qs}` : path)
}

/** Mirrors the backend cap (shared memory-retrieval query limit). */
export const QUERY_MAX_CHARS = 320
/** Mirrors the backend hard cap on results per request. */
export const RESULT_LIMIT = 20

// ── response shapes (exactly the backend allowlists) ─────────────────────────

export interface Disabled {
  enabled: false
  reason: string
}

export interface RagHealth {
  collection_exists: boolean | null
  mode: 'local' | 'server' | null
  reason: null | string
  server_required: boolean | null
  state: 'available' | 'unavailable'
}

export interface ContinuityCounts {
  decisions: null | number
  goals: null | number
  state: 'available' | 'unavailable'
  tasks: null | number
}

export interface StatusEnabled {
  continuity: ContinuityCounts
  enabled: true
  rag: RagHealth
}

export type StatusResponse = Disabled | StatusEnabled

type Provenance = Record<string, boolean | null | number | string> | null

export interface Goal {
  created_at: null | number
  description: null | string
  id: string
  priority: null | number
  provenance: Provenance
  status: string
  title: string
  updated_at: null | number
}

export interface Task extends Goal {
  deadline_at: null | number
  goal_id: null | string
}

export interface Decision {
  created_at: null | number
  decision: string
  id: string
  provenance: Provenance
  rationale: null | string
}

export interface Section<T> {
  items: T[]
  total: number
  truncated: boolean
}

export interface ContinuityEnabled {
  decisions?: Section<Decision>
  enabled: true
  goals?: Section<Goal>
  tasks?: Section<Task>
  view: string
}

export type ContinuityResponse = ContinuityEnabled | Disabled

export interface MemoryResult {
  provenance: Provenance
  score: null | number
  text: string
  text_truncated: boolean
  type: null | string
}

export interface ResultsEnabled {
  count: number
  enabled: true
  results: MemoryResult[]
}

export type ResultsResponse = Disabled | ResultsEnabled

export type PendingSubsystem = 'continuity' | 'memory'

export interface PendingItem {
  action: null | string
  created_at: null | number
  id: null | string
  origin: null | string
  summary: string
  summary_truncated: boolean
}

export interface PendingEnabled {
  enabled: true
  items: PendingItem[]
  subsystem: PendingSubsystem
  total: number
  truncated: boolean
}

export type PendingResponse = Disabled | PendingEnabled

// ── fetchers ─────────────────────────────────────────────────────────────────

export const fetchStatus = () => get<StatusResponse>('/status')

export const fetchContinuity = () => get<ContinuityResponse>('/continuity', { view: 'list' })

/** Trimmed and length-capped exactly like the backend; never empty. */
export function normalizeQuery(raw: string): string {
  return raw.trim().slice(0, QUERY_MAX_CHARS)
}

export const searchMemory = (query: string) =>
  get<ResultsResponse>('/search', { limit: String(RESULT_LIMIT), q: normalizeQuery(query) })

export const searchLearnings = (query: string) =>
  get<ResultsResponse>('/learnings', { limit: String(RESULT_LIMIT), q: normalizeQuery(query) })

export const fetchPending = (subsystem: PendingSubsystem) => get<PendingResponse>('/pending', { subsystem })

export const isDisabled = (value: { enabled: boolean } | undefined): value is Disabled => value?.enabled === false

// ── query keys ───────────────────────────────────────────────────────────────

export const statusKey = ['jarvis', 'status'] as const
export const continuityKey = ['jarvis', 'continuity'] as const
export const searchKey = (query: string) => ['jarvis', 'search', query] as const
export const learningsKey = (query: string) => ['jarvis', 'learnings', query] as const
export const pendingKey = (subsystem: PendingSubsystem) => ['jarvis', 'pending', subsystem] as const
