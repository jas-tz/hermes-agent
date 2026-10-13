/**
 * JARVIS memory page — a strictly read-only view over the JARVIS dashboard
 * plugin API: status, continuity (goals/tasks/decisions), memory search,
 * cross-session learning history and pending proposal summaries.
 *
 * Safety posture:
 *  - Data only arrives through `api.ts` (GET-only, plugin-namespaced REST).
 *  - The page renders plain text nodes only (no HTML/markdown), and only the
 *    explicitly typed fields — provenance is shown through a fixed key list.
 *  - Errors never surface `error.message`; every failure renders a fixed,
 *    generic "unavailable" state.
 *  - There are no approve/discard/delete/edit/write/rollback controls; the only
 *    actions are tab switches, search submits and refresh.
 */
import {
  Badge,
  Button,
  cn,
  EmptyState,
  ErrorState,
  Loader,
  SearchField,
  SegmentedControl,
  useQuery,
  useQueryClient
} from '@hermes/plugin-sdk'
import { type FormEvent, type ReactNode, useState } from 'react'

import {
  type ContinuityCounts,
  continuityKey,
  type Decision,
  fetchContinuity,
  fetchPending,
  fetchStatus,
  type Goal,
  isDisabled,
  learningsKey,
  type MemoryResult,
  normalizeQuery,
  pendingKey,
  type PendingSubsystem,
  QUERY_MAX_CHARS,
  type RagHealth,
  searchKey,
  searchLearnings,
  searchMemory,
  type Section,
  statusKey,
  type Task
} from './api'

type TabId = 'continuity' | 'learnings' | 'pending' | 'search' | 'status'

const TABS: readonly { id: TabId; label: string }[] = [
  { id: 'status', label: 'Status' },
  { id: 'continuity', label: 'Continuity' },
  { id: 'search', label: 'Search' },
  { id: 'learnings', label: 'Learnings' },
  { id: 'pending', label: 'Pending' }
]

const PENDING_TABS: readonly { id: PendingSubsystem; label: string }[] = [
  { id: 'memory', label: 'Memory' },
  { id: 'continuity', label: 'Continuity' }
]

/** Fixed, human copy for every backend reason code — never raw error text. */
const RAG_REASONS: Record<string, string> = {
  collection_missing: 'The JARVIS memory collection was not found.',
  config_error: 'The JARVIS memory backend is not configured.',
  engine_not_installed: 'The JARVIS RAG engine is not installed.',
  engine_unavailable: 'The JARVIS RAG engine could not be loaded.',
  health_check_failed: 'The memory health check did not succeed.',
  local_mode_unverified: 'Local storage mode is not checked from the dashboard.',
  server_unreachable: 'The memory server is not reachable.'
}

/** The only provenance keys ever rendered, in display order. */
const PROVENANCE_LABELS: readonly [string, string][] = [
  ['source', 'Source'],
  ['source_type', 'Kind'],
  ['provenance_status', 'Provenance'],
  ['origin', 'Origin'],
  ['transfer_status', 'Transfer'],
  ['source_id', 'Session'],
  ['created_at', 'Created'],
  ['stored_at', 'Stored']
]

const QUERY_OPTIONS = { refetchOnWindowFocus: false, retry: false } as const

// ── formatting ───────────────────────────────────────────────────────────────

/** Epoch seconds (numbers) or ISO strings → a locale timestamp, else null. */
export function formatTime(value: unknown): null | string {
  let date: Date | null = null

  if (typeof value === 'number' && Number.isFinite(value)) {
    date = new Date(value * 1000)
  } else if (typeof value === 'string' && value) {
    date = new Date(value)
  }

  return date && !Number.isNaN(date.getTime()) ? date.toLocaleString() : null
}

function provenanceEntries(provenance: MemoryResult['provenance']): [string, string][] {
  if (!provenance) {
    return []
  }

  const entries: [string, string][] = []

  for (const [key, label] of PROVENANCE_LABELS) {
    const value = provenance[key]

    if (value === null || value === undefined || value === '') {
      continue
    }

    const text = key.endsWith('_at') ? (formatTime(value) ?? String(value)) : String(value)
    entries.push([label, text])
  }

  return entries
}

// ── shared states ────────────────────────────────────────────────────────────

function LoadingState({ label }: { label: string }) {
  return (
    <div className="grid min-h-48 place-items-center" data-testid="jarvis-loading">
      <Loader label={label} type="lemniscate-bloom" />
    </div>
  )
}

function DisabledState() {
  return (
    <EmptyState
      description='JARVIS is not the active memory provider. Set memory.provider to "jarvis" to view memory here.'
      title="JARVIS memory is disabled"
    />
  )
}

function UnavailableState({ what }: { what: string }) {
  return (
    <ErrorState
      className="py-10"
      description="The JARVIS dashboard could not load this view. Try again later."
      title={`${what} is unavailable`}
    />
  )
}

function Panel({ children, title }: { children: ReactNode; title: string }) {
  return (
    <section className="grid gap-2 rounded-md border border-(--ui-stroke-secondary) p-3">
      <h2 className="text-xs font-semibold tracking-wide text-(--ui-text-tertiary) uppercase">{title}</h2>
      {children}
    </section>
  )
}

function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right text-foreground">{value}</span>
    </div>
  )
}

function Body({ text, truncated }: { text: null | string; truncated?: boolean }) {
  if (!text) {
    return null
  }

  return (
    <p className="text-sm break-words whitespace-pre-wrap text-foreground">
      {text}
      {truncated && <span className="text-muted-foreground"> …</span>}
    </p>
  )
}

function Badges({ items }: { items: (false | null | string | undefined)[] }) {
  const shown = items.filter((item): item is string => Boolean(item))

  if (shown.length === 0) {
    return null
  }

  return (
    <div className="flex flex-wrap gap-1">
      {shown.map(item => (
        <Badge key={item} variant="muted">
          {item}
        </Badge>
      ))}
    </div>
  )
}

// ── status ───────────────────────────────────────────────────────────────────

function ragSummary(rag: RagHealth): string {
  if (rag.state === 'available') {
    return 'Available'
  }

  return (rag.reason && RAG_REASONS[rag.reason]) || 'Unavailable'
}

function countText(value: null | number): string {
  return value === null ? '—' : String(value)
}

function StatusView() {
  const { data, isError, isPending } = useQuery({ queryFn: fetchStatus, queryKey: statusKey, ...QUERY_OPTIONS })

  if (isPending) {
    return <LoadingState label="Loading status" />
  }

  if (isError || !data) {
    return <UnavailableState what="Memory status" />
  }

  if (isDisabled(data)) {
    return <DisabledState />
  }

  const continuity: ContinuityCounts = data.continuity

  return (
    <div className="grid gap-3 md:grid-cols-2">
      <Panel title="Semantic memory">
        <Row
          label="State"
          value={
            <Badge variant={data.rag.state === 'available' ? 'default' : 'destructive'}>
              {data.rag.state === 'available' ? 'Available' : 'Unavailable'}
            </Badge>
          }
        />
        <p className="text-xs text-muted-foreground">{ragSummary(data.rag)}</p>
        <Row label="Mode" value={data.rag.mode ?? '—'} />
        <Row
          label="Collection"
          value={data.rag.collection_exists === null ? '—' : data.rag.collection_exists ? 'Present' : 'Missing'}
        />
        <Row
          label="Server mode required"
          value={data.rag.server_required === null ? '—' : data.rag.server_required ? 'Yes' : 'No'}
        />
      </Panel>
      <Panel title="Continuity">
        {continuity.state === 'available' ? (
          <>
            <Row label="Active or blocked goals" value={countText(continuity.goals)} />
            <Row label="Open or blocked tasks" value={countText(continuity.tasks)} />
            <Row label="Current decisions" value={countText(continuity.decisions)} />
          </>
        ) : (
          <p className="text-sm text-muted-foreground">Continuity data is unavailable.</p>
        )}
      </Panel>
    </div>
  )
}

// ── continuity ───────────────────────────────────────────────────────────────

function SectionList<T>({
  empty,
  render,
  section,
  title
}: {
  empty: string
  render: (item: T) => ReactNode
  section: Section<T> | undefined
  title: string
}) {
  if (!section) {
    return null
  }

  return (
    <Panel title={`${title} (${section.total})`}>
      {section.items.length === 0 ? (
        <p className="text-sm text-muted-foreground">{empty}</p>
      ) : (
        <ul className="grid gap-3">{section.items.map(render)}</ul>
      )}
      {section.truncated && (
        <p className="text-xs text-muted-foreground">
          Showing the first {section.items.length} of {section.total}.
        </p>
      )}
    </Panel>
  )
}

function GoalItem({ goal }: { goal: Goal }) {
  return (
    <li className="grid gap-1">
      <div className="text-sm font-medium break-words">{goal.title}</div>
      <Badges items={[goal.status, goal.priority ? `Priority ${goal.priority}` : null]} />
      <Body text={goal.description} />
    </li>
  )
}

function TaskItem({ task }: { task: Task }) {
  const deadline = formatTime(task.deadline_at)

  return (
    <li className="grid gap-1">
      <div className="text-sm font-medium break-words">{task.title}</div>
      <Badges
        items={[task.status, task.priority ? `Priority ${task.priority}` : null, deadline && `Due ${deadline}`]}
      />
      <Body text={task.description} />
    </li>
  )
}

function DecisionItem({ decision }: { decision: Decision }) {
  const created = formatTime(decision.created_at)

  return (
    <li className="grid gap-1">
      <div className="text-sm font-medium break-words">{decision.decision}</div>
      <Badges items={[created && `Recorded ${created}`]} />
      <Body text={decision.rationale} />
    </li>
  )
}

function ContinuityView() {
  const { data, isError, isPending } = useQuery({
    queryFn: fetchContinuity,
    queryKey: continuityKey,
    ...QUERY_OPTIONS
  })

  if (isPending) {
    return <LoadingState label="Loading continuity" />
  }

  if (isError || !data) {
    return <UnavailableState what="Continuity" />
  }

  if (isDisabled(data)) {
    return <DisabledState />
  }

  return (
    <div className="grid gap-3">
      <SectionList<Goal>
        empty="No active or blocked goals."
        render={goal => <GoalItem goal={goal} key={goal.id} />}
        section={data.goals}
        title="Goals"
      />
      <SectionList<Task>
        empty="No open or blocked tasks."
        render={task => <TaskItem key={task.id} task={task} />}
        section={data.tasks}
        title="Tasks"
      />
      <SectionList<Decision>
        empty="No current decisions."
        render={decision => <DecisionItem decision={decision} key={decision.id} />}
        section={data.decisions}
        title="Decisions"
      />
    </div>
  )
}

// ── search + learnings ───────────────────────────────────────────────────────

function ResultCard({ result }: { result: MemoryResult }) {
  return (
    <li className="grid gap-1.5 rounded-md border border-(--ui-stroke-secondary) p-3">
      <Badges
        items={[
          result.score === null ? null : `Score ${result.score.toFixed(2)}`,
          result.type,
          ...provenanceEntries(result.provenance).map(([label, value]) => `${label}: ${value}`)
        ]}
      />
      <Body text={result.text} truncated={result.text_truncated} />
    </li>
  )
}

interface QueryViewProps {
  emptyResults: string
  intro: string
  kind: 'learnings' | 'search'
  placeholder: string
  what: string
}

function QueryView({ emptyResults, intro, kind, placeholder, what }: QueryViewProps) {
  const [draft, setDraft] = useState('')
  const [submitted, setSubmitted] = useState('')

  const { data, isError, isFetching } = useQuery({
    enabled: submitted.length > 0,
    queryFn: () => (kind === 'search' ? searchMemory(submitted) : searchLearnings(submitted)),
    queryKey: kind === 'search' ? searchKey(submitted) : learningsKey(submitted),
    ...QUERY_OPTIONS
  })

  const submit = (event: FormEvent) => {
    event.preventDefault()
    setSubmitted(normalizeQuery(draft))
  }

  let body: ReactNode

  if (!submitted) {
    body = <EmptyState description={intro} title={what} />
  } else if (isFetching && !data) {
    body = <LoadingState label="Searching" />
  } else if (isError || !data) {
    body = <UnavailableState what={what} />
  } else if (isDisabled(data)) {
    body = <DisabledState />
  } else if (data.results.length === 0) {
    body = <EmptyState title={emptyResults} />
  } else {
    body = (
      <ul className="grid gap-2">
        {data.results.map((result, index) => (
          <ResultCard key={index} result={result} />
        ))}
      </ul>
    )
  }

  return (
    <div className="grid gap-3">
      <form className="flex items-center gap-2" onSubmit={submit} role="search">
        <SearchField
          aria-label={placeholder}
          containerClassName="flex-1"
          loading={isFetching}
          onChange={value => setDraft(value.slice(0, QUERY_MAX_CHARS))}
          onClear={() => {
            setDraft('')
            setSubmitted('')
          }}
          placeholder={placeholder}
          value={draft}
        />
        <Button disabled={!normalizeQuery(draft)} size="sm" type="submit" variant="secondary">
          Search
        </Button>
      </form>
      {body}
    </div>
  )
}

// ── pending ──────────────────────────────────────────────────────────────────

function PendingView() {
  const [subsystem, setSubsystem] = useState<PendingSubsystem>('memory')

  const { data, isError, isPending } = useQuery({
    queryFn: () => fetchPending(subsystem),
    queryKey: pendingKey(subsystem),
    ...QUERY_OPTIONS
  })

  let body: ReactNode

  if (isPending) {
    body = <LoadingState label="Loading pending proposals" />
  } else if (isError || !data) {
    body = <UnavailableState what="Pending proposals" />
  } else if (isDisabled(data)) {
    body = <DisabledState />
  } else if (data.items.length === 0) {
    body = <EmptyState title={`No pending ${subsystem} proposals`} />
  } else {
    body = (
      <ul className="grid gap-2">
        {data.items.map((item, index) => (
          <li className="grid gap-1.5 rounded-md border border-(--ui-stroke-secondary) p-3" key={item.id ?? index}>
            <Badges items={[item.action, item.origin, formatTime(item.created_at)]} />
            <Body text={item.summary} truncated={item.summary_truncated} />
          </li>
        ))}
        {data.truncated && (
          <li className="text-xs text-muted-foreground">
            Showing the first {data.items.length} of {data.total}.
          </li>
        )}
      </ul>
    )
  }

  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <SegmentedControl onChange={setSubsystem} options={PENDING_TABS} value={subsystem} />
        <p className="text-xs text-muted-foreground">Summaries only. Proposals are reviewed outside this view.</p>
      </div>
      {body}
    </div>
  )
}

// ── page ─────────────────────────────────────────────────────────────────────

export function JarvisMemoryPage() {
  const queryClient = useQueryClient()
  const [tab, setTab] = useState<TabId>('status')

  return (
    <div className="relative flex h-full flex-col overflow-hidden bg-(--ui-surface-background)">
      <header className="flex shrink-0 flex-wrap items-center gap-2 px-4 py-2">
        <h1 className="text-sm font-semibold text-foreground">JARVIS Memory</h1>
        <Badge variant="outline">Read-only</Badge>
        <div className="ml-auto flex items-center gap-2">
          <SegmentedControl onChange={setTab} options={TABS} value={tab} />
          <Button
            onClick={() => void queryClient.invalidateQueries({ queryKey: ['jarvis'] })}
            size="sm"
            type="button"
            variant="ghost"
          >
            Refresh
          </Button>
        </div>
      </header>
      <div className={cn('min-h-0 flex-1 overflow-y-auto px-4 pb-6')}>
        {tab === 'status' && <StatusView />}
        {tab === 'continuity' && <ContinuityView />}
        {tab === 'search' && (
          <QueryView
            emptyResults="No matching memories"
            intro="Search JARVIS semantic memory. Results show redacted text and provenance."
            kind="search"
            placeholder="Search memory"
            what="Memory search"
          />
        )}
        {tab === 'learnings' && (
          <QueryView
            emptyResults="No matching learnings"
            intro="Search learnings carried across sessions, with their session provenance."
            kind="learnings"
            placeholder="Search learning history"
            what="Learning history"
          />
        )}
        {tab === 'pending' && <PendingView />}
      </div>
    </div>
  )
}
