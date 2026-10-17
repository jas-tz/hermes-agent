import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { bindApi, type Task } from './api'
import { MEMORY_CARD_LOCALES, MemoryCard } from './memory-card'
import { activeTasks, TASK_LIMIT, TasksBar } from './tasks-bar'

const mocks = vi.hoisted(() => ({ bundle: {} as Record<string, unknown>, navigate: vi.fn() }))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()
  const host = actual.host as Record<string, unknown>

  // Resolve the plugin's own English bundle (registered via ctx.i18n in the app).
  const usePluginI18n =
    () =>
    (key: string, ...args: unknown[]) => {
      const value = key
        .split('.')
        .reduce<unknown>(
          (node, part) => (node && typeof node === 'object' ? (node as Record<string, unknown>)[part] : undefined),
          mocks.bundle
        )

      if (typeof value === 'function') {
        return (value as (...a: unknown[]) => string)(...args)
      }

      return typeof value === 'string' ? value : key
    }

  return { ...actual, host: { ...host, navigate: mocks.navigate }, usePluginI18n }
})

type Handler = (path: string) => unknown

const paths: string[] = []
let handler: Handler = () => ({ enabled: false, reason: 'provider_not_jarvis' })

let unbind: () => void = () => {}

function rest<T>(path: string): Promise<T> {
  paths.push(path.split('?')[0])

  return Promise.resolve(handler(path) as T)
}

function wrap(node: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>)
}

const task = (over: Partial<Task>): Task => ({
  created_at: null,
  deadline_at: null,
  description: null,
  goal_id: null,
  id: over.title ?? 'id',
  priority: null,
  provenance: null,
  status: 'in_progress',
  title: 'Task',
  updated_at: null,
  ...over
})

const continuity = (tasks: Task[], goals: string[] = []) => ({
  enabled: true,
  goals: {
    items: goals.map(title => ({ ...task({ title }), status: 'active' })),
    total: goals.length,
    truncated: false
  },
  tasks: { items: tasks, total: tasks.length, truncated: false },
  view: 'list'
})

beforeAll(() => {
  mocks.bundle = MEMORY_CARD_LOCALES.en as Record<string, unknown>
})

beforeEach(() => {
  paths.length = 0
  mocks.navigate.mockClear()
  unbind = bindApi(rest as never)
})

afterEach(() => {
  cleanup()
  unbind()
})

describe('activeTasks', () => {
  it('keeps open tasks, highest priority first', () => {
    const tasks = [
      task({ priority: 1, title: 'low' }),
      task({ status: 'done', title: 'finished' }),
      task({ priority: 10, title: 'high' }),
      task({ status: 'cancelled', title: 'dropped' }),
      task({ priority: 1, status: 'pending', title: 'low-2' })
    ]

    expect(activeTasks(tasks).map(t => t.title)).toEqual(['high', 'low', 'low-2'])
  })

  it('shows at most TASK_LIMIT tasks', () => {
    expect(activeTasks(Array.from({ length: 9 }, (_, i) => task({ title: `t${i}` })))).toHaveLength(TASK_LIMIT)
  })
})

describe('TasksBar', () => {
  it('lists open tasks with their real status', async () => {
    handler = () =>
      continuity([
        task({ priority: 10, title: 'Use Jarvis in normal work' }),
        task({ status: 'pending', title: 'Eval benchmark' }),
        task({ status: 'done', title: 'Old' })
      ])
    wrap(<TasksBar route="/jarvis" />)

    await waitFor(() => expect(screen.getAllByTestId('jarvis-task')).toHaveLength(2))
    const [first, second] = screen.getAllByTestId('jarvis-task')

    expect(first.textContent).toBe('Use Jarvis in normal workIn progress')
    expect(first.getAttribute('data-status')).toBe('in_progress')
    expect(second.textContent).toBe('Eval benchmarkQueued')
    expect(screen.getByText('Active tasks')).toBeTruthy()
  })

  it('says so when nothing is open', async () => {
    handler = () => continuity([task({ status: 'done' })])
    wrap(<TasksBar route="/jarvis" />)

    await waitFor(() => expect(screen.getByText('No active tasks')).toBeTruthy())
  })

  it('renders nothing when JARVIS memory is not the provider', async () => {
    handler = () => ({ enabled: false, reason: 'provider_not_jarvis' })
    const { container } = wrap(<TasksBar route="/jarvis" />)

    await waitFor(() => expect(paths).toContain('/continuity'))
    expect(container.textContent).toBe('')
  })

  it('opens the memory page from "All tasks"', async () => {
    handler = () => continuity([task({ title: 'One' })])
    wrap(<TasksBar route="/jarvis" />)

    fireEvent.click(await screen.findByText('All tasks →'))
    expect(mocks.navigate).toHaveBeenCalledWith('/jarvis')
  })
})

describe('MemoryCard', () => {
  it('shows the top goals and opens the memory vault', async () => {
    handler = path =>
      path.startsWith('/status')
        ? { continuity: { decisions: 2, goals: 4, state: 'available', tasks: 3 }, enabled: true, rag: {} }
        : continuity([], ['Digital factory', 'Jarvis build', 'Preferences', 'Fourth'])
    wrap(<MemoryCard route="/jarvis" />)

    await waitFor(() =>
      expect(screen.getByTestId('jarvis-memory-card-summary').textContent).toBe(
        'Digital factory · Jarvis build · Preferences'
      )
    )

    fireEvent.click(screen.getByText('Open memory vault →'))
    expect(mocks.navigate).toHaveBeenCalledWith('/jarvis')
  })

  it('falls back to the continuity counts without goals', async () => {
    handler = path =>
      path.startsWith('/status')
        ? { continuity: { decisions: 2, goals: 0, state: 'available', tasks: 3 }, enabled: true, rag: {} }
        : continuity([])
    wrap(<MemoryCard route="/jarvis" />)

    await waitFor(() =>
      expect(screen.getByTestId('jarvis-memory-card-summary').textContent).toBe('0 goals · 3 tasks · 2 decisions')
    )
  })

  it('reads only — status and continuity, nothing else', async () => {
    handler = path => (path.startsWith('/status') ? { enabled: false, reason: 'x' } : continuity([]))
    wrap(
      <>
        <MemoryCard route="/jarvis" />
        <TasksBar route="/jarvis" />
      </>
    )

    await waitFor(() => expect(screen.getByTestId('jarvis-memory-card-summary').textContent).toBe('Memory unavailable'))
    expect(new Set(paths)).toEqual(new Set(['/status', '/continuity']))
  })
})
