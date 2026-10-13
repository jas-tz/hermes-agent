import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { bindApi } from './api'
import { formatTime, JarvisMemoryPage } from './page'

const SECRET = 'sk-A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6'
const ALLOWED_PATHS = ['/status', '/continuity', '/search', '/learnings', '/pending']
const FORBIDDEN_CONTROL = /approve|discard|delete|edit|rollback|remove|store|write/i

type Handler = (path: string) => unknown

const calls: { opts: unknown; path: string }[] = []
let handler: Handler = () => ({ enabled: false, reason: 'provider_not_jarvis' })

let unbind: () => void = () => {}

function rest<T>(path: string, opts?: unknown): Promise<T> {
  calls.push({ opts, path })

  try {
    const value = handler(path)

    return value instanceof Error ? Promise.reject(value) : Promise.resolve(value as T)
  } catch (error) {
    return Promise.reject(error)
  }
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <JarvisMemoryPage />
    </QueryClientProvider>
  )
}

function openTab(label: string) {
  const tab = screen.getAllByRole('button', { name: label }).find(el => el.hasAttribute('aria-pressed'))

  if (!tab) {
    throw new Error(`no tab ${label}`)
  }

  fireEvent.click(tab)
}

const STATUS_OK = {
  continuity: { decisions: 1, goals: 2, state: 'available', tasks: 3 },
  enabled: true,
  rag: { collection_exists: true, mode: 'server', reason: null, server_required: true, state: 'available' }
}

beforeEach(() => {
  calls.length = 0
  handler = () => ({ enabled: false, reason: 'provider_not_jarvis' })
  unbind = bindApi(rest as never)
})

afterEach(() => {
  unbind()
  cleanup()
})

describe('JARVIS memory page', () => {
  it('shows status and counts', async () => {
    handler = () => STATUS_OK
    renderPage()

    expect(screen.getByTestId('jarvis-loading')).toBeTruthy()
    await screen.findByText('Active or blocked goals')
    expect(screen.getAllByText('Available').length).toBeGreaterThan(0)
    expect(screen.getByText('Present')).toBeTruthy()
    expect(calls).toEqual([{ opts: undefined, path: '/status' }])
  })

  it('maps unavailable reasons to fixed copy', async () => {
    handler = () => ({
      ...STATUS_OK,
      rag: {
        collection_exists: null,
        mode: 'server',
        reason: 'server_unreachable',
        server_required: true,
        state: 'unavailable'
      }
    })
    renderPage()

    await screen.findByText('The memory server is not reachable.')
    expect(screen.getByText('Unavailable')).toBeTruthy()
  })

  it('shows the disabled state', async () => {
    renderPage()

    await screen.findByText('JARVIS memory is disabled')
  })

  it('never renders raw error details', async () => {
    handler = () => new Error(`503 ${SECRET} http://127.0.0.1:6333 /home/jas/.hermes/rag`)
    const { container } = renderPage()

    await screen.findByText('Memory status is unavailable')
    expect(container.textContent).not.toContain(SECRET)
    expect(container.textContent).not.toContain('6333')
    expect(container.textContent).not.toContain('/home/jas')
  })

  it('lists continuity sections with empty and truncated states', async () => {
    handler = path =>
      path.startsWith('/continuity')
        ? {
            decisions: { items: [], total: 0, truncated: false },
            enabled: true,
            goals: {
              items: [
                {
                  created_at: 1,
                  description: 'Desktop milestone',
                  id: 'g1',
                  priority: 2,
                  provenance: null,
                  status: 'active',
                  title: 'Ship JARVIS desktop',
                  updated_at: 1
                }
              ],
              total: 51,
              truncated: true
            },
            tasks: { items: [], total: 0, truncated: false },
            view: 'list'
          }
        : STATUS_OK
    renderPage()
    openTab('Continuity')

    await screen.findByText('Ship JARVIS desktop')
    expect(screen.getByText('Priority 2')).toBeTruthy()
    expect(screen.getByText('Showing the first 1 of 51.')).toBeTruthy()
    expect(screen.getByText('No open or blocked tasks.')).toBeTruthy()
    expect(screen.getByText('No current decisions.')).toBeTruthy()
    expect(calls.map(c => c.path)).toContain('/continuity?view=list')
  })

  it('searches only on submit, caps the query, and renders only allowlisted provenance', async () => {
    handler = path =>
      path.startsWith('/search')
        ? {
            count: 1,
            enabled: true,
            results: [
              {
                provenance: { secret_field: SECRET, source: 'user_confirmed', stored_at: '2026-10-01T00:00:00+00:00' },
                score: 0.912,
                text: 'Deploy window is Friday',
                text_truncated: false,
                type: 'memory'
              }
            ]
          }
        : STATUS_OK
    const { container } = renderPage()
    openTab('Search')

    const input = screen.getByRole('textbox', { name: 'Search memory' })
    fireEvent.change(input, { target: { value: 'x'.repeat(400) } })
    expect(calls.some(c => c.path.startsWith('/search'))).toBe(false)

    fireEvent.submit(screen.getByRole('search'))
    await screen.findByText('Deploy window is Friday')

    const search = calls.find(c => c.path.startsWith('/search'))
    expect(search?.opts).toBeUndefined()
    const params = new URLSearchParams(search!.path.split('?')[1])
    expect(params.get('q')).toHaveLength(320)
    expect(params.get('limit')).toBe('20')
    expect(screen.getByText('Score 0.91')).toBeTruthy()
    expect(screen.getByText('Source: user_confirmed')).toBeTruthy()
    expect(container.textContent).not.toContain(SECRET)
    expect(container.textContent).not.toContain('secret_field')
  })

  it('searches learning history through the learnings route', async () => {
    handler = path =>
      path.startsWith('/learnings')
        ? {
            count: 1,
            enabled: true,
            results: [
              {
                provenance: { source_id: '20261001_abc', source_type: 'session_summary' },
                score: 0.88,
                text: 'Prefers phase-by-phase delivery',
                text_truncated: true,
                type: 'memory'
              }
            ]
          }
        : STATUS_OK
    renderPage()
    openTab('Learnings')

    fireEvent.change(screen.getByRole('textbox', { name: 'Search learning history' }), {
      target: { value: '  delivery  ' }
    })
    fireEvent.submit(screen.getByRole('search'))

    await screen.findByText('Prefers phase-by-phase delivery')
    expect(screen.getByText('Session: 20261001_abc')).toBeTruthy()
    expect(calls.map(c => c.path)).toContain('/learnings?limit=20&q=delivery')
  })

  it('shows empty search results and the unavailable state', async () => {
    let fail = false

    handler = path => {
      if (path.startsWith('/search')) {
        return fail ? new Error('boom') : { count: 0, enabled: true, results: [] }
      }

      return STATUS_OK
    }

    const { container } = renderPage()
    openTab('Search')

    fireEvent.change(screen.getByRole('textbox', { name: 'Search memory' }), { target: { value: 'nothing' } })
    fireEvent.submit(screen.getByRole('search'))
    await screen.findByText('No matching memories')

    fail = true
    fireEvent.change(screen.getByRole('textbox', { name: 'Search memory' }), { target: { value: 'other' } })
    fireEvent.submit(screen.getByRole('search'))
    await screen.findByText('Memory search is unavailable')
    expect(container.textContent).not.toContain('boom')
  })

  it('shows pending summaries per subsystem', async () => {
    handler = path =>
      path === '/pending?subsystem=memory'
        ? {
            enabled: true,
            items: [
              {
                action: 'add',
                created_at: 10,
                id: 'a1',
                origin: 'foreground',
                summary: 'Add: prefers dark mode',
                summary_truncated: false
              }
            ],
            subsystem: 'memory',
            total: 1,
            truncated: false
          }
        : path.startsWith('/pending')
          ? { enabled: true, items: [], subsystem: 'continuity', total: 0, truncated: false }
          : STATUS_OK
    renderPage()
    openTab('Pending')

    await screen.findByText('Add: prefers dark mode')
    // The page-level "Continuity" tab and the pending "Continuity" toggle share a
    // label; the pending toggle is the second one.
    const toggles = screen.getAllByRole('button', { name: 'Continuity' })
    fireEvent.click(toggles[toggles.length - 1])

    await waitFor(() => expect(calls.map(c => c.path)).toContain('/pending?subsystem=continuity'))
  })

  it('offers no write controls and only issues GETs to allowed routes', async () => {
    handler = path => {
      if (path.startsWith('/continuity')) {
        return {
          decisions: { items: [], total: 0, truncated: false },
          enabled: true,
          goals: { items: [], total: 0, truncated: false },
          tasks: { items: [], total: 0, truncated: false },
          view: 'list'
        }
      }

      if (path.startsWith('/pending')) {
        return { enabled: true, items: [], subsystem: 'memory', total: 0, truncated: false }
      }

      return STATUS_OK
    }

    renderPage()

    for (const label of ['Status', 'Continuity', 'Search', 'Learnings', 'Pending']) {
      openTab(label)
      await waitFor(() => expect(screen.queryByTestId('jarvis-loading')).toBeNull())

      for (const button of screen.getAllByRole('button')) {
        expect(button.textContent ?? '').not.toMatch(FORBIDDEN_CONTROL)
        expect(button.getAttribute('aria-label') ?? '').not.toMatch(FORBIDDEN_CONTROL)
      }
    }

    expect(calls.length).toBeGreaterThan(0)

    for (const call of calls) {
      expect(call.opts).toBeUndefined()
      expect(ALLOWED_PATHS).toContain(call.path.split('?')[0])
    }
  })
})

describe('formatTime', () => {
  it('formats epoch seconds and ISO strings, rejects junk', () => {
    expect(formatTime(0)).toBe(new Date(0).toLocaleString())
    expect(formatTime('2026-10-01T00:00:00+00:00')).toBe(new Date('2026-10-01T00:00:00+00:00').toLocaleString())
    expect(formatTime('not a date')).toBeNull()
    expect(formatTime(null)).toBeNull()
    expect(formatTime(Number.NaN)).toBeNull()
  })
})
