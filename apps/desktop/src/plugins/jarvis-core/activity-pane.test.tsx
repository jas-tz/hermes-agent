import type * as Sdk from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ACTIVITY_LINES, ActivityPane } from './activity-pane'

const mocks = vi.hoisted(() => ({ logs: vi.fn() }))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const actual = await importOriginal<typeof Sdk>()

  return { ...actual, host: { ...actual.host, logs: mocks.logs } }
})

function renderPane() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <ActivityPane />
    </QueryClientProvider>
  )
}

afterEach(() => {
  cleanup()
  mocks.logs.mockReset()
})

describe('ActivityPane', () => {
  it('shows the live log tail through the read-only host.logs', async () => {
    mocks.logs.mockResolvedValue({ lines: ['tool call: terminal', 'model request ok'] })
    renderPane()

    await waitFor(() =>
      expect(screen.getByTestId('jarvis-activity').textContent).toContain('tool call: terminal\nmodel request ok')
    )
    expect(mocks.logs).toHaveBeenCalledWith({ lines: ACTIVITY_LINES })
  })

  it('says when there is nothing yet, or the log is unavailable', async () => {
    mocks.logs.mockResolvedValue({ lines: [] })
    renderPane()
    await waitFor(() => expect(screen.getByTestId('jarvis-activity').textContent).toBe('activity.empty'))

    cleanup()
    mocks.logs.mockRejectedValue(new Error('offline'))
    renderPane()
    await waitFor(() => expect(screen.getByTestId('jarvis-activity').textContent).toBe('activity.error'))
  })
})
