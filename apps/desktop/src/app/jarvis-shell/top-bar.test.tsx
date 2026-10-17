import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import type * as Hermes from '@/hermes'
import type * as CommandPalette from '@/store/command-palette'
import { $activeSessionId, $gatewayState, $selectedStoredSessionId, $sessions } from '@/store/session'

import { formatClock, timeZoneCity } from './clock'
import { $linkLatencyMs } from './link-latency'
import { JarvisTopBar, linkStateOf } from './top-bar'
import { voiceProviderLabel } from './voice-provider'

const mocks = vi.hoisted(() => ({
  getHermesConfig: vi.fn(async () => ({ tts: { provider: 'piper' } })),
  getStatus: vi.fn(async () => ({})),
  openCommandPalette: vi.fn()
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof Hermes>()),
  getHermesConfig: mocks.getHermesConfig,
  getStatus: mocks.getStatus
}))

vi.mock('@/store/command-palette', async importOriginal => ({
  ...(await importOriginal<typeof CommandPalette>()),
  openCommandPalette: mocks.openCommandPalette
}))

function renderBar() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <JarvisTopBar />
    </QueryClientProvider>
  )
}

beforeEach(() => {
  $gatewayState.set('open')
  $sessions.set([])
  $selectedStoredSessionId.set(null)
  $activeSessionId.set(null)
  mocks.openCommandPalette.mockClear()
})

afterEach(cleanup)

describe('formatClock', () => {
  const at = new Date('2026-10-16T02:46:00+03:00')

  it('reads like the design: time · weekday day month · city', () => {
    expect(formatClock(at, 'en-GB', 'Africa/Dar_es_Salaam')).toBe('02:46 · FRI 16 OCT · DAR ES SALAAM')
  })

  it('uses a 24-hour clock in the given zone', () => {
    expect(formatClock(new Date('2026-10-16T15:05:00Z'), 'en-GB', 'Africa/Dar_es_Salaam')).toMatch(/^18:05 · /)
  })

  it('omits the city for offset-only zones', () => {
    expect(timeZoneCity('UTC')).toBeNull()
    expect(formatClock(at, 'en-GB', 'UTC')).toBe('23:46 · THU 15 OCT')
  })

  it('uses day-before-month for English, as in the design', () => {
    expect(formatClock(at, 'en', 'Africa/Dar_es_Salaam')).toBe('02:46 · FRI 16 OCT · DAR ES SALAAM')
  })

  it('formats in the app language', () => {
    expect(formatClock(at, 'ja', 'Asia/Tokyo')).toContain('TOKYO')
    expect(formatClock(at, 'ja', 'Asia/Tokyo')).toMatch(/^08:46 · /)
  })
})

describe('top bar helpers', () => {
  it('maps the gateway state onto the Link pill', () => {
    expect(linkStateOf('open')).toBe('online')
    expect(linkStateOf('closed')).toBe('offline')
    expect(linkStateOf('error')).toBe('offline')
    expect(linkStateOf('connecting')).toBe('connecting')
    expect(linkStateOf('idle')).toBe('connecting')
  })

  it('labels the configured TTS provider, or nothing', () => {
    expect(voiceProviderLabel({ tts: { provider: 'piper' } })).toBe('Piper')
    expect(voiceProviderLabel({ tts: { provider: ' ' } })).toBeNull()
    expect(voiceProviderLabel({})).toBeNull()
    expect(voiceProviderLabel(null)).toBeNull()
  })
})

describe('JarvisTopBar', () => {
  it('shows the wordmark and the active session title', () => {
    $sessions.set([{ id: 's1', title: 'Friendly greeting #7' } as never])
    $selectedStoredSessionId.set('s1')
    renderBar()

    expect(screen.getByText('J.A.R.V.I.S')).toBeTruthy()
    expect(screen.getByTestId('jarvis-topbar-title').textContent).toBe('Friendly greeting #7')
  })

  it('falls back to "New session" with no active session', () => {
    renderBar()

    expect(screen.getByTestId('jarvis-topbar-title').textContent).toBe('New session')
  })

  it('shows the link state and the measured round trip', () => {
    renderBar()
    act(() => $linkLatencyMs.set(38))

    expect(screen.getByTestId('jarvis-topbar-link').textContent).toBe('Link online · 38 ms')

    act(() => $gatewayState.set('closed'))
    expect(screen.getByTestId('jarvis-topbar-link').textContent).toBe('Link offline')
    expect(screen.getByTestId('jarvis-topbar-link').getAttribute('data-link')).toBe('offline')
  })

  it('shows the configured voice provider', async () => {
    renderBar()

    await waitFor(() => expect(screen.getByTestId('jarvis-topbar-voice').textContent).toBe('Voice: Piper'))
  })

  it('opens the existing command palette', () => {
    renderBar()
    fireEvent.click(screen.getByTestId('jarvis-topbar-command'))

    expect(mocks.openCommandPalette).toHaveBeenCalledTimes(1)
    expect(screen.getByTestId('jarvis-topbar-command').textContent).toContain('Ask or command')
  })

  it('shows a live clock', () => {
    renderBar()

    expect(screen.getByTestId('jarvis-topbar-clock').textContent).toMatch(/^\d\d:\d\d · /)
  })

  it('keeps plugin titlebar items but leaves out the JARVIS layout toggle', () => {
    const dispose = registry.registerMany([
      {
        area: 'titleBar.center',
        id: 'jarvis-core:layout-toggle',
        render: () => <span>JARVIS-TOGGLE</span>,
        source: 'plugin:jarvis-core'
      },
      {
        area: 'titleBar.center',
        id: 'kanban:board-switcher',
        render: () => <span>BOARD</span>,
        source: 'plugin:kanban'
      }
    ])

    renderBar()

    expect(screen.queryByText('JARVIS-TOGGLE')).toBeNull()
    expect(screen.getByText('BOARD')).toBeTruthy()
    dispose()
  })
})
