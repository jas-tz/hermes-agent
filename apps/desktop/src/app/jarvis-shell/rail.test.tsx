import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as TreeStore from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { $sidebarOpen, setSidebarOpen } from '@/store/layout'
import { $selectedStoredSessionId } from '@/store/session'

import { JARVIS_LAYOUT_PRESET, JarvisRail } from './rail'

const mocks = vi.hoisted(() => ({
  applyLayoutPreset: vi.fn(),
  togglePaneVisible: vi.fn()
}))

vi.mock('@/components/pane-shell/tree/presets', () => ({ applyLayoutPreset: mocks.applyLayoutPreset }))

vi.mock('@/components/pane-shell/tree/store', async importOriginal => ({
  ...(await importOriginal<typeof TreeStore>()),
  togglePaneVisible: mocks.togglePaneVisible
}))

function Where() {
  return <p data-testid="where">{useLocation().pathname}</p>
}

function renderRail(path = '/') {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route
          element={
            <>
              <JarvisRail />
              <Where />
            </>
          }
          path="*"
        />
      </Routes>
    </MemoryRouter>
  )
}

const labels = () => [...screen.getAllByTestId(/^jarvis-rail-/)].map(b => b.getAttribute('aria-label'))
const where = () => screen.getByTestId('where').textContent

beforeEach(() => {
  mocks.applyLayoutPreset.mockClear()
  mocks.togglePaneVisible.mockClear()
  setSidebarOpen(true)
  $selectedStoredSessionId.set(null)
})

afterEach(cleanup)

describe('JarvisRail', () => {
  it('shows the design rail, with Settings last', () => {
    renderRail()

    expect(labels()).toEqual([
      'Home',
      'Sessions',
      'New session',
      'Spawn tree',
      'Capabilities',
      'Messaging',
      'Artifacts',
      'Terminal',
      'Settings'
    ])
  })

  it('opens the same pages the command palette opens', () => {
    renderRail()

    fireEvent.click(screen.getByTestId('jarvis-rail-tools'))
    expect(where()).toBe('/skills')
    fireEvent.click(screen.getByTestId('jarvis-rail-messaging'))
    expect(where()).toBe('/messaging')
    fireEvent.click(screen.getByTestId('jarvis-rail-artifacts'))
    expect(where()).toBe('/artifacts')
    fireEvent.click(screen.getByTestId('jarvis-rail-agents'))
    expect(where()).toBe('/agents')
    fireEvent.click(screen.getByTestId('jarvis-rail-settings'))
    expect(where()).toBe('/settings')
    fireEvent.click(screen.getByTestId('jarvis-rail-new-session'))
    expect(where()).toBe('/')
  })

  it('marks the current page', () => {
    renderRail('/messaging')

    expect(screen.getByTestId('jarvis-rail-messaging').getAttribute('aria-current')).toBe('page')
    expect(screen.getByTestId('jarvis-rail-tools').getAttribute('aria-current')).toBeNull()
  })

  it('Home applies the JARVIS layout and returns to the current chat', () => {
    const dispose = registry.registerMany([
      { area: 'layouts', data: { type: 'group', id: 'g', panes: [], active: '' }, id: JARVIS_LAYOUT_PRESET }
    ])

    $selectedStoredSessionId.set('abc')
    renderRail('/skills')
    fireEvent.click(screen.getByTestId('jarvis-rail-home'))

    expect(mocks.applyLayoutPreset).toHaveBeenCalledWith(JARVIS_LAYOUT_PRESET, expect.objectContaining({ id: 'g' }))
    expect(where()).toBe('/abc')
    dispose()
  })

  it('toggles the sessions list and the terminal with the existing actions', () => {
    renderRail()

    fireEvent.click(screen.getByTestId('jarvis-rail-sessions'))
    expect($sidebarOpen.get()).toBe(false)

    fireEvent.click(screen.getByTestId('jarvis-rail-terminal'))
    expect(mocks.togglePaneVisible).toHaveBeenCalledWith('terminal')
  })

  it('adds Memory and Tasks only when their plugins are enabled', () => {
    const dispose = registry.registerMany([
      { area: 'sidebar.nav', data: { label: 'JARVIS Memory', path: '/jarvis' }, id: 'jarvis:nav' },
      { area: 'sidebar.nav', data: { label: 'Kanban', path: '/kanban' }, id: 'kanban:nav' },
      { area: 'sidebar.nav', data: { codicon: 'beaker', label: 'Lab', path: '/lab' }, id: 'lab:nav' }
    ])

    renderRail()

    expect(labels()).toEqual(expect.arrayContaining(['Memory', 'Tasks', 'Lab']))
    fireEvent.click(screen.getByTestId('jarvis-rail-memory'))
    expect(where()).toBe('/jarvis')
    fireEvent.click(screen.getByTestId('jarvis-rail-tasks'))
    expect(where()).toBe('/kanban')
    dispose()
  })
})
