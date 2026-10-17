import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { Contribution } from '@/contrib/types'
// Static, imported once: the plugin's pane data + preset tree are plain data,
// and re-importing the plugin (and with it the whole SDK) inside every test
// after `vi.resetModules()` blew the per-test timeout under load.
import * as jarvis from '@/plugins/jarvis-core/plugin'

// The JARVIS Core plugin's pane + "JARVIS" preset run through the REAL layout
// engine: adoption (`watchContributedPanes`), preset application
// (`applyLayoutPreset`) and the track model (`fixedTrackSize`). Lives outside
// src/plugins because plugin files may only import the SDK.
//
// Core pane sizing below mirrors the real registrations in
// app/contrib/controller.tsx (sessions/workspace/files/review/terminal) — the
// numbers that decide fixed vs flex tracks.

const SIDEBAR = '237px'

const CORE_PANES: Array<Pick<Contribution, 'data' | 'id'>> = [
  { id: 'sessions', data: { placement: 'left', collapsible: true, width: SIDEBAR, minWidth: SIDEBAR, maxWidth: '24rem' } },
  { id: 'workspace', data: { placement: 'main', minWidth: '22vw', uncloseable: true } },
  { id: 'files', data: { placement: 'right', collapsible: true, width: SIDEBAR, minWidth: '10rem', maxWidth: '20rem' } },
  { id: 'review', data: { placement: 'right', collapsible: true, width: SIDEBAR, minWidth: '10rem', maxWidth: '20rem' } },
  { id: 'terminal', data: { placement: 'bottom', height: '20vh', maxHeight: '80vh', revealOnPreset: true } }
]

describe('JARVIS Core in the real layout engine', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    vi.resetModules()
  })

  async function setup() {
    const store = await import('@/components/pane-shell/tree/store')
    const model = await import('@/components/pane-shell/tree/model')
    const presets = await import('@/components/pane-shell/tree/presets')
    const tracks = await import('@/components/pane-shell/tree/renderer/track-model')
    const { registry } = await import('@/contrib/registry')

    for (const pane of CORE_PANES) {
      registry.register({ ...pane, area: 'panes', render: () => null })
    }

    // Default-shaped tree: sessions | chat | (review + files) over terminal.
    store.declareDefaultTree(
      model.split('row', [
        model.group(['sessions'], { id: 'grp-sessions' }),
        model.group(['workspace'], { id: 'grp-main' }),
        model.split('column', [
          model.split('row', [model.group(['review'], { id: 'grp-review' }), model.group(['files'], { id: 'grp-files' })]),
          model.group(['terminal'], { id: 'grp-terminal' })
        ])
      ])
    )

    const registerJarvis = () =>
      registry.register({
        id: jarvis.JARVIS_CORE_TREE_ID,
        area: 'panes',
        source: 'plugin:jarvis-core',
        title: 'JARVIS Core',
        data: jarvis.JARVIS_CORE_PANE_DATA,
        render: () => null
      })

    // Mirrors the renderer's rule (tree-split.tsx `paneGone`): unregistered
    // panes are absent from the layout.
    const ctx = {
      overrides: {},
      paneFor: (id: string) => registry.getArea('panes').find(c => c.id === id),
      paneGone: (id: string) => !registry.getArea('panes').some(c => c.id === id)
    }

    const groupOf = (paneId: string) => model.findGroupOfPane(store.$layoutTree.get()!, paneId) ?? undefined

    return { ctx, groupOf, model, presets, registerJarvis, store, tracks }
  }

  it('enabling it in the normal layout adds a background tab, not a half-window zone', async () => {
    const { ctx, groupOf, model, registerJarvis, store, tracks } = await setup()
    store.watchContributedPanes()

    const before = model.groupLeafIds(store.$layoutTree.get()!)
    registerJarvis()

    // Stacked into the chat's own zone, chat still in front, no new zone.
    const main = groupOf('workspace')!
    expect(groupOf(jarvis.JARVIS_CORE_TREE_ID)?.id).toBe(main.id)
    expect(main.active).toBe('workspace')
    expect(model.groupLeafIds(store.$layoutTree.get()!)).toEqual(before)

    // Every zone keeps its track: chat flexes, sidebars stay fixed.
    expect(tracks.fixedTrackSize(main, 'row', ctx)).toBeNull()
    expect(tracks.fixedTrackSize(groupOf('sessions')!, 'row', ctx)).toBe(SIDEBAR)
    expect(tracks.fixedTrackSize(groupOf('files')!, 'row', ctx)).toBe(SIDEBAR)
  })

  it('the JARVIS preset puts the face centre on a flexible track beside a visible chat', async () => {
    const { ctx, groupOf, presets, registerJarvis, store, tracks } = await setup()
    const openTerminal = vi.fn()
    store.registerPaneOpener('terminal', openTerminal)
    store.watchContributedPanes()
    registerJarvis()

    presets.applyLayoutPreset('jarvis-core:layout', jarvis.JARVIS_LAYOUT_TREE)

    const face = groupOf(jarvis.JARVIS_CORE_TREE_ID)!
    const chat = groupOf('workspace')!

    expect(face.panes).toEqual([jarvis.JARVIS_CORE_TREE_ID])
    expect(tracks.fixedTrackSize(face, 'row', ctx)).toBeNull()
    expect(chat.id).not.toBe(face.id)
    expect(chat.active).toBe('workspace')
    expect(tracks.fixedTrackSize(chat, 'row', ctx)).toBeNull()
    expect(tracks.fixedTrackSize(groupOf('sessions')!, 'row', ctx)).toBe(SIDEBAR)

    // Nothing is lost, and the terminal is carried over without being forced open.
    expect(groupOf('terminal')).toBeDefined()
    expect(openTerminal).not.toHaveBeenCalled()
  })

  it('the Phase 1 pane id left in a saved layout is absent, so its old zone cannot balloon', async () => {
    const { ctx, model, registerJarvis, tracks } = await setup()
    registerJarvis()

    const stale = model.group(['jarvis-core:pane'], { id: 'grp-stale' })

    expect(ctx.paneGone('jarvis-core:pane')).toBe(true)
    expect(tracks.subtreeGone(stale, ctx)).toBe(true)
  })
})
