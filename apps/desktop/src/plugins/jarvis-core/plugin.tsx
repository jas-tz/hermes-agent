/**
 * JARVIS Core — the animated status face, plus a dedicated "JARVIS" layout
 * that puts it centre stage: sessions left, face centre, the REAL chat right.
 *
 * A separate plugin from JARVIS Memory on purpose: closing a plugin pane
 * disables its plugin (pane-shell `closeTreePane`), so folding this pane into
 * `jarvis` would switch the memory page off with it.
 *
 * Ships ON (`defaultEnabled: true`) — this desktop IS JARVIS — and can be
 * switched off in Settings ▸ Plugins. Enabling only adds a background tab in
 * the chat zone (see JARVIS_CORE_PANE_DATA), so no existing layout moves; the
 * titlebar "JARVIS" entry switches to the dedicated layout in one click. It
 * reads only coarse `host.state` signals (turn, prompt, voice, mic, voice
 * level, gateway, active layout) — no REST, no gateway requests, and the face
 * itself has no controls.
 */
import './jarvis-core.css'

import {
  type GroupNode,
  type HermesPlugin,
  host,
  type LayoutNode,
  LAYOUTS_AREA,
  PANES_AREA,
  TITLEBAR_AREAS
} from '@hermes/plugin-sdk'

import { ActivityPane } from './activity-pane'
import { JarvisCorePane } from './core-pane'
import { JARVIS_CORE_LOCALES } from './i18n'
import { JarvisLayoutToggle } from './layout-toggle'
import { trackResponseTimes } from './response-times'

/** Pane id within this plugin. Was `pane` (a fixed 17rem right zone) until the
 *  face went flexible: a flexible pane under the OLD id would inherit its saved
 *  even split and jump to half the window. Under a new id the old entry is just
 *  an unregistered id the layout skips, and this one is adopted fresh. */
export const JARVIS_CORE_PANE_ID = 'core'

/** Scoped id the layout tree knows this pane by (`<plugin id>:<pane id>`). */
export const JARVIS_CORE_TREE_ID = `jarvis-core:${JARVIS_CORE_PANE_ID}`

/** Scoped id of the JARVIS layout preset — what `host.state.layoutPreset`
 *  reports while it is active. */
export const JARVIS_LAYOUT_PRESET_ID = 'jarvis-core:layout'

/** The Activity tab (agent log tail) beside the chat — the design's
 *  Conversation / Activity / Files tabs. */
export const ACTIVITY_TREE_ID = 'jarvis-core:activity'

/**
 * No `width` — so the face can fill the JARVIS layout's centre zone, where it
 * sits alone and its zone is a flexible track. It also declares no `maxWidth`:
 * a cap on a lone pane would cap the face too.
 *
 * Outside that layout it docks as a TAB in the chat's own zone (silently — the
 * chat stays the active tab). That zone always flexes because it holds the
 * workspace, and a flex zone ignores tenant caps, so enabling JARVIS Core
 * changes no existing zone's width. The alternatives fail: a lone flexible
 * zone beside the chat adopts at an even split (half the window), and a tab in
 * the fixed Files rail would uncap that rail (a zone's cap applies only if
 * EVERY tenant declares one).
 *
 * `minWidth` stays below the workspace's own floor at typical window sizes so
 * stacking into the chat zone doesn't raise the chat's minimum.
 *
 * Not `collapsible`: below the narrow-viewport breakpoint (768px — a half-width
 * window at 130% zoom) collapsible panes leave the grid for a hidden edge
 * overlay, which would make the face vanish. It stays in the grid instead,
 * beside the chat, and shrinks with its zone.
 */
export const JARVIS_CORE_PANE_DATA = {
  placement: 'right',
  dock: { pane: 'workspace', pos: 'center' },
  minWidth: '12rem'
} as const

const group = (id: string, panes: string[]): GroupNode => ({ type: 'group', id, panes, active: panes[0] ?? '' })

/**
 * Sessions | JARVIS face | chat. The chat zone keeps the workspace ACTIVE so
 * the transcript, composer (mic, voice conversation, dictation) and inline
 * approval / clarify rows stay on screen beside the face; files and review
 * ride along as tabs there, as in the Focus layout. The terminal is left out
 * on purpose: it opts into `revealOnPreset`, so listing it would force it
 * open — omitted, it carries over from the previous layout untouched.
 *
 * Weights 1 : 6 : 4 make the face zone wide enough for its five status
 * cards in one row on a full-screen 1920×1080 window even at 130% zoom (about
 * 708px of a 1481px viewport, beside the 64px JARVIS rail), while the chat keeps
 * room for the full composer (mic / dictation / read-aloud / wake-word / voice
 * controls): about 470px there, and never below the workspace's own 22vw floor
 * on small windows.
 */
export const JARVIS_LAYOUT_TREE: LayoutNode = {
  type: 'split',
  id: 'jarvis-spl-root',
  orientation: 'row',
  children: [
    group('jarvis-grp-sessions', ['sessions']),
    group('jarvis-grp-face', [JARVIS_CORE_TREE_ID]),
    group('jarvis-grp-chat', ['workspace', ACTIVITY_TREE_ID, 'files', 'review'])
  ],
  weights: [1, 6, 4]
}

const plugin: HermesPlugin = {
  id: 'jarvis-core',
  name: 'JARVIS Core',
  description:
    'Animated JARVIS face and a JARVIS layout (sessions, face, chat): standing by, listening, dictating, transcribing, thinking, needs you, preparing voice, speaking, offline.',
  defaultEnabled: true,
  register(ctx) {
    // Face + toggle copy in en / ja / zh / zh-hant, scoped to this plugin.
    ctx.i18n.register(JARVIS_CORE_LOCALES)
    // Response-time card: turn durations from the read-only busy signal.
    ctx.onDispose(trackResponseTimes(host.state.busy))
    ctx.registerMany([
      {
        id: JARVIS_CORE_PANE_ID,
        area: PANES_AREA,
        title: 'JARVIS Core',
        data: JARVIS_CORE_PANE_DATA,
        render: () => <JarvisCorePane />
      },
      {
        id: 'activity',
        area: PANES_AREA,
        title: ctx.i18n.t('activity.title'),
        // A tab in the chat's own zone, like the face outside the JARVIS layout.
        data: { placement: 'right', dock: { pane: 'workspace', pos: 'center' }, minWidth: '12rem' },
        render: () => <ActivityPane />
      },
      {
        id: 'layout',
        area: LAYOUTS_AREA,
        title: 'JARVIS',
        order: 40,
        data: JARVIS_LAYOUT_TREE
      },
      {
        id: 'layout-toggle',
        // Centre slot: the right slot is offset for a fixed count of app tools
        // and a labelled button there collides with the Layout picker.
        area: TITLEBAR_AREAS.center,
        title: 'JARVIS layout',
        render: () => <JarvisLayoutToggle presetId={JARVIS_LAYOUT_PRESET_ID} tree={JARVIS_LAYOUT_TREE} />
      }
    ])
  }
}

export default plugin
