/**
 * JARVIS Memory — a read-only desktop page over the bundled JARVIS dashboard
 * plugin API (`plugins/memory/jarvis/dashboard/plugin_api.py`), reached only
 * through `ctx.rest` (authenticated, profile-aware, scoped to
 * `/api/plugins/jarvis`). No new backend, no core edits, no direct Qdrant or
 * embedding-service access from the renderer.
 *
 * Ships OFF by default (`defaultEnabled: false`): it inventories in
 * Settings ▸ Plugins and registers nothing until the user flips the switch.
 */
import {
  type HermesPlugin,
  host,
  PALETTE_AREA,
  type PaletteContribution,
  type RouteContribution,
  ROUTES_AREA,
  SIDEBAR_FOOTER_AREA,
  SIDEBAR_NAV_AREA,
  type SidebarNavContribution
} from '@hermes/plugin-sdk'

import { bindApi } from './api'
import { MEMORY_CARD_LOCALES, MemoryCard } from './memory-card'
import { JarvisMemoryPage } from './page'
import { TasksBar } from './tasks-bar'

export const JARVIS_ROUTE = '/jarvis'
/** The JARVIS Core face pane's stage area (plugins/jarvis-core/core-pane.tsx). */
const JARVIS_STAGE_AREA = 'jarvis-core.stage'

const plugin: HermesPlugin = {
  id: 'jarvis',
  name: 'JARVIS Memory',
  description: 'Read-only JARVIS memory status, continuity, search, learning history and pending proposals.',
  defaultEnabled: false,
  register(ctx) {
    ctx.onDispose(bindApi(ctx.rest))
    ctx.i18n.register(MEMORY_CARD_LOCALES)

    ctx.registerMany([
      {
        id: 'page',
        area: ROUTES_AREA,
        data: { path: JARVIS_ROUTE } satisfies RouteContribution,
        render: () => <JarvisMemoryPage />
      },
      {
        id: 'nav',
        area: SIDEBAR_NAV_AREA,
        order: 55,
        data: { codicon: 'database', label: 'JARVIS Memory', path: JARVIS_ROUTE } satisfies SidebarNavContribution
      },
      {
        // Memory card at the foot of the sessions sidebar (JARVIS design).
        id: 'memory-card',
        area: SIDEBAR_FOOTER_AREA,
        render: () => <MemoryCard route={JARVIS_ROUTE} />
      },
      {
        // Active tasks bar under the JARVIS face's status cards.
        id: 'tasks-bar',
        area: JARVIS_STAGE_AREA,
        render: () => <TasksBar route={JARVIS_ROUTE} />
      },
      {
        id: 'open',
        area: PALETTE_AREA,
        data: {
          id: 'jarvis.open',
          label: 'JARVIS: Open memory',
          keywords: ['jarvis', 'memory', 'continuity', 'goals', 'learnings'],
          run: () => host.navigate(JARVIS_ROUTE)
        } satisfies PaletteContribution
      }
    ])
  }
}

export default plugin
