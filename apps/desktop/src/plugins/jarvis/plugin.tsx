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
  SIDEBAR_NAV_AREA,
  type SidebarNavContribution
} from '@hermes/plugin-sdk'

import { bindApi } from './api'
import { JarvisMemoryPage } from './page'

export const JARVIS_ROUTE = '/jarvis'

const plugin: HermesPlugin = {
  id: 'jarvis',
  name: 'JARVIS Memory',
  description: 'Read-only JARVIS memory status, continuity, search, learning history and pending proposals.',
  defaultEnabled: false,
  register(ctx) {
    ctx.onDispose(bindApi(ctx.rest))

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
