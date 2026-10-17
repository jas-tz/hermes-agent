/**
 * @hermes/plugin-sdk — THE plugin language. The vscode-module model: plugin
 * authors import exactly one module and get everything — they never touch
 * `@/…` internals (lint-fenced) and never need codebase access.
 *
 * Two delivery modes, one surface:
 *  - bundled (`src/plugins/<name>/`): the import resolves here via alias;
 *  - runtime-fetched (plugin host, next phase): the loader injects this same
 *    object as `window.__HERMES_PLUGIN_SDK__` and maps the import to it, so a
 *    published plugin builds against the types with the SDK marked external.
 *
 * Capability tiers (WoW-style):
 *  - `host.state.*` — READONLY app state (nanostore atoms; `.get()` or
 *    subscribe; `useValue` in React).
 *  - `host.*` actions — curated, safe verbs (toast, haptic).
 *  - `host.request` — the gateway JSON-RPC door; the plugin's real power,
 *    and the future seam for per-plugin capability grants.
 *  - `ui.*` — the design language, so plugin UI looks native by default.
 */

import { atom, computed, type ReadableAtom } from 'nanostores'

import { $activePresetId, $narrowViewport } from '@/components/pane-shell/tree/store'
import { onGatewayEvent } from '@/contrib/events'
import { getLogs, getStatus } from '@/hermes'
import { $gateway } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionAwaitingInput } from '@/store/prompts'
import { $activeSessionId, $awaitingResponse, $busy, $currentCwd, $currentModel, $gatewayState } from '@/store/session'
import { runGatewayRestart } from '@/store/system-actions'
import { $voiceInput, type VoiceInputStatus } from '@/store/voice-input'
import { $voiceLevel } from '@/store/voice-level'
import { $voicePlayback, type VoicePlaybackStatus } from '@/store/voice-playback'

// -- state: readonly views over the app's live atoms -------------------------

const readonlyAtom = <T>(atomLike: ReadableAtom<T>): ReadableAtom<T> => atomLike

/** Window geometry + the app's responsive posture, one readonly rect. */
export interface ViewportRect {
  width: number
  height: number
  /** Below the app's sidebar-collapse breakpoint (rails become overlays). */
  narrow: boolean
}

const readViewport = (): ViewportRect => ({
  width: typeof window === 'undefined' ? 0 : window.innerWidth,
  height: typeof window === 'undefined' ? 0 : window.innerHeight,
  narrow: $narrowViewport.get()
})

/** Spoken-reply playback phase exposed to plugins. */
export type VoiceStatus = VoicePlaybackStatus
/** Main-composer mic phase exposed to plugins. */
export type { VoiceInputStatus }

// Coarse DERIVED views, not the source atoms: a plugin sees a status value,
// never the playback audio element / message id, and even a cast-away write to
// a derived atom can't reach $voicePlayback / $voiceInput / $busy / the prompt
// stores — so it can't drive the app's own voice, turn or approval state.
// Boolean only: never the approval command, secret prompt or request id, and
// no way to answer — responding stays with the app's own approval UI.
const $awaitingInput = computed($activeSessionAwaitingInput, awaiting => awaiting)
const $agentBusy = computed([$busy, $awaitingResponse], (busy, awaiting) => busy || awaiting)
const $voiceStatus = computed($voicePlayback, playback => playback.status)
const $voiceInputStatus = computed($voiceInput, status => status)
const $voiceLevelView = computed($voiceLevel, level => level)

const $viewport = atom<ViewportRect>(readViewport())

if (typeof window !== 'undefined') {
  const refresh = () => $viewport.set(readViewport())
  window.addEventListener('resize', refresh)
  $narrowViewport.listen(refresh)
}

export const host = {
  state: {
    /** Runtime id of the active chat session (null on a fresh draft). */
    activeSessionId: readonlyAtom<null | string>($activeSessionId),
    /** Foreground chat is blocked on the user (approval, clarify, sudo or
     *  secret prompt). Read-only — responding stays in the app's own UI. */
    awaitingInput: readonlyAtom<boolean>($awaitingInput),
    /** Foreground chat has a turn in flight (sent, streaming, or running tools). */
    busy: readonlyAtom<boolean>($agentBusy),
    /** Active workspace cwd ('' when detached). */
    cwd: readonlyAtom<string>($currentCwd),
    /** Id of the layout preset last applied ('default', or a scoped
     *  `<plugin id>:<preset id>` for a plugin preset). */
    layoutPreset: readonlyAtom<string>($activePresetId),
    /** Gateway socket state: 'idle' | 'connecting' | 'open' | …. */
    gateway: readonlyAtom<string>($gatewayState),
    /** Current main model slug. */
    model: readonlyAtom<string>($currentModel),
    /** Profile the live gateway is routed to. */
    profile: readonlyAtom<string>($activeGatewayProfile),
    /** Window geometry ({ width, height, narrow }). */
    viewport: readonlyAtom<ViewportRect>($viewport),
    /** Spoken-reply playback: 'idle' | 'preparing' (synthesizing) | 'speaking'. */
    voice: readonlyAtom<VoiceStatus>($voiceStatus),
    /** Main-composer mic: 'idle' | 'listening' (voice conversation) |
     *  'dictating' (push-to-talk) | 'transcribing'. */
    voiceInput: readonlyAtom<VoiceInputStatus>($voiceInputStatus),
    /** Live 0–1 loudness of the active voice — reply playback while speaking,
     *  the mic while dictating or listening, else 0. Throttled (~15 Hz); a
     *  number only, never audio samples. */
    voiceLevel: readonlyAtom<number>($voiceLevelView)
  },

  /** Toast into the app's notification stack. */
  notify,
  notifyError,

  // NOTE: every host door is async-safe — wrapped so a sync throw from an
  // internal helper (e.g. no desktop bridge in a plain browser) becomes a
  // rejection a plugin's .catch() sees, never an error-boundary crash.

  /** Tail an app log file (`agent` / `errors` / `gateway` / `gui` / …). */
  logs: async (...args: Parameters<typeof getLogs>) => getLogs(...args),

  /** Navigate the app router (hash routes, e.g. '/command-center?section=system'). */
  navigate: (path: string) => {
    window.location.hash = path.startsWith('#') ? path : `#${path}`
  },

  /** HEAR the gateway stream (message deltas, session lifecycle, tool
   *  activity, …) by event type — `'*'` for everything. Returns a disposer.
   *  Listeners are isolated; a throw can't affect app dispatch. */
  onEvent: onGatewayEvent,

  /** Restart the backend gateway (progress surfaces in the core statusbar). */
  restartGateway: async () => runGatewayRestart(),

  /** One-shot system status snapshot (platforms, versions, …). */
  status: async () => getStatus(),

  /** Gateway JSON-RPC — sessions, config, skills, cron, kanban, everything
   *  the app itself uses. Lazy: resolves the LIVE socket per call. */
  request: async <T>(method: string, params: Record<string, unknown> = {}): Promise<T> => {
    const gateway = $gateway.get()

    if (!gateway) {
      throw new Error('Hermes gateway unavailable')
    }

    return gateway.request<T>(method, params)
  }
}

// -- react bridge -------------------------------------------------------------

// Every contribution surface, plugin-reachable: register keybinds, palette
// commands, routes, themes, panes, composer extensions, and bar items with
// the same area ids + payload types core uses.
export { COMPOSER_AREAS, type ComposerAttachmentProvider, type ComposerMiddleware } from '@/app/chat/composer/contrib'

// -- ui: the design language --------------------------------------------------

export { PALETTE_AREA, type PaletteContribution } from '@/app/command-palette/contrib'
export {
  type RouteContribution,
  ROUTES_AREA,
  SIDEBAR_FOOTER_AREA,
  SIDEBAR_NAV_AREA,
  type SidebarNavContribution
} from '@/app/routes'
/** THE model catalog menu — the same searchable, provider-grouped, family-
 *  collapsing picker the chat composer uses, including the per-row
 *  thinking/effort/fast submenu. Drive it with a `ModelMenuController`: the
 *  menu renders and navigates, your controller decides what a selection MEANS
 *  (write to a session, hold a per-task override, …). Never fork it — a copy
 *  drifts from the composer the first time either side changes. */
export {
  ModelCatalogMenu,
  type ModelChoice,
  ModelMenuCloseContext,
  type ModelMenuController
} from '@/app/shell/model-catalog-menu'
export type { StatusbarItem } from '@/app/shell/statusbar-controls'

export type { TitlebarTool } from '@/app/shell/titlebar-controls'
/** Layout tree node shapes for `LAYOUTS_AREA` presets. */
export type { GroupNode, LayoutNode, SplitNode } from '@/components/pane-shell/tree/model'
/** Apply a layout preset tree — the same path as picking it in the Layout
 *  picker (sizes reset, pane placement handed back to the app). Pass the
 *  preset's scoped id (`<plugin id>:<preset id>`) so `host.state.layoutPreset`
 *  and the picker show it as active. */
export { applyLayoutPreset } from '@/components/pane-shell/tree/presets'
/** Pane placement roles. `'floating'` is the one NON-tiling value: the pane is
 *  excluded from the layout tree and rendered as a fixed, draggable card above
 *  it — it takes no width from any zone, has no tab, and can't be docked.
 *  Pair it with `anchor` (spawn corner, default `'top-right'`) plus
 *  `width`/`height`. */
export type { FloatingAnchor } from '@/components/pane-shell/tree/renderer/floating-rect'
export { StatusDot, type StatusTone } from '@/components/status-dot'
export { Badge } from '@/components/ui/badge'
export { Button } from '@/components/ui/button'
export { Checkbox } from '@/components/ui/checkbox'
export { Codicon } from '@/components/ui/codicon'
export { ConfirmDialog } from '@/components/ui/confirm-dialog'
export {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
export { CopyButton } from '@/components/ui/copy-button'
export { DecodeText } from '@/components/ui/decode-text'
export {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog'
export {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
export { EmptyState } from '@/components/ui/empty-state'
export { ErrorState } from '@/components/ui/error-state'
export { FadeScroll } from '@/components/ui/fade-scroll'
export { GlyphSpinner } from '@/components/ui/glyph-spinner'
export { Input } from '@/components/ui/input'
export { Kbd, KbdGroup } from '@/components/ui/kbd'
/** The app's canonical loader (animated curves; `lemniscate-bloom` for long
 *  page loads) — the same one every core page uses. */
export { Loader, type LoaderType } from '@/components/ui/loader'
export { LogView } from '@/components/ui/log-view'
export { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
export { ScrollArea } from '@/components/ui/scroll-area'
export { SearchField } from '@/components/ui/search-field'
export { SegmentedControl } from '@/components/ui/segmented-control'
export { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
export { Separator } from '@/components/ui/separator'
export { Skeleton } from '@/components/ui/skeleton'
export { Switch } from '@/components/ui/switch'
export { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
export { Textarea } from '@/components/ui/textarea'
export { Tip, Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
export type { GatewayEventListener } from '@/contrib/events'
export type {
  HermesPlugin,
  PluginContext,
  PluginContribution,
  PluginNativeNotificationInput,
  PluginOs,
  PluginRestOptions,
  PluginStorage
} from '@/contrib/plugin'

// -- contracts ----------------------------------------------------------------

/** Mount-scoped contribution: while the rendering component is mounted, its
 *  children render in the target area's slot; unmount disposes it. Use for
 *  page-owned chrome (a page's titlebar control leaves with the page) —
 *  `ctx.register` stays the door for permanent contributions. Namespace the
 *  id with your plugin slug (`kanban:board-switcher`). */
export { Contribute, type ContributeProps } from '@/contrib/react/contribute'
/** Render another surface's contributions inline — e.g. a plugin pane hosting
 *  an area other plugins contribute to (`<Slot area="jarvis-core.stage" />`). */
export { Slot, type SlotProps } from '@/contrib/react/slot'
export type { Contribution } from '@/contrib/types'
/** Grab-to-pan for overflow containers (boards, timelines, wide tables) —
 *  the shared scrub primitive; don't hand-roll drag-to-scroll. */
export { type GrabScroll, useGrabScroll } from '@/hooks/use-grab-scroll'
/** Localized copy. `useI18n` reuses the app's strings; `usePluginI18n(id)` +
 *  `ctx.i18n.register` let a plugin ship its OWN locale bundles, scoped like
 *  `ctx.storage` and resolved against the app's active locale — no core edit. */
export {
  type Locale,
  type PluginI18n,
  type PluginLocaleBundles,
  type PluginMessages,
  type PluginMessageValue,
  type PluginTranslate,
  useI18n,
  usePluginI18n
} from '@/i18n'
/** THE compact-number formatter — every user-facing count/token figure goes
 *  through here (1230 → "1.2k", 1_500_000 → "1.5M"). Don't hand-roll `/1000`. */
export { compactNumber } from '@/lib/format'
export { triggerHaptic as haptic } from '@/lib/haptics'
/** The app's lucide icon set (RefreshCw, LayoutDashboard, Activity, …). */
export * as icons from '@/lib/icons'
export { type KeybindContribution, KEYBINDS_AREA } from '@/lib/keybinds/actions'
/** The app's deterministic identity color for a name (profiles, assignees,
 *  authors) + its translucent tag fill — so plugin-rendered identities read
 *  the same hue as everywhere else. */
export { profileColor, profileColorSoft } from '@/lib/profile-color'
/** The shared client itself, for invalidation OUTSIDE React (e.g. a
 *  `ctx.socket` frame invalidating a query). Inside components keep using
 *  `useQueryClient`. */
export { queryClient } from '@/lib/query-client'

export const PANES_AREA = 'panes'
/** Layout presets (the Layout editor's picker): `data` is a layout tree of
 *  `split`/`group` nodes whose `panes` are pane contribution ids. A plugin's
 *  own panes are referenced by their scoped id, `<plugin id>:<pane id>`. */
export const LAYOUTS_AREA = 'layouts'
/** Hermes' reasoning levels + their compact labels, so a plugin surfacing a
 *  thinking depth uses the same scale and spelling as the rest of the app. */
export {
  DEFAULT_REASONING_EFFORT,
  REASONING_EFFORT_VALUES,
  REASONING_EFFORTS,
  type ReasoningEffort,
  reasoningEffortLabel
} from '@/lib/reasoning-effort'
export const STATUSBAR_AREAS = { left: 'statusBar.left', right: 'statusBar.right' } as const
export const TITLEBAR_AREAS = { center: 'titleBar.center', left: 'titleBar.left', right: 'titleBar.right' } as const

/** The app's own gateway-readiness evaluation (setup.status +
 *  setup.runtime_check, reconciled) — pass `host.request`. Don't hand-roll
 *  readiness from raw RPC shapes. */
export { evaluateRuntimeReadiness, type RuntimeReadinessResult } from '@/lib/runtime-readiness'
export { coarseElapsed, fmtDateTime, fmtDayTime, relativeTime } from '@/lib/time'
export { cn } from '@/lib/utils'
export { THEMES_AREA } from '@/themes/user-themes'
export type { RpcEvent, StatusResponse } from '@/types/hermes'
/** Subscribe a component to a `host.state` atom. */
export { useStore as useValue } from '@nanostores/react'
/** The app's data-fetching layer. Plugins share the ONE QueryClient mounted at
 *  the app root, so their queries cache, dedupe, poll (`refetchInterval`), and
 *  invalidate exactly like core screens — no hand-rolled atoms or polls. */
export { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
/** Plugin-local reactive state (share between a trigger and its panel, poll
 *  loops, cross-component signals) — the same primitive `host.state` uses. */
export { atom, computed } from 'nanostores'
