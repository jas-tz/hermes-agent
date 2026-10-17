/**
 * JARVIS icon rail — the design's 64px left rail
 * (docs/jarvis-design/references/01-main-screen.jpg). Replaces the classic
 * sidebar's top menu (New session, Capabilities, Messaging, Artifacts and
 * plugin rows such as JARVIS Memory) while the JARVIS theme is active.
 *
 * Every button runs something the app already has — a route the command
 * palette also opens, or the same store action as the keybind — so nothing is
 * reimplemented here and every destination stays reachable from Ctrl K too:
 *   Home      → the JARVIS layout (sessions | face | chat) + the current chat
 *   Sessions  → show/hide the sessions list (view.toggleSidebar)
 *   New       → a fresh session (the palette's "New chat")
 *   Memory    → the JARVIS Memory page (plugin route, when enabled)
 *   Agents    → the agents view
 *   Tasks     → the Kanban board (plugin route, when enabled)
 *   Tools     → Capabilities (skills & tools)
 *   Messaging, Artifacts → their pages
 *   Terminal  → show/hide the terminal (view.showTerminal)
 *   Settings  → Settings
 * Other plugin sidebar rows follow the built-ins, with their own codicon.
 */
import './jarvis-shell.css'

import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'
import { useLocation, useNavigate } from 'react-router'

import type { LayoutNode } from '@/components/pane-shell/tree/model'
import { applyLayoutPreset } from '@/components/pane-shell/tree/presets'
import { $activePresetId, $paneVisible, togglePaneVisible } from '@/components/pane-shell/tree/store'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip, TipKeybindLabel } from '@/components/ui/tooltip'
import { useContributions } from '@/contrib/react/use-contributions'
import { registry } from '@/contrib/registry'
import { useI18n } from '@/i18n'
import { $sidebarOpen, toggleSidebarOpen } from '@/store/layout'
import { $newChatProfile } from '@/store/profile'
import { $selectedStoredSessionId } from '@/store/session'

import {
  AGENTS_ROUTE,
  appViewForPath,
  ARTIFACTS_ROUTE,
  MESSAGING_ROUTE,
  NEW_CHAT_ROUTE,
  sessionRoute,
  SETTINGS_ROUTE,
  SIDEBAR_NAV_AREA,
  type SidebarNavContribution,
  SKILLS_ROUTE
} from '../routes'

/** The JARVIS layout preset (plugins/jarvis-core) Home returns to. */
export const JARVIS_LAYOUT_PRESET = 'jarvis-core:layout'
/** Plugin sidebar rows the rail gives a dedicated slot. */
const MEMORY_NAV = 'jarvis:nav'
const TASKS_NAV = 'kanban:nav'

const PATHS: Record<string, ReactNode> = {
  agents: (
    <>
      <circle cx="12" cy="5" r="2.5" />
      <circle cx="5" cy="19" r="2.5" />
      <circle cx="19" cy="19" r="2.5" />
      <path d="M12 7.5v4M12 11.5l-5.5 5.5M12 11.5l5.5 5.5" />
    </>
  ),
  artifacts: (
    <>
      <path d="M7 3h7l5 5v13H7z" />
      <path d="M14 3v5h5M4 7v14h11" />
    </>
  ),
  home: <path d="M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z" />,
  memory: (
    <>
      <ellipse cx="12" cy="5" rx="8" ry="3" />
      <path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5" />
      <path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" />
    </>
  ),
  messaging: (
    <>
      <path d="M3 5h12v8H7l-4 3z" />
      <path d="M15 9h6v8l-3-2h-7v-2" />
    </>
  ),
  new: <path d="M12 5v14M5 12h14" />,
  sessions: <path d="M4 5h16v11H8l-4 4z" />,
  settings: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1" />
    </>
  ),
  tasks: (
    <>
      <path d="M9 6h11M9 12h11M9 18h11" />
      <path d="M3 6l1.5 1.5L7 5M3 12l1.5 1.5L7 11M3 18l1.5 1.5L7 17" />
    </>
  ),
  terminal: (
    <>
      <rect height="16" rx="2" width="18" x="3" y="4" />
      <path d="M7 9l3 3-3 3M13 15h4" />
    </>
  ),
  tools: <path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.6 2.6-2.4-.6-.6-2.4z" />
}

function RailIcon({ name }: { name: string }) {
  return (
    <svg aria-hidden="true" className="jarvis-rail__icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      {PATHS[name]}
    </svg>
  )
}

interface RailItem {
  id: string
  label: string
  icon: ReactNode
  active?: boolean
  actionId?: string
  onSelect: () => void
}

function RailButton({ item }: { item: RailItem }) {
  return (
    <Tip
      label={item.actionId ? <TipKeybindLabel actionId={item.actionId} text={item.label} /> : item.label}
      side="right"
    >
      <Button
        aria-current={item.active ? 'page' : undefined}
        aria-label={item.label}
        className="jarvis-rail__button"
        data-testid={`jarvis-rail-${item.id}`}
        onClick={item.onSelect}
        size="icon-lg"
        type="button"
        variant="ghost"
      >
        {item.icon}
      </Button>
    </Tip>
  )
}

function navRoute(c: { data?: unknown } | undefined): string | undefined {
  const path = (c?.data as Partial<SidebarNavContribution> | undefined)?.path

  return path?.startsWith('/') ? path : undefined
}

export function JarvisRail() {
  const { t } = useI18n()
  const text = t.jarvisShell.rail
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const view = appViewForPath(pathname)
  const sidebarOpen = useStore($sidebarOpen)
  const preset = useStore($activePresetId)
  const storedSessionId = useStore($selectedStoredSessionId)
  const terminalVisible = useStore($paneVisible('terminal'))
  const navRows = useContributions(SIDEBAR_NAV_AREA)
  const memory = navRows.find(c => c.id === MEMORY_NAV)
  const tasks = navRows.find(c => c.id === TASKS_NAV)
  const others = navRows.filter(c => c.id !== MEMORY_NAV && c.id !== TASKS_NAV && navRoute(c))

  const go = (route: string) => () => navigate(route)
  const at = (route: string | undefined) => Boolean(route) && pathname === route

  const goHome = () => {
    const tree = registry.getArea('layouts').find(c => c.id === JARVIS_LAYOUT_PRESET)?.data as LayoutNode | undefined

    if (tree) {
      applyLayoutPreset(JARVIS_LAYOUT_PRESET, tree)
    }

    if (view !== 'chat') {
      navigate(storedSessionId ? sessionRoute(storedSessionId) : NEW_CHAT_ROUTE)
    }
  }

  const top: RailItem[] = [
    {
      active: view === 'chat' && preset === JARVIS_LAYOUT_PRESET,
      icon: <RailIcon name="home" />,
      id: 'home',
      label: text.home,
      onSelect: goHome
    },
    {
      active: sidebarOpen,
      actionId: 'view.toggleSidebar',
      icon: <RailIcon name="sessions" />,
      id: 'sessions',
      label: text.sessions,
      onSelect: toggleSidebarOpen
    },
    {
      actionId: 'session.new',
      icon: <RailIcon name="new" />,
      id: 'new-session',
      label: t.sidebar.nav['new-session'] ?? t.jarvisShell.newSession,
      onSelect: () => {
        // Same as the sidebar row: a plain new session lands in the live profile.
        $newChatProfile.set(null)
        navigate(NEW_CHAT_ROUTE)
      }
    },
    ...(navRoute(memory)
      ? [
          {
            active: at(navRoute(memory)),
            icon: <RailIcon name="memory" />,
            id: 'memory',
            label: text.memory,
            onSelect: go(navRoute(memory)!)
          }
        ]
      : []),
    {
      active: at(AGENTS_ROUTE),
      actionId: 'nav.agents',
      icon: <RailIcon name="agents" />,
      id: 'agents',
      label: t.agents.title,
      onSelect: go(AGENTS_ROUTE)
    },
    ...(navRoute(tasks)
      ? [
          {
            active: at(navRoute(tasks)),
            icon: <RailIcon name="tasks" />,
            id: 'tasks',
            label: text.tasks,
            onSelect: go(navRoute(tasks)!)
          }
        ]
      : []),
    {
      active: view === 'skills',
      actionId: 'nav.skills',
      icon: <RailIcon name="tools" />,
      id: 'tools',
      label: t.sidebar.nav.skills ?? SKILLS_ROUTE,
      onSelect: go(SKILLS_ROUTE)
    },
    {
      active: view === 'messaging',
      actionId: 'nav.messaging',
      icon: <RailIcon name="messaging" />,
      id: 'messaging',
      label: t.sidebar.nav.messaging ?? MESSAGING_ROUTE,
      onSelect: go(MESSAGING_ROUTE)
    },
    {
      active: view === 'artifacts',
      actionId: 'nav.artifacts',
      icon: <RailIcon name="artifacts" />,
      id: 'artifacts',
      label: t.sidebar.nav.artifacts ?? ARTIFACTS_ROUTE,
      onSelect: go(ARTIFACTS_ROUTE)
    },
    ...others.map(c => {
      const data = c.data as Partial<SidebarNavContribution>

      return {
        active: at(navRoute(c)),
        icon: <Codicon className="jarvis-rail__icon" name={data.codicon || 'plug'} />,
        id: c.id,
        label: data.label ?? c.id,
        onSelect: go(navRoute(c)!)
      }
    }),
    {
      active: terminalVisible,
      actionId: 'view.showTerminal',
      icon: <RailIcon name="terminal" />,
      id: 'terminal',
      label: text.terminal,
      onSelect: () => togglePaneVisible('terminal')
    }
  ]

  const settings: RailItem = {
    active: view === 'settings',
    actionId: 'nav.settings',
    icon: <RailIcon name="settings" />,
    id: 'settings',
    label: text.settings,
    onSelect: go(SETTINGS_ROUTE)
  }

  return (
    <nav aria-label={text.label} className="jarvis-rail" data-testid="jarvis-rail">
      {top.map(item => (
        <RailButton item={item} key={item.id} />
      ))}
      <div className="jarvis-rail__spacer" />
      <RailButton item={settings} />
    </nav>
  )
}
