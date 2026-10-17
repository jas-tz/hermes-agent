/**
 * JARVIS top bar (docs/jarvis-design/references/01-main-screen.jpg): the
 * J.A.R.V.I.S wordmark and the active session's title on the left, a live
 * clock/date line in the centre, and the Link pill, Voice pill and "Ask or
 * command" palette button on the right. Replaces the classic Hermes title bar
 * while the JARVIS theme is active (see use-jarvis-shell.ts).
 *
 * Everything it shows is read from state the app already owns: the session
 * list, the gateway link, the backend round trip (link-latency.ts), the
 * configured TTS provider, and the command palette's own open action and
 * keybind. The bar is the window's drag region; only its controls are
 * no-drag. The right edge clears the native window-controls overlay through
 * the same --titlebar-tools-right the classic bar uses.
 */
import './jarvis-shell.css'
import './jarvis-chat.css'

import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import { useContributions } from '@/contrib/react/use-contributions'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { useKeybindHint } from '@/lib/keybinds/use-keybind-hint'
import { openCommandPalette } from '@/store/command-palette'
import { $activeSessionId, $gatewayState, $selectedStoredSessionId, $sessions } from '@/store/session'

import { formatClock, useMinuteClock } from './clock'
import { $linkLatencyMs } from './link-latency'
import { useVoiceProvider } from './voice-provider'

/** Plugin titlebar items the JARVIS bar leaves out: the layout switch is the
 *  rail's Home button here. */
const HIDDEN_TITLEBAR_ITEMS = new Set(['jarvis-core:layout-toggle'])

export type LinkState = 'connecting' | 'offline' | 'online'

export function linkStateOf(gateway: string): LinkState {
  if (gateway === 'open') {
    return 'online'
  }

  return gateway === 'closed' || gateway === 'error' ? 'offline' : 'connecting'
}

function LogoGlyph() {
  return (
    <svg aria-hidden="true" className="jarvis-topbar__logo" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="9" />
      <circle cx="12" cy="12" r="4" />
      <path d="M12 1v4M12 19v4M1 12h4M19 12h4" />
    </svg>
  )
}

function SearchGlyph() {
  return (
    <svg aria-hidden="true" className="jarvis-topbar__icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <circle cx="11" cy="11" r="6" />
      <path d="M20 20l-4.5-4.5" />
    </svg>
  )
}

function useActiveSessionTitle(fallback: string): string {
  const sessions = useStore($sessions)
  const storedId = useStore($selectedStoredSessionId)
  const activeId = useStore($activeSessionId)
  const id = storedId ?? activeId
  const session = id ? sessions.find(s => s.id === id) : undefined

  return session ? sessionTitle(session) : fallback
}

/** Plugin contributions to the classic titlebar slots, rendered in-flow. */
function TitlebarItems({ area }: { area: string }) {
  // Plugin ids arrive scoped (`<plugin id>:<id>`).
  const items = useContributions(area).filter(c => !HIDDEN_TITLEBAR_ITEMS.has(c.id))

  return (
    <>
      {items.map(c => (
        <ContribBoundary id={c.id} key={`${c.source ?? 'core'}:${c.id}`} variant="chip">
          {c.render && <ContribRender render={c.render} />}
        </ContribBoundary>
      ))}
    </>
  )
}

export function JarvisTopBar() {
  const { locale, t } = useI18n()
  const text = t.jarvisShell
  const title = useActiveSessionTitle(text.newSession)
  const now = useMinuteClock()
  const link = linkStateOf(useStore($gatewayState))
  const latency = useStore($linkLatencyMs)
  const paletteHint = useKeybindHint('nav.commandPalette')

  const voice = useVoiceProvider()

  return (
    <header aria-label={text.topBar} className="jarvis-topbar" data-testid="jarvis-topbar">
      <div className="jarvis-topbar__brand">
        <LogoGlyph />
        <span className="jarvis-topbar__wordmark">{text.wordmark}</span>
        <span aria-hidden="true" className="jarvis-topbar__sep">
          /
        </span>
        <span className="jarvis-topbar__title" data-testid="jarvis-topbar-title">
          {title}
        </span>
        <span className="jarvis-topbar__slot">
          <TitlebarItems area="titleBar.left" />
        </span>
      </div>

      <time className="jarvis-topbar__clock" data-testid="jarvis-topbar-clock" dateTime={now.toISOString()}>
        {formatClock(now, locale)}
      </time>

      <div className="jarvis-topbar__actions">
        <span className="jarvis-topbar__slot">
          <TitlebarItems area="titleBar.center" />
          <TitlebarItems area="titleBar.right" />
        </span>
        <span className="jarvis-topbar__pill" data-link={link} data-testid="jarvis-topbar-link">
          <span aria-hidden="true" className="jarvis-topbar__dot" />
          {text.link[link]}
          {link === 'online' && latency !== null && <span> · {text.latency(latency)}</span>}
        </span>
        {voice && (
          <span className="jarvis-topbar__pill" data-testid="jarvis-topbar-voice">
            {text.voice(voice)}
          </span>
        )}
        <Button
          className="jarvis-topbar__command"
          data-testid="jarvis-topbar-command"
          onClick={() => openCommandPalette()}
          size="sm"
          type="button"
          variant="outline"
        >
          <SearchGlyph />
          {text.askOrCommand}
          {paletteHint && <span className="jarvis-topbar__kbd">{paletteHint}</span>}
        </Button>
      </div>
    </header>
  )
}
