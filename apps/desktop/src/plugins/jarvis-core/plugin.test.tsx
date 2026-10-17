import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import type * as Sdk from '@hermes/plugin-sdk'
import {
  Contribute,
  type PluginContext,
  type PluginContribution,
  type VoiceInputStatus,
  type VoiceStatus
} from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { type WritableAtom } from 'nanostores'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { JarvisCorePane } from './core-pane'
import { CORE_MOODS, CORE_STATES, coreMood } from './core-state'
import { JARVIS_CORE_LOCALE_IDS, JARVIS_CORE_LOCALES } from './i18n'
import { JarvisLayoutToggle } from './layout-toggle'
import plugin, {
  ACTIVITY_TREE_ID,
  JARVIS_CORE_PANE_DATA,
  JARVIS_CORE_TREE_ID,
  JARVIS_LAYOUT_PRESET_ID,
  JARVIS_LAYOUT_TREE
} from './plugin'
import { $responseTimes, resetResponseTimes } from './response-times'

// Source-text read: vitest stubs CSS modules (even `?raw`) to empty strings.
const css = readFileSync(resolve(__dirname, 'jarvis-core.css'), 'utf8')

function valueAt(tree: unknown, key: string): unknown {
  return key
    .split('.')
    .reduce<unknown>(
      (node, part) => (node && typeof node === 'object' ? (node as Record<string, unknown>)[part] : undefined),
      tree
    )
}

function messageAt(tree: unknown, key: string): string | undefined {
  const value = key
    .split('.')
    .reduce<unknown>(
      (node, part) => (node && typeof node === 'object' ? (node as Record<string, unknown>)[part] : undefined),
      tree
    )

  return typeof value === 'string' ? value : undefined
}

// Drive the pane through test atoms standing in for the four SDK signals; the
// real SDK wiring (derivation + write isolation) is covered in src/sdk.
type TestLocale = (typeof JARVIS_CORE_LOCALE_IDS)[number]

const signals = vi.hoisted(() => ({
  applyLayoutPreset: null as unknown as ReturnType<typeof vi.fn>,
  /** The app language, standing in for `display.language`. */
  locale: null as unknown as WritableAtom<TestLocale>,
  /** The bundles `ctx.i18n.register` received (set in beforeAll). */
  bundles: null as Sdk.PluginLocaleBundles | null,
  awaitingInput: null as unknown as WritableAtom<boolean>,
  busy: null as unknown as WritableAtom<boolean>,
  gateway: null as unknown as WritableAtom<string>,
  layoutPreset: null as unknown as WritableAtom<string>,
  voice: null as unknown as WritableAtom<VoiceStatus>,
  voiceInput: null as unknown as WritableAtom<VoiceInputStatus>,
  voiceLevel: null as unknown as WritableAtom<number>
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const actual = await importOriginal<typeof Sdk>()
  const { useStore } = await import('@nanostores/react')
  // Loaded here, not from the file's imports: modules imported by this test
  // (response-times) pull in the mocked SDK during the import phase.
  const { atom } = await import('nanostores')
  signals.applyLayoutPreset = vi.fn()
  signals.locale = atom<TestLocale>('en')
  signals.awaitingInput = atom(false)
  signals.busy = atom(false)
  signals.gateway = atom('open')
  signals.layoutPreset = atom('default')
  signals.voice = atom<VoiceStatus>('idle')
  signals.voiceInput = atom<VoiceInputStatus>('idle')
  signals.voiceLevel = atom(0)

  return {
    ...actual,
    applyLayoutPreset: signals.applyLayoutPreset,
    // Plugin i18n as the app resolves it — active locale, then the plugin's
    // English, then the raw key — reading the bundles the plugin registers.
    usePluginI18n: (pluginId: string) => {
      const locale = useStore(signals.locale)

      return (key: string, ...args: unknown[]) => {
        const bundles = pluginId === 'jarvis-core' ? signals.bundles : null
        const value = valueAt(bundles?.[locale], key) ?? valueAt(bundles?.en, key)

        if (typeof value === 'function') {
          return (value as (...a: unknown[]) => string)(...args)
        }

        return typeof value === 'string' ? value : key
      }
    },
    host: {
      ...actual.host,
      state: {
        ...actual.host.state,
        awaitingInput: signals.awaitingInput,
        busy: signals.busy,
        gateway: signals.gateway,
        layoutPreset: signals.layoutPreset,
        voice: signals.voice,
        voiceInput: signals.voiceInput,
        voiceLevel: signals.voiceLevel
      }
    }
  }
})

beforeEach(() => {
  signals.applyLayoutPreset.mockClear()
  signals.awaitingInput.set(false)
  signals.busy.set(false)
  signals.gateway.set('open')
  signals.layoutPreset.set('default')
  signals.voice.set('idle')
  signals.voiceInput.set('idle')
  signals.voiceLevel.set(0)
})

// The copy the plugin registers through `ctx.i18n.register` (asserted below).
beforeAll(() => {
  signals.bundles = JARVIS_CORE_LOCALES
})

afterEach(() => {
  cleanup()
  resetResponseTimes()
  vi.useRealTimers()
  signals.locale.set('en')
})

function visualState() {
  return screen.getByTestId('jarvis-core-visual').getAttribute('data-state')
}

describe('JARVIS Core pane', () => {
  it('starts idle and follows turn and voice signals live', () => {
    render(<JarvisCorePane />)

    expect(visualState()).toBe('idle')
    expect(screen.getByRole('status').textContent).toBe('Standing by')

    act(() => signals.busy.set(true))
    expect(visualState()).toBe('thinking')
    expect(screen.getByRole('status').textContent).toBe('Thinking')
    expect(screen.getByText('Working')).toBeTruthy()

    act(() => signals.voice.set('preparing'))
    expect(visualState()).toBe('preparing')
    expect(screen.getByText('Synthesizing')).toBeTruthy()

    act(() => {
      signals.busy.set(false)
      signals.voice.set('speaking')
    })
    expect(visualState()).toBe('speaking')
    expect(screen.getByRole('status').textContent).toBe('Speaking')

    act(() => signals.voice.set('idle'))
    expect(visualState()).toBe('idle')
  })

  it('follows the microphone signal through a voice turn', () => {
    render(<JarvisCorePane />)

    act(() => signals.voiceInput.set('listening'))
    expect(visualState()).toBe('listening')
    expect(screen.getByRole('status').textContent).toBe('Listening')

    act(() => signals.voiceInput.set('transcribing'))
    expect(visualState()).toBe('transcribing')
    expect(screen.getByRole('status').textContent).toBe('Transcribing')

    act(() => {
      signals.voiceInput.set('idle')
      signals.busy.set(true)
    })
    expect(visualState()).toBe('thinking')
    expect(screen.getByText('Off')).toBeTruthy()

    act(() => {
      signals.busy.set(false)
      signals.voice.set('speaking')
    })
    expect(visualState()).toBe('speaking')

    act(() => {
      signals.voice.set('idle')
      signals.voiceInput.set('dictating')
    })
    expect(visualState()).toBe('dictating')
    expect(screen.getByRole('status').textContent).toBe('Dictating')
    expect(screen.getByText('Recording')).toBeTruthy()
  })

  it('shows needs-you while a turn is blocked on the user, then resumes', () => {
    render(<JarvisCorePane />)

    act(() => signals.busy.set(true))
    expect(visualState()).toBe('thinking')
    expect(screen.getByTestId('jarvis-core-caption').textContent).toBe('Working on it…')

    // An approval arrives: the turn stays busy, but it is waiting on you.
    act(() => signals.awaitingInput.set(true))
    expect(visualState()).toBe('needs-input')
    expect(screen.getByRole('status').textContent).toBe('Needs you')
    expect(screen.getByText('Waiting for you')).toBeTruthy()
    expect(screen.getByTestId('jarvis-core-caption').textContent).toBe('Respond in the chat to continue.')

    // Answered in the chat: back to thinking.
    act(() => signals.awaitingInput.set(false))
    expect(visualState()).toBe('thinking')
    expect(screen.getByTestId('jarvis-core-caption').textContent).toBe('Working on it…')
  })

  it('offers no way to answer a pending prompt', () => {
    render(<JarvisCorePane />)
    act(() => signals.awaitingInput.set(true))

    const pane = screen.getByTestId('jarvis-core-pane')
    expect(pane.querySelectorAll('button, a, input, select, textarea, [role="button"]')).toHaveLength(0)
  })

  it('keeps voice states above needs-you', () => {
    render(<JarvisCorePane />)

    act(() => {
      signals.awaitingInput.set(true)
      signals.voice.set('speaking')
    })
    expect(visualState()).toBe('speaking')
    // The Agent card still tells the user a prompt is waiting.
    const agent = screen.getByTestId('jarvis-core-card-agent')
    expect(agent.textContent).toContain('Waiting for you')
    expect(agent.textContent).toContain('Answer in the chat')

    act(() => signals.voice.set('idle'))
    expect(visualState()).toBe('needs-input')
  })

  it('feeds the live voice level to the visual, clamped to 0–1', () => {
    render(<JarvisCorePane />)
    const visual = screen.getByTestId('jarvis-core-visual')

    expect(visual.style.getPropertyValue('--jarvis-level')).toBe('0')

    act(() => {
      signals.voice.set('speaking')
      signals.voiceLevel.set(0.64)
    })
    expect(visual.style.getPropertyValue('--jarvis-level')).toBe('0.64')

    act(() => signals.voiceLevel.set(3))
    expect(visual.style.getPropertyValue('--jarvis-level')).toBe('1')
  })

  it('reads offline when the backend link drops, and reports the link', () => {
    render(<JarvisCorePane />)
    expect(screen.getByText('Online')).toBeTruthy()

    act(() => {
      signals.busy.set(true)
      signals.gateway.set('closed')
    })
    expect(visualState()).toBe('offline')
    expect(screen.getByRole('status').textContent).toBe('Offline')

    // Reconnecting is not offline: the turn reads as itself again.
    act(() => signals.gateway.set('connecting'))
    expect(visualState()).toBe('thinking')
    expect(screen.getByText('Connecting')).toBeTruthy()
  })

  it('keeps an in-progress playback above offline', () => {
    render(<JarvisCorePane />)

    act(() => {
      signals.gateway.set('error')
      signals.voice.set('speaking')
    })
    expect(visualState()).toBe('speaking')
  })

  it('leaves the J.A.R.V.I.S wordmark to the top bar', () => {
    render(<JarvisCorePane />)

    expect(screen.queryByTestId('jarvis-core-wordmark')).toBeNull()
  })

  it('renders no interactive controls', () => {
    render(<JarvisCorePane />)

    const pane = screen.getByTestId('jarvis-core-pane')
    expect(pane.querySelectorAll('button, a, input, select, textarea, [role="button"]')).toHaveLength(0)
  })

  it('keeps the animated visual out of the accessibility tree', () => {
    render(<JarvisCorePane />)

    expect(screen.getByTestId('jarvis-core-visual').getAttribute('aria-hidden')).toBe('true')
    expect(screen.getByRole('status').getAttribute('aria-live')).toBe('polite')
  })
})

describe('JARVIS Core plugin', () => {
  function registerWith() {
    const registered: PluginContribution[] = []

    const ctx = {
      register: (c: PluginContribution) => {
        registered.push(c)

        return () => undefined
      },
      registerMany: (cs: PluginContribution[]) => {
        registered.push(...cs)

        return () => undefined
      },
      i18n: { register: vi.fn(() => () => undefined), t: (key: string) => key },
      onDispose: () => undefined,
      rest: vi.fn(),
      socket: vi.fn()
    } as unknown as PluginContext

    plugin.register(ctx)

    return { ctx, registered }
  }

  it('ships enabled and stays separate from the JARVIS Memory plugin', () => {
    expect(plugin.id).toBe('jarvis-core')
    expect(plugin.defaultEnabled).toBe(true)
  })

  it('registers the face and Activity panes, one layout preset and one titlebar entry', () => {
    const { registered } = registerWith()

    expect(registered.map(c => [c.area, c.id])).toEqual([
      ['panes', 'core'],
      ['panes', 'activity'],
      ['layouts', 'layout'],
      ['titleBar.center', 'layout-toggle']
    ])
    expect(JARVIS_LAYOUT_PRESET_ID).toBe('jarvis-core:layout')
    expect(registered[0]).toMatchObject({ title: 'JARVIS Core', data: JARVIS_CORE_PANE_DATA })
    // The Activity tab docks in the chat's own zone, like the face does.
    expect(registered[1]).toMatchObject({ data: { dock: { pane: 'workspace', pos: 'center' } } })
    expect(registered[2]).toMatchObject({ title: 'JARVIS', data: JARVIS_LAYOUT_TREE })
    expect(JARVIS_CORE_TREE_ID).toBe('jarvis-core:core')
  })

  it('uses a new pane id so a saved fixed-width zone cannot inherit a flexible size', () => {
    const { registered } = registerWith()

    // `pane` was the Phase 1 fixed 17rem zone; reusing it for a flexible pane
    // would expand the saved zone to its even split (half the window).
    expect(registered.some(c => c.area === 'panes' && c.id === 'pane')).toBe(false)
  })

  it('stays in the grid on narrow windows instead of collapsing into a hidden overlay', () => {
    // Narrow viewports (<768px CSS, e.g. a half-width window at 130% zoom)
    // move every collapsible pane off-grid; the face must stay visible.
    expect(JARVIS_CORE_PANE_DATA).not.toHaveProperty('collapsible')
  })

  it('declares no fixed width or cap, and docks as a tab in the chat zone', () => {
    expect(JARVIS_CORE_PANE_DATA).not.toHaveProperty('width')
    expect(JARVIS_CORE_PANE_DATA).not.toHaveProperty('maxWidth')
    expect(JARVIS_CORE_PANE_DATA).toMatchObject({ dock: { pane: 'workspace', pos: 'center' }, placement: 'right' })
  })

  it('registers its four locale bundles through the plugin i18n door', () => {
    const { ctx } = registerWith()

    expect(ctx.i18n.register).toHaveBeenCalledWith(JARVIS_CORE_LOCALES)
  })

  it('makes no REST or socket calls when registering', () => {
    const { ctx } = registerWith()

    expect(ctx.rest).not.toHaveBeenCalled()
    expect(ctx.socket).not.toHaveBeenCalled()
  })
})

describe('JARVIS titlebar entry', () => {
  const renderToggle = () => render(<JarvisLayoutToggle presetId={JARVIS_LAYOUT_PRESET_ID} tree={JARVIS_LAYOUT_TREE} />)

  it('switches to the JARVIS layout through the regular preset path', () => {
    renderToggle()
    const toggle = screen.getByTestId('jarvis-layout-toggle')

    expect(toggle.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(toggle)

    expect(signals.applyLayoutPreset).toHaveBeenCalledWith(JARVIS_LAYOUT_PRESET_ID, JARVIS_LAYOUT_TREE)
  })

  it('explains itself with a themed tip, never a native title', () => {
    renderToggle()
    const toggle = screen.getByTestId('jarvis-layout-toggle')

    expect(toggle.hasAttribute('title')).toBe(false)
    expect(toggle.getAttribute('aria-label')).toBe('JARVIS layout')
    // Radix's tooltip trigger marks the wrapped control.
    expect(toggle.getAttribute('data-state')).toMatch(/closed|delayed-open|instant-open/)
  })

  it('takes its size from the shared Button size, not className overrides', () => {
    const source = readFileSync(resolve(__dirname, 'layout-toggle.tsx'), 'utf8')
    const className = source.match(/className=\{cn\(([\s\S]*?)\)\}/)?.[1] ?? ''

    expect(source).toContain('size="xs"')
    expect(className).not.toMatch(/\b(h|px|py|rounded|size)-/)
    expect(className).not.toMatch(/text-\[/)
  })

  it('shows as pressed while the JARVIS layout is active', () => {
    renderToggle()

    act(() => signals.layoutPreset.set(JARVIS_LAYOUT_PRESET_ID))
    expect(screen.getByTestId('jarvis-layout-toggle').getAttribute('aria-pressed')).toBe('true')
  })
})

describe('JARVIS layout preset', () => {
  type Node = typeof JARVIS_LAYOUT_TREE

  const groups = (node: Node): Extract<Node, { type: 'group' }>[] =>
    node.type === 'group' ? [node] : node.children.flatMap(groups)

  const zoneOf = (paneId: string) => groups(JARVIS_LAYOUT_TREE).find(g => g.panes.includes(paneId))

  it('is sessions | face | chat, left to right', () => {
    expect(JARVIS_LAYOUT_TREE).toMatchObject({ orientation: 'row', type: 'split' })
    expect(groups(JARVIS_LAYOUT_TREE).map(g => g.panes[0])).toEqual(['sessions', JARVIS_CORE_TREE_ID, 'workspace'])
  })

  it('gives the face its own zone, the largest share', () => {
    expect(zoneOf(JARVIS_CORE_TREE_ID)?.panes).toEqual([JARVIS_CORE_TREE_ID])

    const root = JARVIS_LAYOUT_TREE as Extract<Node, { type: 'split' }>
    expect(Math.max(...root.weights)).toBe(root.weights[1])
  })

  it('leaves the chat column room for the full composer at an ordinary window width', () => {
    // At 940px, minus the 64px JARVIS rail and the 237px sessions zone: the
    // chat's share must clear the ~250px the composer needs to keep every
    // voice control on screen (the layout engine also floors it at 22vw).
    const root = JARVIS_LAYOUT_TREE as Extract<Node, { type: 'split' }>
    const [, face, chat] = root.weights
    const chatPx = ((940 - 64 - 237) * chat) / (face + chat)

    expect(chatPx).toBeGreaterThanOrEqual(250)
  })

  it('gives the face zone room for five cards in a row on a full-screen 1920×1080 window at 130% zoom', () => {
    // 1920 / 1.296 ≈ 1481px viewport; the cards go one-row from 42rem (672px).
    const root = JARVIS_LAYOUT_TREE as Extract<Node, { type: 'split' }>
    const [, face, chat] = root.weights
    const facePx = ((1481 - 64 - 237) * face) / (face + chat)

    expect(facePx).toBeGreaterThanOrEqual(672)
  })

  it('gives the chat zone the design tabs: Conversation, Activity, Files', () => {
    expect(zoneOf('workspace')?.panes).toEqual(['workspace', ACTIVITY_TREE_ID, 'files', 'review'])
    expect(ACTIVITY_TREE_ID).toBe('jarvis-core:activity')
  })

  it('keeps the real chat visible beside the face', () => {
    const chat = zoneOf('workspace')

    expect(chat).not.toBe(zoneOf(JARVIS_CORE_TREE_ID))
    expect(chat?.active).toBe('workspace')
  })

  it('leaves the terminal out so the preset never forces it open', () => {
    expect(zoneOf('terminal')).toBeUndefined()
  })
})

function visualMood() {
  return screen.getByTestId('jarvis-core-visual').getAttribute('data-mood')
}

// A CSS rule's body for an exact selector list (vitest stubs CSS imports, so
// the stylesheet is checked as source text, like the motion tests below).
function ruleFor(selector: string): string | undefined {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

  return css.match(new RegExp(`(?:^|\\})\\s*([^{}]*${escaped}[^{}]*)\\{([^}]*)\\}`))?.[2]
}

describe('JARVIS Core moods', () => {
  it('shows the four design moods as the agent works', () => {
    render(<JarvisCorePane />)
    expect(visualMood()).toBe('standby')

    act(() => signals.voiceInput.set('listening'))
    expect(visualMood()).toBe('listening')

    act(() => {
      signals.voiceInput.set('idle')
      signals.busy.set(true)
    })
    expect(visualMood()).toBe('thinking')

    act(() => {
      signals.busy.set(false)
      signals.voice.set('speaking')
    })
    expect(visualMood()).toBe('speaking')
  })

  it("keeps each state's own label while sharing a mood", () => {
    render(<JarvisCorePane />)

    act(() => signals.voiceInput.set('dictating'))
    expect(visualMood()).toBe('listening')
    expect(screen.getByRole('status').textContent).toBe('Dictating')

    act(() => signals.voiceInput.set('transcribing'))
    expect(visualMood()).toBe('thinking')
    expect(screen.getByRole('status').textContent).toBe('Transcribing')

    act(() => {
      signals.voiceInput.set('idle')
      signals.voice.set('preparing')
    })
    expect(visualMood()).toBe('thinking')
    expect(screen.getByRole('status').textContent).toBe('Preparing voice')
  })

  it('shows needs-you in amber and offline as a dimmed standby', () => {
    render(<JarvisCorePane />)

    act(() => {
      signals.busy.set(true)
      signals.awaitingInput.set(true)
    })
    expect(visualMood()).toBe('needs-you')
    expect(screen.getByRole('status').textContent).toBe('Needs you')
    expect(ruleFor(".jarvis-core[data-mood='needs-you']")).toContain('color: var(--jarvis-thinking)')

    act(() => signals.gateway.set('closed'))
    expect(visualMood()).toBe('offline')
    expect(screen.getByRole('status').textContent).toBe('Offline')
    expect(ruleFor(".jarvis-core[data-mood='offline']")).toMatch(/opacity: 0\.4/)
  })

  it('draws every layer and expression glyph the moods switch between', () => {
    render(<JarvisCorePane />)

    for (const id of ['ticks', 'orbits', 'bars', 'expr-line', 'expr-eyes', 'expr-ellipsis', 'expr-wave']) {
      expect(screen.getByTestId(`jarvis-core-${id}`)).toBeTruthy()
    }

    expect(screen.getByTestId('jarvis-core-ticks').querySelectorAll('rect')).toHaveLength(60)
    expect(screen.getByTestId('jarvis-core-bars').querySelectorAll('rect')).toHaveLength(36)
    // Two orbiting dots until a running-tool count exists.
    expect(screen.getByTestId('jarvis-core-orbits').querySelectorAll('circle')).toHaveLength(2)
  })

  it('gives each mood its design colour, fixed regardless of the app theme', () => {
    const palette = css.match(/\n\.jarvis-core-pane,\n\.jarvis-core \{([^}]*)\}/)?.[1] ?? ''
    // The face's own rule (not the palette rule it shares with the pane).
    const base = css.match(/\}\n\n\.jarvis-core \{([^}]*)\}/)?.[1] ?? ''

    expect(palette).toContain('--jarvis-standby: #3fa9d6')
    expect(palette).toContain('--jarvis-listening: #4fd1ff')
    expect(palette).toContain('--jarvis-thinking: #ffb547')
    expect(palette).toContain('--jarvis-speaking: #7fe3ff')
    expect(base).toContain('color: var(--jarvis-standby)')
    expect(ruleFor(".jarvis-core[data-mood='listening']")).toContain('color: var(--jarvis-listening)')
    expect(ruleFor(".jarvis-core[data-mood='thinking']")).toContain('color: var(--jarvis-thinking)')
    expect(ruleFor(".jarvis-core[data-mood='speaking']")).toContain('color: var(--jarvis-speaking)')

    // The face itself never borrows the theme colour.
    const face = css.slice(css.indexOf('/* ---- the face'))
    expect(face).not.toContain('--theme-primary')
  })

  it('runs the rings at half strength only while standing by', () => {
    const palette = css.match(/\n\.jarvis-core-pane,\n\.jarvis-core \{([^}]*)\}/)?.[1] ?? ''

    expect(palette).toContain('--jarvis-ring: 0.5')
    expect(ruleFor(".jarvis-core[data-mood='listening']")).toContain('--jarvis-ring: 1')
    expect(ruleFor(".jarvis-core[data-mood='speaking']")).toContain('--jarvis-ring: 1')
    expect(ruleFor('.jarvis-core__rings')).toContain('opacity: var(--jarvis-ring)')
  })

  it('pairs each mood with its layer and expression', () => {
    const shown = (layer: string) =>
      css.match(new RegExp(`([^{}]*\\.jarvis-core__${layer}[^{}]*)\\{\\s*display: inline;`))?.[1] ?? ''

    const exprRule = css.match(/([^{}]*\.jarvis-core__expr--line[^{}]*)\{\s*display: inline;/)?.[1] ?? ''

    expect(shown('ticks')).toContain("[data-mood='listening'] .jarvis-core__ticks")
    expect(shown('ticks')).toContain("[data-mood='thinking'] .jarvis-core__orbits")
    expect(shown('ticks')).toContain("[data-mood='speaking'] .jarvis-core__bars")

    expect(exprRule).toContain("[data-mood='standby'] .jarvis-core__expr--line")
    expect(exprRule).toContain("[data-mood='offline'] .jarvis-core__expr--line")
    expect(exprRule).toContain("[data-mood='listening'] .jarvis-core__expr--eyes")
    expect(exprRule).toContain("[data-mood='needs-you'] .jarvis-core__expr--eyes")
    expect(exprRule).toContain("[data-mood='thinking'] .jarvis-core__expr--ellipsis")
    expect(exprRule).toContain("[data-mood='speaking'] .jarvis-core__expr--wave")
  })
})

describe('JARVIS Core centre stage', () => {
  const caption = () => screen.getByTestId('jarvis-core-caption').textContent
  const card = (id: string) => screen.getByTestId(`jarvis-core-card-${id}`)

  it('captions every state under the mood label', () => {
    render(<JarvisCorePane />)
    expect(caption()).toBe('Start a voice conversation or type in the chat.')

    act(() => signals.voiceInput.set('listening'))
    expect(caption()).toBe("I'm listening — go ahead.")

    act(() => signals.voiceInput.set('dictating'))
    expect(caption()).toBe('Recording your dictation.')

    act(() => signals.voiceInput.set('transcribing'))
    expect(caption()).toBe('Turning your speech into text…')

    act(() => {
      signals.voiceInput.set('idle')
      signals.voice.set('preparing')
    })
    expect(caption()).toBe('Preparing the spoken reply…')

    act(() => signals.voice.set('speaking'))
    expect(caption()).toBe('Reading the reply aloud.')

    act(() => {
      signals.voice.set('idle')
      signals.gateway.set('error')
    })
    expect(caption()).toBe("Can't reach the Hermes backend.")
  })

  it('shows Agent, Microphone, Voice and Link cards, in the design order', () => {
    render(<JarvisCorePane />)
    const cards = screen.getByTestId('jarvis-core-cards')

    expect(cards.tagName).toBe('DL')
    expect([...cards.children].map(c => c.getAttribute('data-testid'))).toEqual([
      'jarvis-core-card-agent',
      'jarvis-core-card-microphone',
      'jarvis-core-card-voice',
      'jarvis-core-card-link',
      'jarvis-core-card-response-time'
    ])

    // label, value, sub-line
    expect([...card('agent').children].map(c => c.textContent)).toEqual(['Agent', 'Idle', 'Ready'])
    expect([...card('microphone').children].map(c => c.textContent)).toEqual(['Microphone', 'Off', 'Mic closed'])
    expect([...card('voice').children].map(c => c.textContent)).toEqual(['Voice', 'Idle', 'Spoken replies'])
    expect([...card('link').children].map(c => c.textContent)).toEqual(['Link', 'Online', 'Hermes backend'])
  })

  it('keeps every card in step with its signal', () => {
    render(<JarvisCorePane />)

    act(() => {
      signals.busy.set(true)
      signals.voiceInput.set('listening')
      signals.voice.set('preparing')
      signals.gateway.set('connecting')
    })
    expect(card('agent').textContent).toBe('AgentWorkingRunning a turn')
    expect(card('microphone').textContent).toBe('MicrophoneListeningVoice conversation')
    expect(card('voice').textContent).toBe('VoiceSynthesizingGenerating speech')
    expect(card('link').textContent).toBe('LinkConnectingReaching Hermes…')

    act(() => {
      signals.voiceInput.set('dictating')
      signals.voice.set('speaking')
      signals.gateway.set('closed')
    })
    expect(card('microphone').textContent).toBe('MicrophoneRecordingDictation')
    expect(card('voice').textContent).toBe('VoicePlayingReply audio')
    expect(card('link').textContent).toBe('LinkOfflineBackend unreachable')
  })

  it('tints the mood label with the face mood', () => {
    render(<JarvisCorePane />)
    const label = screen.getByRole('status')

    expect(label.getAttribute('data-mood')).toBe('standby')
    act(() => signals.busy.set(true))
    expect(label.getAttribute('data-mood')).toBe('thinking')

    expect(ruleFor('.jarvis-core-pane__status {')).toBeUndefined()
    const base = css.match(/\n\.jarvis-core-pane__status \{([^}]*)\}/)?.[1] ?? ''
    expect(base).toContain('--jarvis-mood-color: var(--jarvis-standby)')
    // Mixed toward the theme's text colour so it reads on light and dark skins.
    expect(base).toContain('color: color-mix(in srgb, var(--jarvis-mood-color) 78%, var(--ui-text-primary))')
    expect(ruleFor(".jarvis-core-pane__status[data-mood='listening']")).toContain('var(--jarvis-listening)')
    expect(ruleFor(".jarvis-core-pane__status[data-mood='speaking']")).toContain('var(--jarvis-speaking)')
    expect(ruleFor(".jarvis-core-pane__status[data-mood='offline']")).toContain('var(--ui-text-tertiary)')
    expect(css).toMatch(
      /\.jarvis-core-pane__status\[data-mood='thinking'\],\n\.jarvis-core-pane__status\[data-mood='needs-you'\] \{\s*--jarvis-mood-color: var\(--jarvis-thinking\);/
    )
  })

  it('shares the face palette with the label from one rule', () => {
    expect(css.match(/--jarvis-standby: #/g)).toHaveLength(1)
    expect(css).toMatch(/\n\.jarvis-core-pane,\n\.jarvis-core \{[^}]*--jarvis-standby: #3fa9d6/)
  })

  it('sizes the face from the height left over, so the stage fits without scrolling', () => {
    const stage = css.match(/\n\.jarvis-core-pane__stage \{([^}]*)\}/)?.[1] ?? ''
    const slot = css.match(/\n\.jarvis-core-pane__face-slot \{([^}]*)\}/)?.[1] ?? ''

    expect(stage).toContain('height: 100%')
    expect(stage).toContain('justify-content: safe center')
    // The face is the only flexible item: it takes the leftover height,
    // never below 6rem, at most the design's 25rem (≈400px at 1920×1080).
    expect(slot).toContain('flex: 1 1 0')
    expect(slot).toContain('min-height: 6rem')
    expect(slot).toContain('max-height: 25rem')
    expect(css).toMatch(/\.jarvis-core-pane__status,\n\.jarvis-core-pane__caption \{\s*flex: none;/)
    expect(css).not.toContain('--jarvis-stage-rest')
  })

  it('lays the cards in one row when wide, two across only when narrow', () => {
    const grid = css.match(/\n\.jarvis-core-pane__cards \{([^}]*)\}/)?.[1] ?? ''
    const wide = css.match(/@container \(min-width: 42rem\) \{([\s\S]*?)\n\}/)?.[1] ?? ''

    expect(grid).toContain('grid-template-columns: repeat(2, minmax(0, 1fr))')
    expect(wide).toContain('grid-template-columns: repeat(5, minmax(0, 1fr))')
  })

  it('styles from the JARVIS theme tokens, with app-token fallbacks for other skins', () => {
    const cardRule = css.match(/\n\.jarvis-core-card \{([^}]*)\}/)?.[1] ?? ''

    expect(cardRule).toContain('border: 1px solid var(--jarvis-line, var(--ui-stroke-tertiary))')
    expect(cardRule).toContain('border-radius: var(--jarvis-radius-card, var(--radius-lg))')
    expect(cardRule).toContain('background: var(--jarvis-panel, var(--ui-widget-surface-background))')
    expect(css).toContain('font-family: var(--jarvis-font-display, inherit)')
    expect(css).not.toMatch(/\.jarvis-core-card[^{]*\{[^}]*#[0-9a-f]{3,6}/i)
  })

  it('freezes the label colour change under prefers-reduced-motion', () => {
    const reduced = css.match(/@media \(prefers-reduced-motion: reduce\) \{([\s\S]*?)\n\}/)?.[1] ?? ''

    expect(reduced).toMatch(/\.jarvis-core-pane__status,[\s\S]*?transition: none !important;/)
  })

  it('translates the caption and cards with the app language', () => {
    render(<JarvisCorePane />)

    act(() => signals.locale.set('zh'))
    expect(caption()).toBe('开始语音对话，或在聊天中输入。')
    expect(card('microphone').textContent).toBe('麦克风关闭麦克风已关闭')

    act(() => signals.locale.set('ja'))
    expect(card('link').textContent).toBe('接続オンラインHermes バックエンド')
  })
})

describe('JARVIS Core response time and stage widgets', () => {
  it('shows the last turn time over a sparkline once turns complete', () => {
    render(<JarvisCorePane />)
    const card = screen.getByTestId('jarvis-core-card-response-time')

    expect(card.textContent).toContain('Response time')
    expect(card.textContent).toContain('—')
    expect(card.textContent).toContain('No replies yet')

    act(() => $responseTimes.set([4200, 2100]))

    expect(card.textContent).toContain('2.1 sec · last 2 replies')
    expect(screen.getByTestId('jarvis-core-sparkline').getAttribute('points')).not.toBe('')
  })

  it('renders widgets other plugins contribute under the cards', () => {
    render(
      <>
        <JarvisCorePane />
        <Contribute area="jarvis-core.stage" id="test:tasks">
          <p>TASKS-BAR</p>
        </Contribute>
      </>
    )

    expect(screen.getByTestId('jarvis-core-stage-slot').textContent).toBe('TASKS-BAR')
  })
})

describe('JARVIS Core voice level', () => {
  it('follows the live level while it flows, then hands back to the built-in loop', () => {
    vi.useFakeTimers()
    render(<JarvisCorePane />)
    const visual = screen.getByTestId('jarvis-core-visual')

    act(() => signals.voiceInput.set('listening'))
    expect(visual.hasAttribute('data-driven')).toBe(false)

    act(() => signals.voiceLevel.set(0.5))
    expect(visual.hasAttribute('data-driven')).toBe(true)

    // A pause between words keeps it driven…
    act(() => signals.voiceLevel.set(0))
    act(() => vi.advanceTimersByTime(200))
    expect(visual.hasAttribute('data-driven')).toBe(true)

    act(() => signals.voiceLevel.set(0.4))
    act(() => signals.voiceLevel.set(0))
    act(() => vi.advanceTimersByTime(399))
    expect(visual.hasAttribute('data-driven')).toBe(true)

    // …real silence hands back to the built-in loop.
    act(() => vi.advanceTimersByTime(1))
    expect(visual.hasAttribute('data-driven')).toBe(false)
  })

  it('drives the tick ring with the mic and the bars with the reply audio', () => {
    expect(ruleFor('.jarvis-core[data-driven] .jarvis-core__tick')).toMatch(
      /transform: scaleY\(calc\(0\.4 \+ var\(--jarvis-level, 0\)/
    )
    expect(ruleFor('.jarvis-core[data-driven] .jarvis-core__bar')).toMatch(
      /transform: scaleY\(calc\(0\.15 \+ var\(--jarvis-level, 0\)/
    )
    // The ticks and bars only exist on screen in their own moods.
    expect(css).toContain("[data-mood='listening'] .jarvis-core__ticks")
    expect(css).toContain("[data-mood='speaking'] .jarvis-core__bars")
  })
})

describe('JARVIS Core motion', () => {
  it('only animates transform and opacity', () => {
    const keyframes = css.match(/@keyframes[^{]+\{([\s\S]*?)\n\}/g) ?? []
    const props = keyframes.flatMap(block => [...block.matchAll(/^\s+([a-z-]+):/gm)].map(m => m[1]))

    expect(keyframes.length).toBeGreaterThan(0)
    expect(new Set(props)).toEqual(new Set(['opacity', 'transform']))
  })

  it('styles every mood, and every state maps to one', () => {
    for (const mood of CORE_MOODS) {
      expect(css).toContain(`[data-mood='${mood}']`)
    }

    for (const state of CORE_STATES) {
      expect(CORE_MOODS).toContain(coreMood(state))
    }
  })

  it('breathes slowly (6 s) while standing by', () => {
    expect(ruleFor(".jarvis-core[data-mood='standby'] .jarvis-core__glow")).toContain('animation-duration: 6s')
  })

  it('holds offline completely still', () => {
    const still = css.match(/([^{}]*\[data-mood='offline'\] \.jarvis-core__ring-outer[^{}]*)\{\s*animation: none;/)?.[1]

    expect(still).toContain("[data-mood='offline'] .jarvis-core__ring-segmented")
    expect(still).toContain("[data-mood='offline'] .jarvis-core__glow")
  })

  it('disables every animation and transition under prefers-reduced-motion', () => {
    const reduced = css.match(/@media \(prefers-reduced-motion: reduce\) \{([\s\S]*?)\n\}/)?.[1] ?? ''

    expect(reduced).toMatch(/\.jarvis-core \*[^{]*\{\s*animation: none !important;\s*transition: none !important;/)
  })

  it('freezes the level-driven ticks and bars under prefers-reduced-motion', () => {
    const reduced = css.match(/@media \(prefers-reduced-motion: reduce\) \{([\s\S]*?)\n\}/)?.[1] ?? ''

    expect(reduced).toMatch(/\[data-driven\] \.jarvis-core__bar \{\s*transform: none !important;/)
    expect(reduced).toContain('.jarvis-core[data-driven] .jarvis-core__tick')
  })
})

describe('JARVIS Core translations', () => {
  type Tree = Record<string, unknown>

  const keys = (tree: Tree, prefix = ''): string[] =>
    Object.entries(tree).flatMap(([k, v]) =>
      v && typeof v === 'object' ? keys(v as Tree, `${prefix}${k}.`) : [`${prefix}${k}`]
    )

  it('ships the same complete set of strings in en, ja, zh and zh-hant', () => {
    const en = keys(JARVIS_CORE_LOCALES.en as Tree).sort()

    expect(Object.keys(JARVIS_CORE_LOCALES).sort()).toEqual([...JARVIS_CORE_LOCALE_IDS].sort())

    for (const locale of JARVIS_CORE_LOCALE_IDS) {
      expect(keys(JARVIS_CORE_LOCALES[locale] as Tree).sort()).toEqual(en)

      for (const key of en) {
        const value = valueAt(JARVIS_CORE_LOCALES[locale], key)
        const text = typeof value === 'function' ? (value as (...args: number[]) => string)(3) : value

        expect(typeof text === 'string' && text.trim(), `${locale}:${key}`).toBeTruthy()
      }
    }
  })

  it('translates every state label', () => {
    for (const locale of ['ja', 'zh', 'zh-hant'] as const) {
      for (const state of CORE_STATES) {
        expect(messageAt(JARVIS_CORE_LOCALES[locale], `states.${state}`)).not.toBe(
          messageAt(JARVIS_CORE_LOCALES.en, `states.${state}`)
        )
      }
    }
  })

  it('follows the app language live', () => {
    render(
      <>
        <JarvisCorePane />
        <JarvisLayoutToggle presetId={JARVIS_LAYOUT_PRESET_ID} tree={JARVIS_LAYOUT_TREE} />
      </>
    )

    expect(screen.getByRole('status').textContent).toBe('Standing by')

    act(() => signals.locale.set('ja'))
    expect(screen.getByRole('status').textContent).toBe('待機中')
    expect(screen.getByText('マイク')).toBeTruthy()
    expect(screen.getByTestId('jarvis-layout-toggle').getAttribute('aria-label')).toBe('JARVIS レイアウト')

    act(() => signals.voice.set('speaking'))
    expect(screen.getByRole('status').textContent).toBe('話しています')

    act(() => signals.locale.set('zh-hant'))
    expect(screen.getByRole('status').textContent).toBe('說話中')
    expect(screen.getByText('麥克風')).toBeTruthy()
  })
})
