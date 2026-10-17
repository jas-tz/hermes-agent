import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ChatBarState } from '@/app/chat/composer/types'
import { I18nProvider } from '@/i18n'
import { applyWakeStartResult, applyWakeStatus, resetWakeWordState } from '@/store/wake-word'

import { ComposerControls } from './controls'

vi.mock('./model-pill', () => ({ ModelPill: () => null }))

const shell = vi.hoisted(() => ({ jarvis: false }))

vi.mock('@/app/jarvis-shell/use-jarvis-shell', () => ({ useJarvisShell: () => shell.jarvis }))

const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { active: false, enabled: false }
}

function renderControls(overrides: Partial<React.ComponentProps<typeof ComposerControls>> = {}) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <ComposerControls
        autoSpeak={false}
        busy={false}
        busyAction="stop"
        canSubmit={true}
        conversation={{
          active: false,
          level: 0,
          muted: false,
          onEnd: vi.fn(),
          onStart: vi.fn(),
          onStopTurn: vi.fn(),
          onToggleMute: vi.fn(),
          status: 'idle'
        }}
        disabled={false}
        hasComposerPayload={true}
        onDictate={vi.fn()}
        onQueue={vi.fn()}
        onToggleAutoSpeak={vi.fn()}
        state={state}
        voiceStatus="idle"
        {...overrides}
      />
    </I18nProvider>
  )
}

async function expectShortcutTooltip(label: string, shortcut: string) {
  fireEvent.pointerMove(screen.getByLabelText(label), { pointerType: 'mouse' })

  const tooltip = await screen.findByRole('tooltip')

  expect(tooltip.textContent).toContain(label)
  expect(tooltip.textContent).toContain(shortcut)
}

afterEach(() => {
  cleanup()
  shell.jarvis = false
  vi.useRealTimers()
})

describe('ComposerControls shortcut tooltips', () => {
  it('shows Enter for Send', async () => {
    renderControls()

    await expectShortcutTooltip('Send', '↵')
  })

  it('shows Enter for Steer', async () => {
    renderControls({ busy: true, busyAction: 'steer' })

    await expectShortcutTooltip('Steer the current run', '↵')
  })

  it('shows Ctrl+Enter for Queue', async () => {
    renderControls({ busy: true, busyAction: 'queue' })

    await expectShortcutTooltip('Queue message', 'Ctrl+↵')
  })
})

describe('wake-word ear visibility', () => {
  afterEach(() => {
    resetWakeWordState()
  })

  it('stays mounted during a busy agent turn', () => {
    applyWakeStatus({ available: true, enabled: true, listening: true, phrase: 'hey hermes' })
    renderControls({ busy: true, busyAction: 'stop' })

    expect(screen.getByLabelText('Wake word: "hey hermes" — listening')).toBeTruthy()
  })

  it('stays mounted (enabled in config) even when a start was refused', () => {
    applyWakeStatus({ available: true, enabled: true, listening: false, phrase: 'hey hermes' })
    // Transient refusal marks available false but enabled keeps it mounted.
    applyWakeStartResult({ hint: 'mic busy', reason: 'unavailable', started: false })
    renderControls()

    expect(screen.getByLabelText('Wake word: "hey hermes" — off')).toBeTruthy()
  })

  it('stays visible (never hides) even when unavailable and not enabled', () => {
    applyWakeStatus({ available: false, enabled: false, listening: false, phrase: 'hey hermes' })
    renderControls()

    // The ear ALWAYS shows so the user can click to enable; a failed start
    // surfaces its reason in the tooltip rather than hiding the control.
    expect(screen.getByLabelText('Wake word: "hey hermes" — off')).toBeTruthy()
  })

  it('surfaces the backend refusal reason in the tooltip, still visible', () => {
    applyWakeStatus({ available: false, enabled: false, listening: false, phrase: 'hey hermes' })
    applyWakeStartResult({ hint: 'run `hermes tools` (Voice section)', reason: 'unavailable', started: false })
    renderControls()

    const ear = screen.getByLabelText('Wake word: "hey hermes" — off')
    expect(ear).toBeTruthy()
  })

  it('shows a disabled paused ear inside the voice-conversation pill', () => {
    applyWakeStatus({ available: true, enabled: true, listening: true, phrase: 'hey hermes' })
    renderControls({
      conversation: {
        active: true,
        level: 0,
        muted: false,
        onEnd: vi.fn(),
        onStart: vi.fn(),
        onStopTurn: vi.fn(),
        onToggleMute: vi.fn(),
        status: 'listening'
      }
    })

    const ear = screen.getByLabelText('Wake word: "hey hermes" — paused during voice chat')
    expect((ear as HTMLButtonElement).disabled).toBe(true)
  })
})

describe('JARVIS hold-to-talk mic', () => {
  const voiceOn: ChatBarState = { ...state, voice: { active: false, enabled: true } }

  it('keeps the small inline dictation toggle in the classic shell', () => {
    renderControls({ state: voiceOn })

    expect(screen.getByLabelText('Voice dictation')).toBeTruthy()
    expect(screen.queryByTestId('jarvis-mic')).toBeNull()
  })

  it('becomes the big mic at the end of the composer in the JARVIS shell', () => {
    shell.jarvis = true
    const { container } = renderControls({ state: voiceOn })
    const mic = screen.getByTestId('jarvis-mic')
    const buttons = [...container.querySelectorAll('button')]

    expect(buttons.at(-1)).toBe(mic)
    expect(mic.getAttribute('aria-label')).toBe('Hold to talk — or click to start and stop')
    expect(mic.className).toContain('jarvis-mic')
    expect(screen.queryByLabelText('Voice dictation')).toBeNull()
  })

  it('records while held and stops on release', () => {
    shell.jarvis = true
    vi.useFakeTimers()
    const onDictate = vi.fn()
    const { rerender } = renderControls({ onDictate, state: voiceOn })

    fireEvent.pointerDown(screen.getByTestId('jarvis-mic'), { button: 0, pointerId: 1 })
    expect(onDictate).toHaveBeenCalledTimes(1)

    // The recorder reports it is recording; the user keeps holding…
    rerender(
      <I18nProvider configClient={null} initialLocale="en">
        <ComposerControls
          autoSpeak={false}
          busy={false}
          busyAction="stop"
          canSubmit={true}
          conversation={{
            active: false,
            level: 0,
            muted: false,
            onEnd: vi.fn(),
            onStart: vi.fn(),
            onStopTurn: vi.fn(),
            onToggleMute: vi.fn(),
            status: 'idle'
          }}
          disabled={false}
          hasComposerPayload={true}
          onDictate={onDictate}
          onQueue={vi.fn()}
          onToggleAutoSpeak={vi.fn()}
          state={voiceOn}
          voiceStatus="recording"
        />
      </I18nProvider>
    )
    vi.advanceTimersByTime(800)
    fireEvent.pointerUp(screen.getByTestId('jarvis-mic'), { button: 0, pointerId: 1 })
    fireEvent.click(screen.getByTestId('jarvis-mic'))

    // …and letting go stops it — the follow-up click is not a third toggle.
    expect(onDictate).toHaveBeenCalledTimes(2)
  })

  it('a quick click starts recording without stopping it', () => {
    shell.jarvis = true
    vi.useFakeTimers()
    const onDictate = vi.fn()
    renderControls({ onDictate, state: voiceOn })
    const mic = screen.getByTestId('jarvis-mic')

    fireEvent.pointerDown(mic, { button: 0, pointerId: 1 })
    vi.advanceTimersByTime(100)
    fireEvent.pointerUp(mic, { button: 0, pointerId: 1 })
    fireEvent.click(mic)

    expect(onDictate).toHaveBeenCalledTimes(1)
  })

  it('still toggles from the keyboard (a click with no pointer press)', () => {
    shell.jarvis = true
    const onDictate = vi.fn()
    renderControls({ onDictate, state: voiceOn })

    fireEvent.click(screen.getByTestId('jarvis-mic'))

    expect(onDictate).toHaveBeenCalledTimes(1)
  })
})
