import type { VoiceInputStatus, VoiceStatus } from '@hermes/plugin-sdk'
import { describe, expect, it } from 'vitest'

import { CORE_MOODS, CORE_STATES, coreMood, type CoreState, linkStatus, resolveCoreState } from './core-state'

const BUSY = [false, true] as const
const VOICE: readonly VoiceStatus[] = ['idle', 'preparing', 'speaking']
const INPUT: readonly VoiceInputStatus[] = ['idle', 'listening', 'dictating', 'transcribing']

describe('resolveCoreState', () => {
  it('is idle only with no input, no turn and no playback', () => {
    expect(resolveCoreState({ awaitingInput: false, busy: false, voice: 'idle', voiceInput: 'idle' })).toBe('idle')

    for (const busy of BUSY) {
      for (const voice of VOICE) {
        for (const input of INPUT) {
          const idle = !busy && voice === 'idle' && input === 'idle'
          expect(resolveCoreState({ awaitingInput: false, busy, voice, voiceInput: input }) === 'idle').toBe(idle)
        }
      }
    }
  })

  it('thinks while a turn is in flight and nothing else is active', () => {
    expect(resolveCoreState({ awaitingInput: false, busy: true, voice: 'idle', voiceInput: 'idle' })).toBe('thinking')
  })

  it('listens only when the conversation reports the mic open', () => {
    expect(resolveCoreState({ awaitingInput: false, busy: false, voice: 'idle', voiceInput: 'listening' })).toBe(
      'listening'
    )
    // An enabled conversation mid-generation reports `thinking` upstream, so
    // voiceInput stays idle — the core must not claim to be listening.
    expect(resolveCoreState({ awaitingInput: false, busy: true, voice: 'idle', voiceInput: 'idle' })).toBe('thinking')
  })

  it('shows dictation and transcription over everything else', () => {
    for (const busy of BUSY) {
      for (const voice of VOICE) {
        expect(resolveCoreState({ awaitingInput: false, busy, voice, voiceInput: 'dictating' })).toBe('dictating')
        expect(resolveCoreState({ awaitingInput: false, busy, voice, voiceInput: 'transcribing' })).toBe('transcribing')
      }
    }
  })

  it('lets reply audio win over a running turn and an idle mic', () => {
    expect(resolveCoreState({ awaitingInput: false, busy: true, voice: 'speaking', voiceInput: 'idle' })).toBe(
      'speaking'
    )
    expect(resolveCoreState({ awaitingInput: false, busy: false, voice: 'speaking', voiceInput: 'idle' })).toBe(
      'speaking'
    )
    expect(resolveCoreState({ awaitingInput: false, busy: true, voice: 'preparing', voiceInput: 'idle' })).toBe(
      'preparing'
    )
    expect(resolveCoreState({ awaitingInput: false, busy: false, voice: 'preparing', voiceInput: 'idle' })).toBe(
      'preparing'
    )
  })

  it('ranks speaking above preparing above listening', () => {
    expect(resolveCoreState({ awaitingInput: false, busy: false, voice: 'speaking', voiceInput: 'listening' })).toBe(
      'speaking'
    )
    expect(resolveCoreState({ awaitingInput: false, busy: false, voice: 'preparing', voiceInput: 'listening' })).toBe(
      'preparing'
    )
  })

  it('shows needs-input over a busy turn, and without one', () => {
    expect(resolveCoreState({ awaitingInput: true, busy: true, voice: 'idle', voiceInput: 'idle' })).toBe('needs-input')
    expect(resolveCoreState({ awaitingInput: true, busy: false, voice: 'idle', voiceInput: 'idle' })).toBe(
      'needs-input'
    )
  })

  it('keeps every voice state above needs-input', () => {
    const awaiting = { awaitingInput: true, busy: true }

    expect(resolveCoreState({ ...awaiting, voice: 'idle', voiceInput: 'dictating' })).toBe('dictating')
    expect(resolveCoreState({ ...awaiting, voice: 'idle', voiceInput: 'transcribing' })).toBe('transcribing')
    expect(resolveCoreState({ ...awaiting, voice: 'speaking', voiceInput: 'idle' })).toBe('speaking')
    expect(resolveCoreState({ ...awaiting, voice: 'preparing', voiceInput: 'idle' })).toBe('preparing')
    expect(resolveCoreState({ ...awaiting, voice: 'idle', voiceInput: 'listening' })).toBe('listening')
  })

  it('leaves voice precedence unchanged when nothing is awaited', () => {
    for (const busy of BUSY) {
      for (const voice of VOICE) {
        for (const voiceInput of INPUT) {
          expect(resolveCoreState({ awaitingInput: false, busy, voice, voiceInput })).not.toBe('needs-input')
        }
      }
    }
  })

  it('walks a full voice-conversation turn in order', () => {
    const steps: Array<[boolean, VoiceStatus, VoiceInputStatus, CoreState]> = [
      [false, 'idle', 'listening', 'listening'], // mic open, waiting for speech
      [false, 'idle', 'transcribing', 'transcribing'], // utterance captured
      [true, 'idle', 'idle', 'thinking'], // submitted; barge monitor reports thinking
      [true, 'idle', 'transcribing', 'transcribing'], // user barges in mid-generation
      [true, 'idle', 'idle', 'thinking'], // interjection submitted
      [true, 'preparing', 'idle', 'preparing'], // reply arrives, synthesis starts
      [false, 'speaking', 'idle', 'speaking'], // reply audible, turn done
      [false, 'idle', 'listening', 'listening'] // mic re-arms for the next turn
    ]

    expect(
      steps.map(([busy, voice, voiceInput]) => resolveCoreState({ awaitingInput: false, busy, voice, voiceInput }))
    ).toEqual(steps.map(([, , , expected]) => expected))
  })

  it('reads offline only when the link is closed or errored', () => {
    const base = { awaitingInput: false, busy: false, voice: 'idle', voiceInput: 'idle' } as const

    expect(resolveCoreState({ ...base, gateway: 'closed' })).toBe('offline')
    expect(resolveCoreState({ ...base, gateway: 'error' })).toBe('offline')

    // Startup and reconnect are not "offline", and an unknown link is ignored.
    for (const gateway of ['idle', 'connecting', 'open', undefined]) {
      expect(resolveCoreState({ ...base, gateway })).toBe('idle')
    }
  })

  it('ranks offline above needs-you and thinking, below the voice states', () => {
    const down = { gateway: 'closed', voice: 'idle', voiceInput: 'idle' } as const

    expect(resolveCoreState({ ...down, awaitingInput: true, busy: true })).toBe('offline')

    for (const [voice, voiceInput, expected] of [
      ['speaking', 'idle', 'speaking'],
      ['preparing', 'idle', 'preparing'],
      ['idle', 'listening', 'listening'],
      ['idle', 'dictating', 'dictating'],
      ['idle', 'transcribing', 'transcribing']
    ] as const) {
      expect(resolveCoreState({ awaitingInput: false, busy: true, gateway: 'error', voice, voiceInput })).toBe(expected)
    }
  })
})

describe('linkStatus', () => {
  it('maps the gateway state to a short status key', () => {
    expect(linkStatus('open')).toBe('online')
    expect(linkStatus('connecting')).toBe('connecting')
    expect(linkStatus('idle')).toBe('connecting')
    expect(linkStatus(undefined)).toBe('connecting')
    expect(linkStatus('closed')).toBe('offline')
    expect(linkStatus('error')).toBe('offline')
  })
})

describe('coreMood', () => {
  it("maps every state onto the design's moods", () => {
    const moods = Object.fromEntries(CORE_STATES.map(state => [state, coreMood(state)]))

    expect(moods).toEqual({
      dictating: 'listening',
      idle: 'standby',
      listening: 'listening',
      'needs-input': 'needs-you',
      offline: 'offline',
      preparing: 'thinking',
      speaking: 'speaking',
      thinking: 'thinking',
      transcribing: 'thinking'
    })
  })

  it('uses each of the four design moods plus needs-you and offline', () => {
    expect(new Set(CORE_STATES.map(coreMood))).toEqual(new Set(CORE_MOODS))
  })

  it('lists every state the resolver can produce', () => {
    const produced = new Set<CoreState>()

    for (const busy of BUSY) {
      for (const voice of VOICE) {
        for (const voiceInput of INPUT) {
          for (const awaitingInput of [false, true]) {
            for (const gateway of ['open', 'closed']) {
              produced.add(resolveCoreState({ awaitingInput, busy, gateway, voice, voiceInput }))
            }
          }
        }
      }
    }

    expect(new Set(CORE_STATES)).toEqual(produced)
  })
})
