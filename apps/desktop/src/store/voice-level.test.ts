import type { WritableAtom } from 'nanostores'
import { afterEach, describe, expect, it } from 'vitest'

import { host } from '@/sdk'

import { $voiceInput, type VoiceInputStatus } from './voice-input'
import { $micLevel, $playbackLevel, $voiceLevel, publishMicLevel, publishPlaybackLevel } from './voice-level'
import { setVoicePlaybackState, type VoicePlaybackStatus } from './voice-playback'

function playback(status: VoicePlaybackStatus) {
  setVoicePlaybackState({ audioElement: null, messageId: null, sequence: 0, source: null, status })
}

// Far-apart timestamps so the ~15 Hz throttle never interferes unless a test
// is about the throttle itself.
let clock = 1_000_000
const later = () => (clock += 1_000)

afterEach(() => {
  publishPlaybackLevel(0, later())
  publishMicLevel(0, later())
  playback('idle')
  $voiceInput.set('idle')
})

describe('voice level channels', () => {
  it('clamps to 0–1 and rounds to hundredths', () => {
    publishPlaybackLevel(1.7, later())
    expect($playbackLevel.get()).toBe(1)

    publishPlaybackLevel(0.4567, later())
    expect($playbackLevel.get()).toBe(0.46)

    publishPlaybackLevel(Number.NaN, later())
    expect($playbackLevel.get()).toBe(0)
  })

  it('throttles to ~15 Hz and ignores sub-threshold jitter', () => {
    const t0 = later()
    publishMicLevel(0.5, t0)
    publishMicLevel(0.9, t0 + 20) // too soon
    expect($micLevel.get()).toBe(0.5)

    publishMicLevel(0.51, t0 + 200) // too small a change
    expect($micLevel.get()).toBe(0.5)

    publishMicLevel(0.9, t0 + 300)
    expect($micLevel.get()).toBe(0.9)
  })

  it('drops to zero immediately, even inside the throttle window', () => {
    const t0 = later()
    publishPlaybackLevel(0.8, t0)
    publishPlaybackLevel(0, t0 + 1)

    expect($playbackLevel.get()).toBe(0)
  })
})

describe('$voiceLevel', () => {
  const set = (voice: VoicePlaybackStatus, input: VoiceInputStatus) => {
    playback(voice)
    $voiceInput.set(input)
  }

  it('follows reply playback while speaking', () => {
    publishPlaybackLevel(0.7, later())
    publishMicLevel(0.2, later())
    set('speaking', 'idle')

    expect($voiceLevel.get()).toBe(0.7)
  })

  it('follows the mic while listening or dictating — dictation even over playback (barge-in)', () => {
    publishPlaybackLevel(0.7, later())
    publishMicLevel(0.3, later())

    set('idle', 'listening')
    expect($voiceLevel.get()).toBe(0.3)

    set('speaking', 'dictating')
    expect($voiceLevel.get()).toBe(0.3)
  })

  it('is zero in phases without live audio', () => {
    publishPlaybackLevel(0.7, later())
    publishMicLevel(0.3, later())

    for (const [voice, input] of [
      ['idle', 'idle'],
      ['preparing', 'idle'],
      ['idle', 'transcribing']
    ] as const) {
      set(voice, input)
      expect($voiceLevel.get()).toBe(0)
    }
  })
})

describe('host.state.voiceLevel', () => {
  it('mirrors the selected level to plugins', () => {
    publishPlaybackLevel(0.6, later())
    playback('speaking')

    expect(host.state.voiceLevel.get()).toBe(0.6)
  })

  it('is write-isolated: a cast-away write cannot reach the app level stores', () => {
    ;(host.state.voiceLevel as unknown as WritableAtom<number>).set(1)

    expect($playbackLevel.get()).toBe(0)
    expect($micLevel.get()).toBe(0)
    expect($voiceLevel.get()).toBe(0)
  })
})
