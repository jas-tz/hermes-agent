import { atom, computed } from 'nanostores'

import { $voiceInput } from './voice-input'
import { $voicePlayback } from './voice-playback'

/**
 * Live 0–1 loudness of the voice the user is hearing or giving — a read-only
 * mirror of levels the app ALREADY measures, so a surface outside the composer
 * (the JARVIS face) can move with the actual audio:
 *
 *   playback — the Read Aloud waveform's analyser (an observe-only
 *              captureStream tap; see voice-activity.tsx)
 *   mic      — the main composer's recorder meter (dictation or the voice
 *              conversation)
 *
 * Publishing only stores numbers: nothing here opens a mic, touches an audio
 * graph or can affect playback / capture.
 */

// ~15 Hz is plenty for a glow and keeps subscribers from re-rendering every
// animation frame; small jitter below MIN_DELTA is dropped for the same reason.
const PUBLISH_INTERVAL_MS = 66
const MIN_DELTA = 0.02

function createLevelChannel() {
  const $level = atom(0)
  let lastAt = Number.NEGATIVE_INFINITY

  const publish = (level: number, now: number = performance.now()) => {
    const next = Number.isFinite(level) ? Math.min(1, Math.max(0, level)) : 0

    // Silence / teardown lands immediately so the face never sticks mid-swell.
    if (next === 0) {
      lastAt = now
      $level.set(0)

      return
    }

    if (now - lastAt < PUBLISH_INTERVAL_MS || Math.abs(next - $level.get()) < MIN_DELTA) {
      return
    }

    lastAt = now
    $level.set(Math.round(next * 100) / 100)
  }

  return { $level, publish }
}

const playback = createLevelChannel()
const mic = createLevelChannel()

export const $playbackLevel = playback.$level
export const $micLevel = mic.$level
export const publishPlaybackLevel = playback.publish
export const publishMicLevel = mic.publish

/**
 * The level that matches what the face is showing, in the face's own
 * precedence (jarvis-core/core-state.ts): the user's dictation outranks reply
 * playback (a barge-in), playback outranks an open conversation mic. Every
 * other phase — transcribing, thinking, idle — has no live audio: 0.
 */
export const $voiceLevel = computed(
  [$voicePlayback, $voiceInput, $playbackLevel, $micLevel],
  (playbackState, input, playbackLevel, micLevel) => {
    if (input === 'dictating') {
      return micLevel
    }

    if (playbackState.status === 'speaking') {
      return playbackLevel
    }

    return input === 'listening' ? micLevel : 0
  }
)
