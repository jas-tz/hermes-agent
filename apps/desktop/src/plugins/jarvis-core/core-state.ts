/**
 * JARVIS Core — the one place the visual's state is decided, from the coarse
 * read-only signals the SDK exposes (`host.state.busy`,
 * `host.state.awaitingInput`, `host.state.voice`, `host.state.voiceInput`,
 * `host.state.gateway`).
 * Pure, so the precedence is testable without a DOM.
 *
 * Precedence, highest first:
 *  1. dictating / transcribing — the user's own speech is being captured or
 *     turned into text. Shown even over a running turn or playback: a barge-in
 *     is transcribed while the interrupted turn is still busy, and an explicit
 *     push-to-talk must never read as anything but "recording".
 *  2. speaking — reply audio is playing.
 *  3. preparing — reply audio is being synthesized (Piper's 20s+ gap).
 *  4. listening — the voice conversation has the mic open between turns. It
 *     only listens while no turn is running; during generation the barge-in
 *     monitor reports `thinking`, so an enabled conversation alone is never
 *     "listening".
 *  5. offline — the backend link is down (`closed` / `error`). Nothing can
 *     answer or progress, so it outranks needs-input and thinking; a capture or
 *     playback already under way still reads as itself. `idle`/`connecting`
 *     (startup, reconnect) are not offline.
 *  6. needs-input — the turn is blocked on the user (approval, clarify, sudo
 *     or secret prompt). The turn is still "busy" while it waits, so this must
 *     outrank thinking; it sits BELOW the voice states so it never changes how
 *     an in-progress capture or playback reads.
 *  7. thinking — a turn is in flight.
 *  8. idle.
 */
import type { VoiceInputStatus, VoiceStatus } from '@hermes/plugin-sdk'

export type CoreState =
  | 'dictating'
  | 'idle'
  | 'listening'
  | 'needs-input'
  | 'offline'
  | 'preparing'
  | 'speaking'
  | 'thinking'
  | 'transcribing'

export interface CoreSignals {
  awaitingInput: boolean
  busy: boolean
  /** `host.state.gateway`: 'idle' | 'connecting' | 'open' | 'closed' | 'error'. */
  gateway?: string
  voice: VoiceStatus
  voiceInput: VoiceInputStatus
}

export function resolveCoreState({ awaitingInput, busy, gateway, voice, voiceInput }: CoreSignals): CoreState {
  if (voiceInput === 'dictating' || voiceInput === 'transcribing') {
    return voiceInput
  }

  if (voice === 'speaking') {
    return 'speaking'
  }

  if (voice === 'preparing') {
    return 'preparing'
  }

  if (voiceInput === 'listening') {
    return 'listening'
  }

  if (isGatewayDown(gateway)) {
    return 'offline'
  }

  if (awaitingInput) {
    return 'needs-input'
  }

  return busy ? 'thinking' : 'idle'
}

export function isGatewayDown(gateway: string | undefined): boolean {
  return gateway === 'closed' || gateway === 'error'
}

/** Link status key for the gateway state (copy lives in `i18n.ts`). */
export type LinkStatus = 'connecting' | 'offline' | 'online'

export function linkStatus(gateway: string | undefined): LinkStatus {
  if (gateway === 'open') {
    return 'online'
  }

  return isGatewayDown(gateway) ? 'offline' : 'connecting'
}

/** Every state, in precedence order (highest first). */
export const CORE_STATES: readonly CoreState[] = [
  'dictating',
  'transcribing',
  'speaking',
  'preparing',
  'listening',
  'offline',
  'needs-input',
  'thinking',
  'idle'
]

/**
 * The face's visual mood — the design's "one face, four moods"
 * (docs/jarvis-design/references/03-face-voice-states.jpg) plus the two states
 * the design leaves out. The state keeps its own label; only the look is
 * shared:
 *  - standby   — idle: dim cyan, rings at half, slow breath, calm line.
 *  - listening — listening / dictating: full cyan, tick ring follows the mic.
 *  - thinking  — thinking / transcribing / preparing: amber, two orbiting dots,
 *                three-dot ellipsis (speech becoming text and reply audio being
 *                synthesized are both "work in progress").
 *  - speaking  — reply audio playing: ice-cyan, radial bars follow the audio.
 *  - needs-you — blocked on the user: amber, open eyes, insistent pulse.
 *  - offline   — backend link down: the standby face, dimmed and still.
 */
export type CoreMood = 'listening' | 'needs-you' | 'offline' | 'speaking' | 'standby' | 'thinking'

export const CORE_MOODS: readonly CoreMood[] = ['standby', 'listening', 'thinking', 'speaking', 'needs-you', 'offline']

const MOOD_OF: Record<CoreState, CoreMood> = {
  dictating: 'listening',
  idle: 'standby',
  listening: 'listening',
  'needs-input': 'needs-you',
  offline: 'offline',
  preparing: 'thinking',
  speaking: 'speaking',
  thinking: 'thinking',
  transcribing: 'thinking'
}

export function coreMood(state: CoreState): CoreMood {
  return MOOD_OF[state]
}
