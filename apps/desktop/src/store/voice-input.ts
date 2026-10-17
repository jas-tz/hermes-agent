import { atom } from 'nanostores'

/**
 * What the MAIN composer's mic is doing right now — a read-only mirror of the
 * composer-local voice state (dictation recorder + voice-conversation loop) so
 * surfaces outside the composer can show it. Mirror only: nothing here opens a
 * mic or drives the pipeline.
 *
 *   listening    — voice conversation has the mic open, waiting for speech
 *   dictating    — push-to-talk dictation is recording
 *   transcribing — captured audio is being turned into text (either mode)
 */
export type VoiceInputStatus = 'dictating' | 'idle' | 'listening' | 'transcribing'

export const $voiceInput = atom<VoiceInputStatus>('idle')

/**
 * Map the composer's two voice sources onto one input status. Dictation wins
 * when it is active (it is the explicit gesture); the conversation counts only
 * while it is switched on, and only its mic-side phases — its thinking /
 * speaking phases are already covered by the turn and playback signals.
 */
export function resolveVoiceInput(
  dictation: 'idle' | 'recording' | 'transcribing',
  conversationActive: boolean,
  conversation: 'idle' | 'listening' | 'speaking' | 'thinking' | 'transcribing'
): VoiceInputStatus {
  if (dictation === 'recording') {
    return 'dictating'
  }

  if (dictation === 'transcribing') {
    return 'transcribing'
  }

  if (conversationActive && (conversation === 'listening' || conversation === 'transcribing')) {
    return conversation
  }

  return 'idle'
}

// Ownership token: the composer that published last owns the value. A stale
// owner (e.g. an old main composer whose unmount cleanup lands after its
// replacement already published) can never reset the live one's status.
let owner: symbol | null = null

export function publishVoiceInput(token: symbol, status: VoiceInputStatus): void {
  owner = token
  $voiceInput.set(status)
}

/** Unmount cleanup: back to idle, but only if `token` still owns the value. */
export function releaseVoiceInput(token: symbol): void {
  if (owner !== token) {
    return
  }

  owner = null
  $voiceInput.set('idle')
}
