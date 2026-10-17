import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $voiceInput } from '@/store/voice-input'
import { $micLevel, publishMicLevel } from '@/store/voice-level'

import { requestVoiceToggle } from '../focus'

import { useComposerVoice } from './use-composer-voice'

// The two real voice sources are composer-local hooks; stub them with settable
// statuses so this test exercises only the mirror into $voiceInput — no mic,
// no conversation loop, no TTS.
const voice = vi.hoisted(() => ({
  conversation: 'idle' as 'idle' | 'listening' | 'speaking' | 'thinking' | 'transcribing',
  conversationLevel: 0,
  dictation: 'idle' as 'idle' | 'recording' | 'transcribing',
  dictationLevel: 0
}))

vi.mock('./use-voice-recorder', () => ({
  useVoiceRecorder: () => ({
    dictate: vi.fn(),
    voiceActivityState: { elapsedSeconds: 0, level: voice.dictationLevel, status: voice.dictation },
    voiceStatus: voice.dictation
  })
}))

vi.mock('./use-voice-conversation', () => ({
  useVoiceConversation: () => ({
    end: vi.fn(async () => undefined),
    level: voice.conversationLevel,
    muted: false,
    start: vi.fn(async () => undefined),
    status: voice.conversation,
    stopTurn: vi.fn(),
    toggleMute: vi.fn()
  })
}))

vi.mock('./use-auto-speak-replies', () => ({ useAutoSpeakReplies: () => undefined }))

function renderVoice(target: string) {
  return renderHook(({ target }) =>
    useComposerVoice({
      busy: false,
      clearDraft: vi.fn(),
      disabled: false,
      focusInput: vi.fn(),
      insertText: vi.fn(),
      maxRecordingSeconds: 60,
      onSubmit: vi.fn(),
      onTranscribeAudio: vi.fn(),
      sessionId: 'session-1',
      target
    }), { initialProps: { target } }
  )
}

// The focus bus dispatches on a 0ms timer — let it land inside act().
async function toggleMainVoice() {
  await act(async () => {
    requestVoiceToggle('main')
    await new Promise(resolve => setTimeout(resolve, 0))
  })
}

let fakeNow = 1_000_000

beforeEach(() => {
  voice.conversation = 'idle'
  voice.conversationLevel = 0
  voice.dictation = 'idle'
  voice.dictationLevel = 0
  // The mic-level store throttles on performance.now(); keep calls far apart
  // (and monotonic across tests — the store remembers its last publish time).
  vi.spyOn(performance, 'now').mockImplementation(() => (fakeNow += 1_000))
})

afterEach(() => {
  cleanup()
  $voiceInput.set('idle')
  publishMicLevel(0)
  vi.restoreAllMocks()
})

describe('useComposerVoice → $voiceInput mirror', () => {
  it('publishes the main composer dictation phases', () => {
    const { rerender } = renderVoice('main')
    expect($voiceInput.get()).toBe('idle')

    voice.dictation = 'recording'
    rerender({ target: 'main' })
    expect($voiceInput.get()).toBe('dictating')

    voice.dictation = 'transcribing'
    rerender({ target: 'main' })
    expect($voiceInput.get()).toBe('transcribing')

    voice.dictation = 'idle'
    rerender({ target: 'main' })
    expect($voiceInput.get()).toBe('idle')
  })

  it('publishes conversation listening only while the conversation is on', async () => {
    voice.conversation = 'listening'
    const { rerender } = renderVoice('main')

    // Status alone is not enough — the conversation is still switched off.
    expect($voiceInput.get()).toBe('idle')

    await toggleMainVoice()
    expect($voiceInput.get()).toBe('listening')

    voice.conversation = 'transcribing'
    rerender({ target: 'main' })
    expect($voiceInput.get()).toBe('transcribing')

    voice.conversation = 'speaking'
    rerender({ target: 'main' })
    expect($voiceInput.get()).toBe('idle')

    await toggleMainVoice()
    voice.conversation = 'listening'
    rerender({ target: 'main' })
    expect($voiceInput.get()).toBe('idle')
  })

  it('never lets a tile composer write the shared status', () => {
    voice.dictation = 'recording'
    renderVoice('tile:abc')

    expect($voiceInput.get()).toBe('idle')
  })

  it('a tile unmounting does not reset the main composer status', () => {
    voice.dictation = 'recording'
    renderVoice('main')
    const tile = renderVoice('tile:abc')
    expect($voiceInput.get()).toBe('dictating')

    tile.unmount()
    expect($voiceInput.get()).toBe('dictating')
  })

  it('resets to idle when the main composer unmounts', () => {
    voice.dictation = 'recording'
    const main = renderVoice('main')
    expect($voiceInput.get()).toBe('dictating')

    main.unmount()
    expect($voiceInput.get()).toBe('idle')
  })

  it('a replaced main composer unmounting late does not clobber the new one', () => {
    const previous = renderVoice('main')

    voice.dictation = 'recording'
    renderVoice('main')
    expect($voiceInput.get()).toBe('dictating')

    previous.unmount()
    expect($voiceInput.get()).toBe('dictating')
  })
})

describe('useComposerVoice → $micLevel mirror', () => {
  it('publishes the dictation meter while recording, then drops to zero', () => {
    voice.dictation = 'recording'
    voice.dictationLevel = 0.4
    const { rerender } = renderVoice('main')
    expect($micLevel.get()).toBe(0.4)

    voice.dictation = 'transcribing'
    rerender({ target: 'main' })
    expect($micLevel.get()).toBe(0)
  })

  it('publishes the conversation meter only while it is listening', async () => {
    voice.conversation = 'listening'
    voice.conversationLevel = 0.6
    const { rerender } = renderVoice('main')

    // Switched off: no live mic, no level.
    expect($micLevel.get()).toBe(0)

    await toggleMainVoice()
    expect($micLevel.get()).toBe(0.6)

    voice.conversation = 'thinking'
    rerender({ target: 'main' })
    expect($micLevel.get()).toBe(0)
  })

  it('never lets a tile composer write the shared level', () => {
    voice.dictation = 'recording'
    voice.dictationLevel = 0.8
    renderVoice('tile:abc')

    expect($micLevel.get()).toBe(0)
  })

  it('resets to zero when the main composer unmounts', () => {
    voice.dictation = 'recording'
    voice.dictationLevel = 0.5
    const { unmount } = renderVoice('main')
    expect($micLevel.get()).toBe(0.5)

    unmount()
    expect($micLevel.get()).toBe(0)
  })
})
