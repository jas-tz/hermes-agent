import type { WritableAtom } from 'nanostores'
import { afterEach, describe, expect, it } from 'vitest'

import { $activePresetId } from '@/components/pane-shell/tree/store'
import { $approvalRequest, clearAllPrompts, setApprovalRequest, setSudoRequest } from '@/store/prompts'
import { $awaitingResponse, $busy } from '@/store/session'
import { $voiceInput, publishVoiceInput, releaseVoiceInput } from '@/store/voice-input'
import { $voicePlayback, setVoicePlaybackState, type VoicePlaybackState } from '@/store/voice-playback'

import { applyLayoutPreset, host } from './index'

const IDLE_PLAYBACK: VoicePlaybackState = {
  audioElement: null,
  messageId: null,
  sequence: 0,
  source: null,
  status: 'idle'
}

afterEach(() => {
  $busy.set(false)
  $awaitingResponse.set(false)
  setVoicePlaybackState(IDLE_PLAYBACK)
  $voiceInput.set('idle')
  clearAllPrompts()
})

describe('host.state.busy', () => {
  it('is true while a turn is awaiting a response or running', () => {
    expect(host.state.busy.get()).toBe(false)

    $awaitingResponse.set(true)
    expect(host.state.busy.get()).toBe(true)

    $awaitingResponse.set(false)
    $busy.set(true)
    expect(host.state.busy.get()).toBe(true)

    $busy.set(false)
    expect(host.state.busy.get()).toBe(false)
  })

  it('notifies subscribers on change', () => {
    const seen: boolean[] = []
    const stop = host.state.busy.subscribe(value => seen.push(value))

    $busy.set(true)
    $busy.set(false)
    stop()

    expect(seen).toEqual([false, true, false])
  })
})

describe('host.state.voice', () => {
  it('exposes only the playback status, never the audio element or message id', () => {
    const audioElement = {} as HTMLAudioElement

    setVoicePlaybackState({
      audioElement,
      messageId: 'msg-1',
      sequence: 3,
      source: 'read-aloud',
      status: 'speaking'
    })

    expect(host.state.voice.get()).toBe('speaking')

    setVoicePlaybackState({ ...IDLE_PLAYBACK, status: 'preparing' })
    expect(host.state.voice.get()).toBe('preparing')
  })
})

describe('host.state.voiceInput', () => {
  it('follows the main composer mic status published to the store', () => {
    const composer = Symbol('main-composer')
    const seen: string[] = []
    const stop = host.state.voiceInput.subscribe(value => seen.push(value))

    publishVoiceInput(composer, 'listening')
    publishVoiceInput(composer, 'transcribing')
    releaseVoiceInput(composer)
    stop()

    expect(seen).toEqual(['idle', 'listening', 'transcribing', 'idle'])
  })
})

describe('host.state.layoutPreset', () => {
  afterEach(() => $activePresetId.set('default'))

  it('follows the active layout preset, including one applied through the SDK', () => {
    expect(host.state.layoutPreset.get()).toBe($activePresetId.get())

    applyLayoutPreset('jarvis-core:layout', {
      type: 'split',
      id: 'spl',
      orientation: 'row',
      children: [{ type: 'group', id: 'grp', panes: ['workspace'], active: 'workspace' }],
      weights: [1]
    })

    expect(host.state.layoutPreset.get()).toBe('jarvis-core:layout')
  })
})

describe('host.state.awaitingInput', () => {
  it('is true while the active session has a blocking prompt, and exposes only a boolean', () => {
    expect(host.state.awaitingInput.get()).toBe(false)

    setApprovalRequest({ command: 'rm -rf /tmp/x', description: 'dangerous', sessionId: null })
    expect(host.state.awaitingInput.get()).toBe(true)

    clearAllPrompts(null)
    expect(host.state.awaitingInput.get()).toBe(false)

    setSudoRequest({ requestId: 'r1', sessionId: null })
    expect(host.state.awaitingInput.get()).toBe(true)
  })

  it('ignores a prompt parked on a background session', () => {
    setApprovalRequest({ command: 'ls', description: 'bg', sessionId: 'background-session' })

    expect(host.state.awaitingInput.get()).toBe(false)
  })
})

describe('host.state derived signals are write-isolated', () => {
  it('a cast-away write cannot reach the app voice or turn state', () => {
    ;(host.state.voice as unknown as WritableAtom<string>).set('speaking')
    ;(host.state.busy as unknown as WritableAtom<boolean>).set(true)
    ;(host.state.voiceInput as unknown as WritableAtom<string>).set('listening')
    ;(host.state.awaitingInput as unknown as WritableAtom<boolean>).set(true)

    expect($voicePlayback.get().status).toBe('idle')
    expect($busy.get()).toBe(false)
    expect($awaitingResponse.get()).toBe(false)
    expect($voiceInput.get()).toBe('idle')
    // Forcing the derived flag neither raises nor answers an approval.
    expect($approvalRequest.get()).toBeNull()
  })
})
