import { afterEach, describe, expect, it } from 'vitest'

import { $voiceInput, publishVoiceInput, releaseVoiceInput, resolveVoiceInput } from './voice-input'

afterEach(() => {
  // Leave no owner behind for the next test.
  const reset = Symbol('reset')
  publishVoiceInput(reset, 'idle')
  releaseVoiceInput(reset)
})

describe('resolveVoiceInput', () => {
  it('is idle when neither source has the mic', () => {
    expect(resolveVoiceInput('idle', false, 'idle')).toBe('idle')
    expect(resolveVoiceInput('idle', true, 'idle')).toBe('idle')
  })

  it('maps push-to-talk dictation', () => {
    expect(resolveVoiceInput('recording', false, 'idle')).toBe('dictating')
    expect(resolveVoiceInput('transcribing', false, 'idle')).toBe('transcribing')
  })

  it('maps the conversation mic phases only while the conversation is on', () => {
    expect(resolveVoiceInput('idle', true, 'listening')).toBe('listening')
    expect(resolveVoiceInput('idle', true, 'transcribing')).toBe('transcribing')
    // A lingering status after the conversation was switched off is not input.
    expect(resolveVoiceInput('idle', false, 'listening')).toBe('idle')
  })

  it('leaves thinking/speaking to the turn and playback signals', () => {
    expect(resolveVoiceInput('idle', true, 'thinking')).toBe('idle')
    expect(resolveVoiceInput('idle', true, 'speaking')).toBe('idle')
  })

  it('lets active dictation win over the conversation', () => {
    expect(resolveVoiceInput('recording', true, 'listening')).toBe('dictating')
  })
})

describe('publish / release ownership', () => {
  it('publishes the latest status', () => {
    const token = Symbol('a')

    publishVoiceInput(token, 'listening')
    expect($voiceInput.get()).toBe('listening')

    publishVoiceInput(token, 'transcribing')
    expect($voiceInput.get()).toBe('transcribing')
  })

  it('resets to idle when the owner releases', () => {
    const token = Symbol('a')

    publishVoiceInput(token, 'dictating')
    releaseVoiceInput(token)

    expect($voiceInput.get()).toBe('idle')
  })

  it('ignores a release from a stale owner', () => {
    const old = Symbol('old')
    const current = Symbol('current')

    publishVoiceInput(old, 'idle')
    publishVoiceInput(current, 'listening')
    releaseVoiceInput(old)

    expect($voiceInput.get()).toBe('listening')

    releaseVoiceInput(current)
    expect($voiceInput.get()).toBe('idle')
  })
})
