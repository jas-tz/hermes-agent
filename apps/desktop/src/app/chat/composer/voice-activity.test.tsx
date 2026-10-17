import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $playbackLevel } from '@/store/voice-level'
import { setVoicePlaybackState } from '@/store/voice-playback'

import { VoicePlaybackActivity } from './voice-activity'

class FakeStream extends EventTarget {
  tracks: { kind: string; stop: ReturnType<typeof vi.fn> }[] = []

  getAudioTracks = () => this.tracks.filter(track => track.kind === 'audio')
  getTracks = () => this.tracks

  addAudioTrack() {
    const track = { kind: 'audio', stop: vi.fn() }
    this.tracks.push(track)
    this.dispatchEvent(new Event('addtrack'))

    return track
  }
}

function makeAnalyser() {
  return {
    connect: vi.fn(),
    disconnect: vi.fn(),
    fftSize: 0,
    frequencyBinCount: 256,
    getByteFrequencyData: vi.fn(),
    smoothingTimeConstant: 0
  }
}

const contexts: FakeAudioContext[] = []

class FakeAudioContext extends EventTarget {
  static failStreamSource = false

  analyser = makeAnalyser()
  destination = { kind: 'destination' }
  source = { connect: vi.fn(), disconnect: vi.fn() }
  state: AudioContextState = 'running'

  close = vi.fn(async () => {
    this.state = 'closed'
  })
  createAnalyser = vi.fn(() => this.analyser)
  createMediaElementSource = vi.fn()
  createMediaStreamSource = vi.fn((_stream: MediaStream) => {
    if (FakeAudioContext.failStreamSource) {
      throw new DOMException('MediaStream has no audio track', 'InvalidStateError')
    }

    return this.source
  })
  resume = vi.fn(async () => undefined)

  constructor() {
    super()
    contexts.push(this)
  }
}

function makeAudio(stream: FakeStream | null) {
  const audio = new Audio()

  if (stream) {
    Object.assign(audio, { captureStream: vi.fn(() => stream) })
  }

  return audio
}

let sequence = 0

function speak(audioElement: HTMLAudioElement) {
  sequence += 1
  act(() =>
    setVoicePlaybackState({ audioElement, messageId: 'm1', sequence, source: 'read-aloud', status: 'speaking' })
  )
}

function stopSpeaking() {
  sequence += 1
  act(() => setVoicePlaybackState({ audioElement: null, messageId: null, sequence, source: null, status: 'idle' }))
}

function renderActivity() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <VoicePlaybackActivity />
    </I18nProvider>
  )
}

const rafCallbacks = new Map<number, FrameRequestCallback>()
let nextRaf = 0

beforeEach(() => {
  rafCallbacks.clear()
  vi.stubGlobal('AudioContext', FakeAudioContext)
  vi.stubGlobal(
    'requestAnimationFrame',
    vi.fn((callback: FrameRequestCallback) => {
      nextRaf += 1
      rafCallbacks.set(nextRaf, callback)

      return nextRaf
    })
  )
  vi.stubGlobal(
    'cancelAnimationFrame',
    vi.fn((id: number) => rafCallbacks.delete(id))
  )
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    setTransform: vi.fn()
  } as unknown as CanvasRenderingContext2D)
})

afterEach(() => {
  stopSpeaking()
  cleanup()
  // The visualizer caches its context module-wide; retire each fake so the
  // next test starts from a fresh one.
  contexts.splice(0).forEach(context => (context.state = 'closed'))
  FakeAudioContext.failStreamSource = false
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('VoicePlaybackActivity waveform', () => {
  it('observes a captured copy of the speech and never routes the element through WebAudio output', () => {
    const stream = new FakeStream()
    stream.addAudioTrack()
    const audio = makeAudio(stream)

    const { container } = renderActivity()
    speak(audio)

    const context = contexts.at(-1)!

    expect(container.querySelector('canvas')).not.toBeNull()
    expect(context.createMediaElementSource).not.toHaveBeenCalled()
    expect(context.createMediaStreamSource).toHaveBeenCalledWith(stream)
    expect(context.source.connect).toHaveBeenCalledWith(context.analyser)
    // Observe-only: nothing reaches the context's speakers, so a dead device
    // behind this context cannot silence (or double) the speech.
    expect(context.analyser.connect).not.toHaveBeenCalled()
  })

  it('keeps the analyser updating on each animation frame', () => {
    const stream = new FakeStream()
    stream.addAudioTrack()

    renderActivity()
    speak(makeAudio(stream))

    const { analyser } = contexts.at(-1)!

    expect(analyser.getByteFrequencyData).toHaveBeenCalledTimes(1)

    act(() => {
      const [[id, frame]] = [...rafCallbacks]
      rafCallbacks.delete(id)
      frame(16)
    })

    expect(analyser.getByteFrequencyData).toHaveBeenCalledTimes(2)
  })

  it('waits for the captured audio track before wiring the analyser', () => {
    const stream = new FakeStream()

    renderActivity()
    speak(makeAudio(stream))

    const context = contexts.at(-1)!

    // createMediaStreamSource throws on a track-less stream in Chromium.
    expect(context.createMediaStreamSource).not.toHaveBeenCalled()

    act(() => {
      stream.addAudioTrack()
    })

    expect(context.createMediaStreamSource).toHaveBeenCalledTimes(1)
    expect(context.source.connect).toHaveBeenCalledWith(context.analyser)
  })

  it('falls back to static bars when captureStream is unavailable', () => {
    const { container } = renderActivity()
    speak(makeAudio(null))

    expect(container.querySelector('canvas')).toBeNull()
    expect(container.querySelectorAll('span.animate-pulse')).toHaveLength(5)
    expect(contexts.at(-1)!.createMediaStreamSource).not.toHaveBeenCalled()
  })

  it('falls back to static bars without touching the element when AudioContext is unavailable', () => {
    vi.stubGlobal('AudioContext', undefined)
    const stream = new FakeStream()
    stream.addAudioTrack()
    const audio = makeAudio(stream)

    const { container } = renderActivity()
    speak(audio)

    expect(container.querySelector('canvas')).toBeNull()
    expect(container.querySelectorAll('span.animate-pulse')).toHaveLength(5)
    expect((audio as HTMLAudioElement & { captureStream: () => MediaStream }).captureStream).not.toHaveBeenCalled()
  })

  it('falls back to static bars when analysis setup throws', () => {
    FakeAudioContext.failStreamSource = true
    const stream = new FakeStream()
    stream.addAudioTrack()

    const { container } = renderActivity()
    speak(makeAudio(stream))

    expect(contexts.at(-1)!.createMediaStreamSource).toHaveBeenCalled()
    expect(container.querySelector('canvas')).toBeNull()
    expect(container.querySelectorAll('span.animate-pulse')).toHaveLength(5)
  })

  it('cancels the animation frame, disconnects the tap and stops the captured copy when playback ends', () => {
    const stream = new FakeStream()
    const track = stream.addAudioTrack()

    renderActivity()
    speak(makeAudio(stream))

    const context = contexts.at(-1)!
    const [pendingFrame] = [...rafCallbacks.keys()]

    stopSpeaking()

    expect(cancelAnimationFrame).toHaveBeenCalledWith(pendingFrame)
    expect(rafCallbacks.size).toBe(0)
    expect(context.source.disconnect).toHaveBeenCalled()
    expect(context.analyser.disconnect).toHaveBeenCalled()
    expect(track.stop).toHaveBeenCalled()
  })

  it('mirrors the analyser reading into the playback level and resets it when playback ends', () => {
    // The level store throttles on performance.now(); keep calls far apart.
    let now = 1_000_000
    vi.spyOn(performance, 'now').mockImplementation(() => (now += 1_000))

    const stream = new FakeStream()
    stream.addAudioTrack()

    renderActivity()
    speak(makeAudio(stream))

    const { analyser } = contexts.at(-1)!

    // Full-scale spectrum on the next frame → full level.
    analyser.getByteFrequencyData.mockImplementation((data: Uint8Array) => data.fill(255))
    act(() => {
      const [[id, frame]] = [...rafCallbacks]
      rafCallbacks.delete(id)
      frame(16)
    })

    expect($playbackLevel.get()).toBe(1)

    stopSpeaking()

    expect($playbackLevel.get()).toBe(0)
  })

  it('replaces a context that reported a device error', () => {
    const stream = new FakeStream()
    stream.addAudioTrack()

    renderActivity()
    speak(makeAudio(stream))

    const broken = contexts.at(-1)!

    act(() => {
      broken.dispatchEvent(new Event('error'))
    })

    expect(broken.close).toHaveBeenCalled()

    const next = new FakeStream()
    next.addAudioTrack()
    speak(makeAudio(next))

    const fresh = contexts.at(-1)!

    expect(fresh).not.toBe(broken)
    expect(fresh.createMediaStreamSource).toHaveBeenCalledWith(next)
  })
})
