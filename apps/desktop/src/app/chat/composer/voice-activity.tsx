import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { iconSize, Loader2, Mic, Volume2, VolumeX } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { stopVoicePlayback } from '@/lib/voice-playback'
import { publishPlaybackLevel } from '@/store/voice-level'
import { $voicePlayback } from '@/store/voice-playback'

import type { VoiceActivityState } from './types'

type BrowserAudioContext = typeof AudioContext
type CapturableAudioElement = HTMLAudioElement & { captureStream?: () => MediaStream }

// Visualizer-only context. It must never sit in the speech output path: the
// analyser observes a captured copy of the element's audio (captureStream), and
// nothing here connects to `destination`, so the element keeps playing through
// Chromium's own media path. Routing the element through this context with
// createMediaElementSource made speech depend on it — after a WSLg/RDP audio
// device loss the long-lived context stayed "running" against a dead sink and
// every Read Aloud played to completion in silence. Now a broken context can
// only cost the waveform.
let analysisAudioContext: AudioContext | null = null

function getAnalysisAudioContext(): AudioContext | null {
  if (analysisAudioContext && analysisAudioContext.state !== 'closed') {
    return analysisAudioContext
  }

  const audioWindow = window as Window & { webkitAudioContext?: BrowserAudioContext }
  const AudioContextCtor = window.AudioContext || audioWindow.webkitAudioContext

  if (!AudioContextCtor) {
    return null
  }

  try {
    const context = new AudioContextCtor()

    // A device error leaves the context unusable; drop it so the next clip
    // gets a fresh one instead of a permanently blank waveform.
    context.addEventListener('error', () => {
      if (analysisAudioContext === context) {
        analysisAudioContext = null
      }

      void context.close().catch(() => undefined)
    })

    analysisAudioContext = context

    return context
  } catch {
    return null
  }
}

function captureAudio(audioElement: HTMLAudioElement): MediaStream | null {
  const capture = (audioElement as CapturableAudioElement).captureStream

  if (typeof capture !== 'function') {
    return null
  }

  try {
    return capture.call(audioElement)
  } catch {
    return null
  }
}

function formatElapsed(seconds: number) {
  const safeSeconds = Math.max(0, Math.floor(seconds))
  const minutes = Math.floor(safeSeconds / 60)
  const remainingSeconds = safeSeconds % 60

  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
}

function VoiceLevelBars({ level, active }: { active: boolean; level: number }) {
  const normalized = Math.max(0, Math.min(level, 1))
  const bars = [0.5, 0.78, 1, 0.78, 0.5]

  return (
    <div aria-hidden="true" className="flex h-4 items-center gap-0.5">
      {bars.map((weight, index) => {
        const height = active ? 0.25 + Math.min(0.68, normalized * weight) : 0.25

        return (
          <span
            className={cn(
              'w-0.5 rounded-full bg-current transition-[height,opacity] duration-100 ease-out',
              active ? 'opacity-80' : 'animate-pulse opacity-45'
            )}
            key={index}
            style={{ height: `${height * 100}%` }}
          />
        )
      })}
    </div>
  )
}

const WAVE_W = 88
const WAVE_H = 16
const BAR_W = 2
const BAR_GAP = 5
const STEP = BAR_W + BAR_GAP
const BARS = Math.floor((WAVE_W + BAR_GAP) / STEP)
const X0 = Math.round((WAVE_W - (BARS * STEP - BAR_GAP)) / 2)

function PlaybackWaveform({ audioElement }: { audioElement: HTMLAudioElement | null }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [unavailable, setUnavailable] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current

    if (!canvas || !audioElement) {
      return
    }

    const ctx = canvas.getContext('2d')
    const context = ctx ? getAnalysisAudioContext() : null
    const stream = context ? captureAudio(audioElement) : null

    if (!ctx || !context || !stream) {
      setUnavailable(true)

      return
    }

    let disposed = false
    let raf = 0
    let source: MediaStreamAudioSourceNode | null = null
    let analyser: AnalyserNode | null = null

    // A captured stream has no audio track until the element actually starts
    // playing (createMediaStreamSource throws on a track-less stream), and this
    // mounts as soon as playback state flips to speaking — so wire on addtrack.
    const attach = () => {
      if (disposed || source || stream.getAudioTracks().length === 0) {
        return
      }

      try {
        source = context.createMediaStreamSource(stream)
        analyser = context.createAnalyser()
        analyser.fftSize = 512
        analyser.smoothingTimeConstant = 0.65
        // Observe only: an AnalyserNode is pulled without an output connection,
        // so it is deliberately NOT connected to `destination` (that would
        // double the speech, and re-couple it to this context's device).
        source.connect(analyser)
      } catch {
        source?.disconnect()
        source = null
        analyser = null
        setUnavailable(true)

        return
      }

      void context.resume().catch(() => undefined)
      draw(analyser)
    }

    const draw = (node: AnalyserNode) => {
      const dpr = Math.max(1, window.devicePixelRatio || 1)
      const buf = new Uint8Array(node.frequencyBinCount)
      const hi = Math.floor(buf.length * 0.9)

      canvas.width = Math.round(WAVE_W * dpr)
      canvas.height = Math.round(WAVE_H * dpr)
      canvas.style.width = `${WAVE_W}px`
      canvas.style.height = `${WAVE_H}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.imageSmoothingEnabled = false
      ctx.fillStyle = getComputedStyle(canvas).color

      const tick = () => {
        node.getByteFrequencyData(buf)
        ctx.clearRect(0, 0, WAVE_W, WAVE_H)

        let ampSum = 0

        for (let i = 0; i < BARS; i++) {
          const a = Math.floor((i / BARS) * hi)
          const b = Math.floor(((i + 1) / BARS) * hi)
          let peak = 0

          for (let j = a; j < b; j++) {
            peak = Math.max(peak, buf[j] ?? 0)
          }

          const amp = Math.sqrt(peak / 255)
          ampSum += amp
          const bh = Math.max(3, Math.round((0.18 + amp * 0.82) * WAVE_H))
          ctx.fillRect(X0 + i * STEP, Math.round((WAVE_H - bh) / 2), BAR_W, bh)
        }

        // Mirror the same reading for the JARVIS face (store-only, throttled).
        publishPlaybackLevel(ampSum / BARS)
        raf = requestAnimationFrame(tick)
      }

      tick()
    }

    stream.addEventListener('addtrack', attach)
    attach()

    return () => {
      disposed = true
      stream.removeEventListener('addtrack', attach)
      cancelAnimationFrame(raf)
      publishPlaybackLevel(0)
      source?.disconnect()
      analyser?.disconnect()
      // Stops only the captured copy; the element's own output is unaffected.
      stream.getTracks().forEach(track => track.stop())
    }
  }, [audioElement])

  if (unavailable) {
    return <VoiceLevelBars active={false} level={0} />
  }

  return <canvas aria-hidden="true" className="block h-4 w-[88px]" ref={canvasRef} />
}

export function VoiceActivity({ state }: { state: VoiceActivityState }) {
  const { t } = useI18n()

  if (state.status === 'idle') {
    return null
  }

  const recording = state.status === 'recording'
  const title = recording ? t.composer.dictating : t.composer.transcribing

  return (
    <div
      aria-live="polite"
      className={cn(
        'flex h-8 items-center gap-2 rounded-xl border border-border/55 bg-muted/55 px-2.5 text-xs text-muted-foreground',
        'shadow-[inset_0_1px_0_rgba(255,255,255,0.35)] backdrop-blur-sm'
      )}
      role="status"
    >
      <div
        className={cn(
          'flex size-5 shrink-0 items-center justify-center rounded-full',
          recording ? 'bg-primary/15 text-primary' : 'bg-primary/10 text-primary'
        )}
      >
        {recording ? <Mic className={iconSize.xs} /> : <Loader2 className={cn('animate-spin', iconSize.xs)} />}
      </div>

      <div className="flex min-w-0 flex-1 items-center gap-2">
        <span className="truncate font-medium text-foreground/85">{title}</span>
        <span className="font-mono text-[0.6875rem] text-muted-foreground/85">
          {formatElapsed(state.elapsedSeconds)}
        </span>
      </div>

      <VoiceLevelBars active={recording} level={state.level} />
    </div>
  )
}

export function VoicePlaybackActivity() {
  const { t } = useI18n()
  const playback = useStore($voicePlayback)

  if (playback.status === 'idle') {
    return null
  }

  const preparing = playback.status === 'preparing'

  const title = preparing
    ? t.composer.preparingAudio
    : playback.source === 'voice-conversation'
      ? t.composer.speakingResponse
      : t.composer.readingAloud

  return (
    <div
      aria-live="polite"
      className={cn(
        'flex h-8 items-center gap-2 rounded-xl border border-primary/20 bg-primary/10 px-2.5 text-xs text-primary',
        'shadow-[inset_0_1px_0_rgba(255,255,255,0.35)] backdrop-blur-sm'
      )}
      role="status"
    >
      <div className="flex size-5 shrink-0 items-center justify-center rounded-full bg-primary/15 text-primary">
        {preparing ? <Loader2 className={cn('animate-spin', iconSize.xs)} /> : <Volume2 className={iconSize.xs} />}
      </div>

      <div className="flex min-w-0 flex-1 items-center gap-2">
        <span className="truncate font-medium text-foreground/85">{title}</span>
        {!preparing && <PlaybackWaveform audioElement={playback.audioElement} key={playback.sequence} />}
      </div>

      <Button
        className="h-6 shrink-0 gap-1 rounded-full px-2 text-[0.6875rem]"
        onClick={stopVoicePlayback}
        size="sm"
        type="button"
        variant="ghost"
      >
        <VolumeX className={iconSize.xs} />
        Stop
      </Button>
    </div>
  )
}
