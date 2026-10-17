/**
 * The animated JARVIS face — "one face, four moods" from the design
 * (docs/jarvis-design/references/03-face-voice-states.jpg), geometry after the
 * reference prototype (docs/jarvis-design/prototype/jarvis-ui/jarvis-face.js):
 * a dotted outer ring, a segmented ring, a 60-tick listening ring, two
 * orbiting dots, 36 radial speaking bars, a breathing core and an expression
 * glyph. Purely presentational: the caller resolves the state; this maps it to
 * a mood and paints it. Colours are the face's own fixed palette (see
 * jarvis-core.css), motion is CSS transform/opacity only, and
 * `prefers-reduced-motion` freezes it to a static, mood-tinted mark.
 *
 * `level` (0–1, `host.state.voiceLevel`) is the live loudness of the reply or
 * the user's mic. While it flows the face is "driven": the listening ticks and
 * speaking bars follow it instead of their built-in loops, which return
 * ~400 ms after the level goes quiet (so pauses between words don't flicker).
 */
import { cn } from '@hermes/plugin-sdk'
import { type CSSProperties, useEffect, useId, useState } from 'react'

import { coreMood, type CoreState } from './core-state'

/** Below this the level counts as silence. */
const LEVEL_FLOOR = 0.02
/** How long the face stays level-driven after the audio goes quiet. */
const DRIVEN_HOLD_MS = 400

// Deterministic per-element variety (heights, delays, gain), as in the prototype.
const TICKS = Array.from({ length: 60 }, (_, i) => ({
  angle: i * 6,
  delay: `${((i % 10) * 0.14).toFixed(2)}s`,
  gain: (0.5 + ((i * 53) % 10) / 20).toFixed(2),
  height: 6 + ((i * 37) % 11) * 2
}))

const BARS = Array.from({ length: 36 }, (_, j) => ({
  angle: j * 10,
  delay: `${(((j * 7) % 9) * 0.1).toFixed(2)}s`,
  gain: (0.45 + ((j * 29) % 11) / 20).toFixed(2)
}))

const clampLevel = (level: number) => Math.min(1, Math.max(0, level || 0))

function useLevelDriven(level: number): boolean {
  const live = level > LEVEL_FLOOR
  const [held, setHeld] = useState(false)

  useEffect(() => {
    if (live) {
      setHeld(true)

      return
    }

    const timer = setTimeout(() => setHeld(false), DRIVEN_HOLD_MS)

    return () => clearTimeout(timer)
  }, [live])

  return live || held
}

export function CoreVisual({ className, level = 0, state }: { className?: string; level?: number; state: CoreState }) {
  // Per-instance gradient id: the face can be mounted twice (layout + tab).
  const coreFill = `jarvis-core-fill-${useId().replace(/:/g, '')}`
  const clamped = clampLevel(level)
  const driven = useLevelDriven(clamped)

  return (
    <div
      aria-hidden="true"
      className={cn('jarvis-core', className)}
      data-driven={driven ? '' : undefined}
      data-mood={coreMood(state)}
      data-state={state}
      data-testid="jarvis-core-visual"
      style={{ '--jarvis-level': clamped } as CSSProperties}
    >
      <svg className="jarvis-core__svg" viewBox="0 0 400 400">
        <defs>
          <radialGradient id={coreFill}>
            <stop className="jarvis-core__glow-hot" offset="0%" stopOpacity="0.95" />
            <stop offset="35%" stopColor="currentColor" stopOpacity="0.9" />
            <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
          </radialGradient>
        </defs>

        <g className="jarvis-core__rings">
          <g className="jarvis-core__ring-outer">
            <circle cx="200" cy="200" r="192" />
            <path d="M200 4v14M200 382v14M4 200h14M382 200h14" />
          </g>
          <g className="jarvis-core__ring-segmented">
            <circle cx="200" cy="200" r="166" />
          </g>
        </g>
        <circle className="jarvis-core__guide" cx="200" cy="200" r="138" />

        <g className="jarvis-core__layer jarvis-core__ticks" data-testid="jarvis-core-ticks">
          {TICKS.map(tick => (
            <g key={tick.angle} transform={`rotate(${tick.angle} 200 200)`}>
              <rect
                className="jarvis-core__tick"
                height={tick.height}
                rx="1"
                style={{ animationDelay: tick.delay, '--jarvis-gain': tick.gain } as CSSProperties}
                width="2"
                x="199"
                y={88 - tick.height}
              />
            </g>
          ))}
        </g>

        <g className="jarvis-core__layer jarvis-core__orbits" data-testid="jarvis-core-orbits">
          <g className="jarvis-core__orbit jarvis-core__orbit--a">
            <circle className="jarvis-core__orbit-dot" cx="200" cy="80" r="6" />
          </g>
          <g className="jarvis-core__orbit jarvis-core__orbit--b">
            <circle className="jarvis-core__orbit-dot" cx="200" cy="104" r="4" />
            <path className="jarvis-core__orbit-arc" d="M120 200a80 80 0 0 1 80-80" />
          </g>
        </g>

        <g className="jarvis-core__layer jarvis-core__bars" data-testid="jarvis-core-bars">
          {BARS.map(bar => (
            <g key={bar.angle} transform={`rotate(${bar.angle} 200 200)`}>
              <rect
                className="jarvis-core__bar"
                height="34"
                rx="2"
                style={{ animationDelay: bar.delay, '--jarvis-gain': bar.gain } as CSSProperties}
                width="4"
                x="198"
                y="66"
              />
            </g>
          ))}
        </g>

        <g className="jarvis-core__glow">
          <circle cx="200" cy="200" fill={`url(#${coreFill})`} r="96" />
        </g>
        <circle className="jarvis-core__well" cx="200" cy="200" r="64" />

        <path
          className="jarvis-core__expr jarvis-core__expr--line"
          d="M176 200h48"
          data-testid="jarvis-core-expr-line"
        />
        <g className="jarvis-core__expr jarvis-core__expr--eyes" data-testid="jarvis-core-expr-eyes">
          <circle cx="184" cy="198" r="5" />
          <circle cx="216" cy="198" r="5" />
        </g>
        <g className="jarvis-core__expr jarvis-core__expr--ellipsis" data-testid="jarvis-core-expr-ellipsis">
          <circle cx="184" cy="200" r="4" />
          <circle cx="200" cy="200" opacity="0.7" r="4" />
          <circle cx="216" cy="200" opacity="0.4" r="4" />
        </g>
        <path
          className="jarvis-core__expr jarvis-core__expr--wave"
          d="M174 200h6l4-10 6 20 6-26 6 30 6-22 4 8h12"
          data-testid="jarvis-core-expr-wave"
        />
      </svg>
    </div>
  )
}
