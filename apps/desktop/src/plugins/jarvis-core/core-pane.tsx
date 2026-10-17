/**
 * JARVIS Core pane — the face. Centre stage in the JARVIS layout (sessions |
 * face | chat), or a tab in the chat zone elsewhere. Never a replacement for
 * the chat: ChatView (and with it the mic, voice-conversation and read-aloud
 * hooks) stays mounted beside or behind it.
 *
 * Read-only by construction: it subscribes to coarse SDK signals (turn, prompt,
 * voice, mic and gateway statuses plus a 0–1 voice level) and renders them. No
 * requests, no controls, no access to messages, audio or prompt contents — a
 * pending approval is answered in the chat, never here.
 *
 * Layout follows the design's centre stage
 * (docs/jarvis-design/references/01-main-screen.jpg): face, mood label in the
 * mood's colour, a one-line caption, then Agent / Microphone / Voice / Link
 * cards (four across, two across in a narrow zone). The pane is a SIZE
 * container, so the face scales with the zone in both axes and the cards
 * reflow by width; the column centres when there is room and scrolls when
 * there isn't. Styling lives in jarvis-core.css.
 */
import { host, Slot, useI18n, useValue } from '@hermes/plugin-sdk'

import { coreMood, linkStatus, resolveCoreState } from './core-state'
import { CoreVisual } from './core-visual'
import { useJarvisCoreText } from './i18n'
import { $responseTimes, formatDuration, sparklinePoints } from './response-times'

/** Area other plugins contribute stage widgets to (the JARVIS Memory plugin's
 *  Active tasks bar), rendered under the status cards. */
export const JARVIS_STAGE_AREA = 'jarvis-core.stage'

/** One status card: eyebrow label, current value, and a quieter sub-line. */
function StatusCard({ id, label, sub, value }: { id: string; label: string; sub: string; value: string }) {
  return (
    <div className="jarvis-core-card" data-testid={`jarvis-core-card-${id}`}>
      <dt className="jarvis-core-card__label">{label}</dt>
      <dd className="jarvis-core-card__value">{value}</dd>
      <dd className="jarvis-core-card__sub">{sub}</dd>
    </div>
  )
}

/** Response time, as in the design: a sparkline of recent turn times in the
 *  value row, the latest time and how many turns in the sub-line. */
function ResponseTimeCard({ label, last, none }: { label: string; last: (count: number) => string; none: string }) {
  const { locale } = useI18n()
  const times = useValue($responseTimes)
  const latest = times.at(-1)

  return (
    <div className="jarvis-core-card" data-testid="jarvis-core-card-response-time">
      <dt className="jarvis-core-card__label">{label}</dt>
      <dd className="jarvis-core-card__value jarvis-core-card__spark">
        {latest === undefined ? (
          '—'
        ) : (
          <svg aria-hidden="true" preserveAspectRatio="none" viewBox="0 0 120 28">
            <polyline data-testid="jarvis-core-sparkline" points={sparklinePoints(times)} />
          </svg>
        )}
      </dd>
      <dd className="jarvis-core-card__sub">
        {latest === undefined ? none : `${formatDuration(latest, locale)} · ${last(times.length)}`}
      </dd>
    </div>
  )
}

export function JarvisCorePane() {
  const awaitingInput = useValue(host.state.awaitingInput)
  const busy = useValue(host.state.busy)
  const gateway = useValue(host.state.gateway)
  const voice = useValue(host.state.voice)
  const voiceInput = useValue(host.state.voiceInput)
  const voiceLevel = useValue(host.state.voiceLevel)
  const state = resolveCoreState({ awaitingInput, busy, gateway, voice, voiceInput })
  const mood = coreMood(state)
  const text = useJarvisCoreText()
  const agent = awaitingInput ? 'waiting' : busy ? 'working' : 'idle'
  const link = linkStatus(gateway)

  return (
    <div
      className="jarvis-core-pane h-full min-h-0 overflow-y-auto [container-type:size]"
      data-testid="jarvis-core-pane"
    >
      <div className="jarvis-core-pane__stage">
        <div className="jarvis-core-pane__face-slot">
          <CoreVisual className="jarvis-core-pane__face" level={voiceLevel} state={state} />
        </div>
        <p
          aria-live="polite"
          className="jarvis-core-pane__status"
          data-mood={mood}
          data-testid="jarvis-core-state"
          role="status"
        >
          {text.states[state]}
        </p>
        <p className="jarvis-core-pane__caption" data-testid="jarvis-core-caption">
          {text.captions[state]}
        </p>
        <dl className="jarvis-core-pane__cards" data-testid="jarvis-core-cards">
          <StatusCard id="agent" label={text.rows.agent} sub={text.cardSub.agent[agent]} value={text.agent[agent]} />
          <StatusCard
            id="microphone"
            label={text.rows.microphone}
            sub={text.cardSub.microphone[voiceInput]}
            value={text.voiceInput[voiceInput]}
          />
          <StatusCard id="voice" label={text.rows.voice} sub={text.cardSub.voice[voice]} value={text.voice[voice]} />
          <StatusCard id="link" label={text.rows.link} sub={text.cardSub.link[link]} value={text.link[link]} />
          <ResponseTimeCard
            label={text.rows.responseTime}
            last={text.cardSub.responseTime.last}
            none={text.cardSub.responseTime.none}
          />
        </dl>
        <div className="jarvis-core-pane__widgets" data-testid="jarvis-core-stage-slot">
          <Slot area={JARVIS_STAGE_AREA} />
        </div>
      </div>
    </div>
  )
}
