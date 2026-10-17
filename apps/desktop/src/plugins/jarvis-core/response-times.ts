/**
 * Response time for the JARVIS Core card: how long each turn takes, from the
 * moment the agent goes busy (prompt sent) until it settles. Measured from the
 * coarse read-only `host.state.busy` signal only — no message contents, no
 * requests. Kept for this app session (the last RESPONSE_HISTORY turns).
 */
import { atom } from '@hermes/plugin-sdk'

/** The read side of a store (SDK `host.state.*` signals). */
interface Signal<T> {
  get: () => T
  listen: (listener: (value: T) => void) => () => void
}

export const RESPONSE_HISTORY = 20

const $times = atom<readonly number[]>([])

/** Durations (ms) of the most recent turns, oldest first. */
export const $responseTimes = $times

/**
 * Follow a busy signal and record each busy→idle span. Returns the
 * unsubscribe. `now` is injectable for tests.
 */
export function trackResponseTimes(busy: Signal<boolean>, now: () => number = () => Date.now()): () => void {
  let startedAt: null | number = busy.get() ? now() : null

  return busy.listen(isBusy => {
    if (isBusy) {
      startedAt ??= now()

      return
    }

    if (startedAt !== null) {
      const elapsed = Math.max(0, now() - startedAt)
      startedAt = null
      $times.set([...$times.get(), elapsed].slice(-RESPONSE_HISTORY))
    }
  })
}

/** Test helper: forget recorded turns. */
export function resetResponseTimes() {
  $times.set([])
}

/** A turn duration in the app language: 850 → "850 ms", 4200 → "4.2 s",
 *  95000 → "1 min 35 s" (en; Intl unit formatting elsewhere). */
export function formatDuration(ms: number, locale = 'en'): string {
  const unit = (value: number, name: string, digits = 0) =>
    new Intl.NumberFormat(locale, {
      maximumFractionDigits: digits,
      style: 'unit',
      unit: name,
      unitDisplay: 'short'
    }).format(value)

  if (ms < 1000) {
    return unit(Math.round(ms), 'millisecond')
  }

  if (ms < 60_000) {
    return unit(ms / 1000, 'second', ms < 10_000 ? 1 : 0)
  }

  return `${unit(Math.floor(ms / 60_000), 'minute')} ${unit(Math.round((ms % 60_000) / 1000), 'second')}`
}

/** Polyline points for a sparkline of `values` in a width×height box. */
export function sparklinePoints(values: readonly number[], width = 120, height = 28, pad = 2): string {
  if (values.length === 0) {
    return ''
  }

  const max = Math.max(...values, 1)
  const step = values.length > 1 ? width / (values.length - 1) : 0

  return values
    .map((v, i) => {
      const x = values.length > 1 ? i * step : width / 2
      const y = height - pad - (v / max) * (height - pad * 2)

      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
}
