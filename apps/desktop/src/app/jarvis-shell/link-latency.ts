import { atom, onMount } from 'nanostores'

import { getStatus } from '@/hermes'

/** How often the top bar's Link pill re-measures the backend round trip. */
export const LINK_LATENCY_POLL_MS = 15_000

/**
 * Round-trip time (ms) of the backend's cheap liveness call (`/api/status`) —
 * the "38 ms" on the JARVIS top bar's Link pill. null until measured or when the
 * last probe failed. Polls only while something subscribes (the top bar is
 * mounted), so the classic shell pays nothing.
 */
export const $linkLatencyMs = atom<null | number>(null)

onMount($linkLatencyMs, () => {
  let stopped = false

  const probe = async () => {
    const started = performance.now()

    try {
      await getStatus()

      if (!stopped) {
        $linkLatencyMs.set(Math.round(performance.now() - started))
      }
    } catch {
      if (!stopped) {
        $linkLatencyMs.set(null)
      }
    }
  }

  void probe()
  const timer = setInterval(() => void probe(), LINK_LATENCY_POLL_MS)

  return () => {
    stopped = true
    clearInterval(timer)
  }
})
