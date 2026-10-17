import { atom } from '@hermes/plugin-sdk'
import { afterEach, describe, expect, it } from 'vitest'

import {
  $responseTimes,
  formatDuration,
  resetResponseTimes,
  RESPONSE_HISTORY,
  sparklinePoints,
  trackResponseTimes
} from './response-times'

afterEach(resetResponseTimes)

describe('trackResponseTimes', () => {
  it('records how long each turn stays busy', () => {
    const busy = atom(false)
    let clock = 1000
    const stop = trackResponseTimes(busy, () => clock)

    busy.set(true)
    clock = 5200
    busy.set(false)
    busy.set(true)
    clock = 6050
    busy.set(false)

    expect($responseTimes.get()).toEqual([4200, 850])
    stop()
  })

  it('measures a turn already running when tracking starts', () => {
    const busy = atom(true)
    let clock = 0
    const stop = trackResponseTimes(busy, () => clock)

    clock = 3000
    busy.set(false)

    expect($responseTimes.get()).toEqual([3000])
    stop()
  })

  it('keeps only the recent history', () => {
    const busy = atom(false)
    let clock = 0
    const stop = trackResponseTimes(busy, () => clock)

    for (let i = 1; i <= RESPONSE_HISTORY + 5; i++) {
      busy.set(true)
      clock += i
      busy.set(false)
    }

    expect($responseTimes.get()).toHaveLength(RESPONSE_HISTORY)
    expect($responseTimes.get().at(-1)).toBe(RESPONSE_HISTORY + 5)
    stop()
  })

  it('stops recording once unsubscribed', () => {
    const busy = atom(false)
    const stop = trackResponseTimes(busy, () => 0)

    stop()
    busy.set(true)
    busy.set(false)

    expect($responseTimes.get()).toEqual([])
  })
})

describe('formatDuration', () => {
  it('formats milliseconds, seconds and minutes', () => {
    expect(formatDuration(850)).toBe('850 ms')
    expect(formatDuration(4200)).toBe('4.2 sec')
    expect(formatDuration(42_000)).toBe('42 sec')
    expect(formatDuration(95_000)).toBe('1 min 35 sec')
  })

  it('uses the app language', () => {
    expect(formatDuration(4200, 'ja')).toBe('4.2 秒')
  })
})

describe('sparklinePoints', () => {
  it('scales values into the box, highest at the top', () => {
    expect(sparklinePoints([0, 10], 100, 20, 0)).toBe('0.0,20.0 100.0,0.0')
  })

  it('draws nothing without data and a centred point for one value', () => {
    expect(sparklinePoints([])).toBe('')
    expect(sparklinePoints([5], 120, 28, 2)).toBe('60.0,2.0')
  })
})
