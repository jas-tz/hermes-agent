import { useEffect, useState } from 'react'

/** "Africa/Dar_es_Salaam" → "Dar es Salaam"; null for UTC/offset-only zones. */
export function timeZoneCity(timeZone: string | undefined): null | string {
  const city = timeZone?.split('/').pop()

  if (!city || !timeZone?.includes('/')) {
    return null
  }

  return city.replace(/_/g, ' ')
}

/**
 * The top bar's centre line, e.g. "02:46 · FRI 16 OCT · DAR ES SALAAM" —
 * 24-hour time, weekday, day, month, and the system time zone's city, in the
 * app's language. Pure so it can be tested against fixed dates and zones.
 */
export function formatClock(date: Date, appLocale: string, timeZone?: string): string {
  // English renders day-before-month ("FRI 16 OCT") as in the design; plain
  // `en` would give the US order.
  const locale = appLocale === 'en' || appLocale.startsWith('en-US') ? 'en-GB' : appLocale
  const zone = timeZone ? { timeZone } : {}

  const time = new Intl.DateTimeFormat(locale, {
    hour: '2-digit',
    hourCycle: 'h23',
    minute: '2-digit',
    ...zone
  }).format(date)

  const day = new Intl.DateTimeFormat(locale, { day: 'numeric', month: 'short', weekday: 'short', ...zone })
    .format(date)
    .replace(/,/g, '')

  const city = timeZoneCity(timeZone ?? Intl.DateTimeFormat().resolvedOptions().timeZone)

  return [time, day, city].filter(Boolean).join(' · ').toLocaleUpperCase(locale)
}

/** Re-renders once a minute (aligned to the minute) with the current time. */
export function useMinuteClock(): Date {
  const [now, setNow] = useState(() => new Date())

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>

    const schedule = () => {
      const current = new Date()
      setNow(current)
      timer = setTimeout(schedule, 60_000 - (current.getSeconds() * 1000 + current.getMilliseconds()))
    }

    timer = setTimeout(schedule, 60_000 - (now.getSeconds() * 1000 + now.getMilliseconds()))

    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- schedule once; it re-arms itself
  }, [])

  return now
}
