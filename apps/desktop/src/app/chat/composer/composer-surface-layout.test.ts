import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

// Source-text read: jsdom does no layout, so the overflow this guards against
// can't be measured in a render test. The composer surface is a one-column
// grid whose rows are the git status bar, then the input + controls. With an
// implicit (`auto`) column, the status bar's min-content width (branch + diff
// stat + hover actions) sized the column wider than the surface in a narrow
// zone such as the JARVIS layout's chat column; `overflow-hidden` then clipped
// the right-aligned mic / read-aloud / wake-word / voice-conversation buttons
// off-screen. A `minmax(0,1fr)` column keeps every row inside the surface.
const source = readFileSync(resolve(__dirname, 'index.tsx'), 'utf8')

describe('composer surface layout', () => {
  it('bounds its grid column to the surface so a long status row cannot push the controls out of view', () => {
    const surface = source.match(/'group\/composer-surface [^']*'/)?.[0]

    expect(surface).toBeDefined()
    expect(surface).toContain('grid-cols-[minmax(0,1fr)]')
  })
})
