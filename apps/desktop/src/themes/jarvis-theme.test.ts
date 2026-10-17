import { existsSync, readFileSync, statSync } from 'node:fs'
import { resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

import { getBaseColors } from './context'
import { BUILTIN_THEMES, DEFAULT_SKIN_NAME, EMOJI_FALLBACK, jarvisTheme } from './presets'

// Source-text read: vitest stubs CSS imports.
const css = readFileSync(resolve(__dirname, '../styles.css'), 'utf8')

const jarvisBlock = css.match(/:root\[data-hermes-theme='jarvis'\] \{([\s\S]*?)\n\}/)?.[1] ?? ''

function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map(i => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255

    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  })

  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x)

  return (hi + 0.05) / (lo + 0.05)
}

describe('JARVIS theme', () => {
  it('is the default skin, with Nous kept selectable as the fallback', () => {
    expect(BUILTIN_THEMES.jarvis).toBe(jarvisTheme)
    expect(jarvisTheme).toMatchObject({ label: 'JARVIS', name: 'jarvis' })
    expect(DEFAULT_SKIN_NAME).toBe('jarvis')
    expect(BUILTIN_THEMES.nous).toMatchObject({ label: 'Nous', name: 'nous' })
  })

  it('carries the design palette', () => {
    expect(jarvisTheme.colors).toMatchObject({
      background: '#06090D',
      border: '#16222D',
      card: '#0A1117',
      foreground: '#DCE8F2',
      input: '#1E2A35',
      mutedForeground: '#8DA1B3',
      primary: '#4FD1FF',
      sidebarBackground: '#080D12'
    })
  })

  it('stays dark in light mode as well as dark mode', () => {
    expect(getBaseColors('jarvis', 'dark').background).toBe('#06090D')
    expect(getBaseColors('jarvis', 'light').background).toBe('#06090D')
    expect(getBaseColors('jarvis', 'light')).toEqual(getBaseColors('jarvis', 'dark'))
  })

  it('keeps text readable (WCAG AA, AAA for body text)', () => {
    const c = jarvisTheme.colors

    expect(contrast(c.foreground, c.background)).toBeGreaterThanOrEqual(7)
    expect(contrast(c.cardForeground, c.card)).toBeGreaterThanOrEqual(7)
    expect(contrast(c.mutedForeground, c.background)).toBeGreaterThanOrEqual(4.5)
    expect(contrast(c.mutedForeground, c.card)).toBeGreaterThanOrEqual(4.5)
    expect(contrast(c.primaryForeground, c.primary)).toBeGreaterThanOrEqual(4.5)
    expect(contrast(c.destructiveForeground, c.destructive)).toBeGreaterThanOrEqual(4.5)
  })

  it('uses the design fonts with system and emoji fallbacks', () => {
    const typo = jarvisTheme.typography

    expect(typo?.fontSans).toMatch(/^"IBM Plex Sans", /)
    expect(typo?.fontMono).toMatch(/^"JetBrains Mono", /)
    expect(typo?.fontSans).toContain(EMOJI_FALLBACK)
  })

  it('needs no network for its fonts', () => {
    // An own `fontUrl: undefined` — not merely absent — so applyTheme's merge
    // over Nous's typography cannot inherit Nous's Google Fonts stylesheet.
    expect(jarvisTheme.typography).toHaveProperty('fontUrl', undefined)
  })
})

describe('JARVIS design tokens (styles.css)', () => {
  it('are scoped to the JARVIS skin only', () => {
    expect(jarvisBlock).not.toBe('')
    expect(css.match(/--jarvis-space-1:/g)).toHaveLength(1)
  })

  it('sit outside @layer base so they override its defaults', () => {
    const baseLayer = css.indexOf('@layer base {')
    // The token block itself (the sidebar `compact` variant also names the skin).
    const block = css.indexOf(":root[data-hermes-theme='jarvis'] {")
    const afterBase = css.indexOf('\n}\n', baseLayer)

    expect(block).toBeGreaterThan(afterBase)
  })

  it('set the design radius through the app scale (8 / 12 / 16 px)', () => {
    expect(jarvisBlock).toMatch(/--radius-scalar: 1;/)
    expect(jarvisBlock).toContain('--jarvis-radius-control: var(--radius-sm)')
    expect(jarvisBlock).toContain('--jarvis-radius-card: var(--radius-lg)')
  })

  it('define colour, type, spacing and layout tokens', () => {
    for (const token of [
      '--jarvis-bg',
      '--jarvis-panel',
      '--jarvis-line',
      '--jarvis-text',
      '--jarvis-cyan',
      '--jarvis-amber',
      '--jarvis-green',
      '--jarvis-red',
      '--jarvis-glow',
      '--jarvis-font-display',
      '--jarvis-font-body',
      '--jarvis-font-mono',
      '--jarvis-space-1',
      '--jarvis-space-6',
      '--jarvis-topbar-height',
      '--jarvis-rail-width',
      '--jarvis-sidebar-width',
      '--jarvis-chat-width'
    ]) {
      expect(jarvisBlock, token).toContain(`${token}:`)
    }

    expect(jarvisBlock).toContain("--jarvis-font-display: 'Chakra Petch', var(--dt-font-sans)")
  })

  it('reference the theme colours rather than repeating them', () => {
    expect(jarvisBlock).toContain('--jarvis-bg: var(--theme-background-seed)')
    expect(jarvisBlock).toContain('--jarvis-cyan: var(--theme-primary)')
    expect(jarvisBlock).not.toMatch(/#06090d|#4fd1ff|#dce8f2/i)
  })
})

describe('JARVIS fonts (bundled)', () => {
  const faces = [...css.matchAll(/@font-face \{([\s\S]*?)\n\}/g)].map(m => m[1])
  const facesFor = (family: string) => faces.filter(f => f.includes(`font-family: '${family}'`))
  const fontsDir = resolve(__dirname, '../fonts')

  it('declares Chakra Petch 500/600, IBM Plex Sans 400–600 and JetBrains Mono 500', () => {
    const weights = (family: string) => new Set(facesFor(family).map(f => f.match(/font-weight: ([\d ]+);/)?.[1]))

    expect(weights('Chakra Petch')).toEqual(new Set(['500', '600']))
    expect(weights('IBM Plex Sans')).toEqual(new Set(['400 600']))
    expect(weights('JetBrains Mono')).toEqual(new Set(['400', '500', '700']))
  })

  it('loads every face from a bundled file that exists, never the network', () => {
    for (const family of ['Chakra Petch', 'IBM Plex Sans', 'JetBrains Mono']) {
      for (const face of facesFor(family)) {
        const file = face.match(/url\('\.\/fonts\/([^']+)'\)/)?.[1]

        expect(file, face).toBeTruthy()
        expect(existsSync(resolve(fontsDir, file!)), file).toBe(true)
        expect(statSync(resolve(fontsDir, file!)).size, file).toBeGreaterThan(1000)
      }
    }

    expect(css).not.toMatch(/fonts\.(googleapis|gstatic)\.com/)
  })

  it('splits the new faces into Latin and Latin Extended subsets', () => {
    for (const family of ['Chakra Petch', 'IBM Plex Sans']) {
      for (const face of facesFor(family)) {
        expect(face).toMatch(/unicode-range:/)
      }
    }
  })

  it('ships the SIL Open Font License for each bundled family', () => {
    for (const name of ['ChakraPetch', 'IBMPlexSans', 'JetBrainsMono']) {
      expect(readFileSync(resolve(fontsDir, `OFL-${name}.txt`), 'utf8')).toContain('SIL OPEN FONT LICENSE')
    }
  })
})
