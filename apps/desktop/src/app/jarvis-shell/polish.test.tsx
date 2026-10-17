import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { TTS_TOOL, useSpokeChip } from './scaffold-chips'
import { $voiceProvider } from './voice-provider'

const shell = vi.hoisted(() => ({ jarvis: true }))

vi.mock('./use-jarvis-shell', () => ({ useJarvisShell: () => shell.jarvis }))

const chatCss = readFileSync(resolve(__dirname, 'jarvis-chat.css'), 'utf8')
const shellCss = readFileSync(resolve(__dirname, 'jarvis-shell.css'), 'utf8')
const appCss = readFileSync(resolve(__dirname, '../../styles.css'), 'utf8')

function Chip({ tool }: { tool: string }) {
  const chip = useSpokeChip(tool)

  return <p data-testid="chip">{chip ? `${chip.title} · ${chip.meta ?? ''}` : 'classic'}</p>
}

afterEach(() => {
  cleanup()
  shell.jarvis = true
  $voiceProvider.set(null)
})

describe('Spoke chip', () => {
  it('relabels the text-to-speech row with the configured voice', () => {
    $voiceProvider.set('Piper')
    render(<Chip tool={TTS_TOOL} />)

    expect(screen.getByTestId('chip').textContent).toBe('Spoke · Piper')
  })

  it('leaves other tools, and the classic shell, untouched', () => {
    render(<Chip tool="terminal" />)
    expect(screen.getByTestId('chip').textContent).toBe('classic')

    cleanup()
    shell.jarvis = false
    render(<Chip tool={TTS_TOOL} />)
    expect(screen.getByTestId('chip').textContent).toBe('classic')
  })
})

describe('transcript chips', () => {
  it('show collapsed scaffold rows as inline rounded chips that hug their label', () => {
    expect(chatCss).toMatch(/> \[data-conversation-scaffold\] \{\s*display: inline-flex;\s*width: fit-content;/)
    expect(chatCss).toMatch(/border-radius: var\(--jarvis-radius-pill, 999px\)/)
  })

  it('give an expanded row the full width back, and keep the click-to-expand button', () => {
    expect(chatCss).toContain(":has([data-slot='disclosure-row-button'][aria-expanded='true'])")
    expect(chatCss).toMatch(/\[data-tool-open\] \{\s*display: flex;\s*width: 100%;/)
  })
})

describe('sidebar polish', () => {
  it('flattens the JARVIS sidebar into one scroll, so sections never cut a row', () => {
    expect(appCss).toMatch(
      /@custom-variant compact \{\s*@media \(max-height: 768px\) \{\s*@slot;\s*\}\s*:root\[data-hermes-theme='jarvis'\] & \{\s*@slot;/
    )
  })

  it('fades the long session list at its edge instead of cutting a row in half', () => {
    expect(shellCss).toContain(":root[data-hermes-theme='jarvis'] .sidebar-sessions-scroller {")
    expect(shellCss).toContain('animation-timeline: scroll(self)')
  })
})
