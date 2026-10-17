import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { WorkspaceTabTitle } from './workspace-tab-title'

const shell = vi.hoisted(() => ({ jarvis: false }))

vi.mock('./use-jarvis-shell', () => ({ useJarvisShell: () => shell.jarvis }))

// Source-text read: vitest stubs CSS imports.
const css = readFileSync(resolve(__dirname, 'jarvis-chat.css'), 'utf8')

const rule = (selector: string) => {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

  return css.match(new RegExp(`${escaped} \\{([^}]*)\\}`))?.[1] ?? ''
}

afterEach(() => {
  cleanup()
  shell.jarvis = false
})

describe('chat tab title', () => {
  it('is "Conversation" in the JARVIS shell', () => {
    shell.jarvis = true
    render(<WorkspaceTabTitle classic={() => 'Friendly greeting #7'} />)

    expect(screen.getByText('Conversation')).toBeTruthy()
  })

  it('keeps the session title in the classic shell', () => {
    render(<WorkspaceTabTitle classic={() => 'Friendly greeting #7'} />)

    expect(screen.getByText('Friendly greeting #7')).toBeTruthy()
  })
})

describe('JARVIS chat styles', () => {
  const JARVIS = ":root[data-hermes-theme='jarvis']"

  it('only apply with the JARVIS theme', () => {
    const selectors = [...css.matchAll(/^([^\s@}/][^{]*)\{/gm)].map(m => m[1].trim())

    expect(selectors.length).toBeGreaterThan(0)

    for (const selector of selectors.filter(sel => !sel.match(/^\d+%$/))) {
      for (const part of selector.split(',')) {
        expect(part.trim().startsWith(JARVIS), part).toBe(true)
      }
    }
  })

  it('stop pinning your message over the reply, and right-align it', () => {
    const root = rule(`${JARVIS} [data-slot='aui_user-message-root']`)

    expect(root).toContain('position: static')
    expect(root).toContain('align-items: flex-end')
    expect(rule(`${JARVIS} [data-slot='aui_user-message-root'] .sticky-human-clamp`)).toContain('max-height: none')
  })

  it("put the glowing ring avatar beside JARVIS's replies", () => {
    expect(rule(`${JARVIS} [data-slot='aui_assistant-message-root']`)).toContain('padding-left')

    const ring = rule(`${JARVIS} [data-slot='aui_assistant-message-root']::before`)
    expect(ring).toContain('border-radius: 50%')
    expect(ring).toContain('box-shadow')
  })

  it('make the dictation control a big glowing cyan mic', () => {
    const mic = rule(`${JARVIS} .jarvis-mic`)

    expect(mic).toContain('width: 2.75rem')
    expect(mic).toContain('border-radius: 50%')
    expect(mic).toContain('background: var(--jarvis-cyan')
  })

  it('respect reduced motion', () => {
    const reduced = [...css.matchAll(/@media \(prefers-reduced-motion: reduce\) \{([\s\S]*?)\n\}/g)].map(m => m[1])

    expect(reduced.join('\n')).toContain("[data-streaming='true']::before")
    expect(reduced.join('\n')).toContain(".jarvis-mic[data-status='recording']")
  })
})
