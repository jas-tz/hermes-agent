import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

// Static-analysis guard: no <button> or <Button> element in the desktop renderer
// may use the native HTML `title=` attribute. Native tooltips are unstyled,
// delayed (~500ms OS default), and visually inconsistent with the themed `Tip`.
// When a tip is warranted (see DESIGN.md — not every icon, never menu triggers),
// use `<Tip label={...}>` instead of `title=`.
//
// This is a source-text scan, not a behavior test — it's the same category as
// an ESLint rule, expressed as a vitest so it runs with the rest of the suite.

// Recursively walk a directory and collect all .tsx file paths.
function collectTsxFiles(dir: string): string[] {
  const results: string[] = []

  for (const entry of readdirSync(dir)) {
    // Skip node_modules, dist, and __tests__ (this file itself)
    if (entry === 'node_modules' || entry === 'dist' || entry === '__tests__') {
      continue
    }

    const fullPath = join(dir, entry)
    const stat = statSync(fullPath)

    if (stat.isDirectory()) {
      results.push(...collectTsxFiles(fullPath))
    } else if (entry.endsWith('.tsx')) {
      results.push(fullPath)
    }
  }

  return results
}

describe('no native title= on button elements', () => {
  // Scan every .tsx file under src/ for <button or <Button opening tags that
  // also carry a title= attribute (anywhere in the opening tag, which may span
  // multiple lines).
  it('uses <Tip> instead of native title= on all button elements', () => {
    const violations: string[] = []
    const srcDir = resolve(__dirname, '../..')

    for (const filePath of collectTsxFiles(srcDir)) {
      const content = readFileSync(filePath, 'utf-8')
      const relativePath = filePath.replace(srcDir + '/', '')

      // Match <Button ...> or <button ...> opening tags (may span multiple lines).
      // We use a non-greedy match up to the closing > — this won't perfectly
      // handle every edge case (e.g. > inside a string literal), but it's good
      // enough for a lint-style guard.
      const tagPattern = /<(Button|button)\b([^>]*?)>/gsu
      let match: RegExpExecArray | null

      while ((match = tagPattern.exec(content)) !== null) {
        const tagName = match[1]
        const attrs = match[2]

        if (/\btitle=/.test(attrs)) {
          const lineNum = content.slice(0, match.index).split('\n').length
          violations.push(`${relativePath}:${lineNum} <${tagName}> has title= — use <Tip>`)
        }
      }
    }

    expect(violations, violations.join('\n')).toEqual([])
  })
})

// The scan above ends an opening tag at the first `>`, so an arrow function in
// an earlier prop (`onClick={() => …}`) cuts the tag short and a later `title=`
// goes unseen. This brace-aware pattern skips `{…}` prop values. It is applied
// to the JARVIS Core plugin only: across all of src/ it currently flags
// pre-existing buttons outside that plugin's scope, which are left as-is here.
const BRACED_VALUE = String.raw`\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}`
const BRACE_AWARE_TAG = new RegExp(String.raw`<(Button|button)\b((?:[^>{]|${BRACED_VALUE})*?)>`, 'gsu')

function nativeTitleTags(content: string): number[] {
  const lines: number[] = []

  for (const match of content.matchAll(BRACE_AWARE_TAG)) {
    if (/\btitle=/.test(match[2])) {
      lines.push(content.slice(0, match.index).split('\n').length)
    }
  }

  return lines
}

describe('no native title= on JARVIS Core buttons (brace-aware)', () => {
  it('sees a title= that follows an arrow-function prop', () => {
    const shape = `<Button\n  onClick={() => applyLayoutPreset(presetId, tree)}\n  title={active ? 'a' : 'b'}\n>`

    expect(nativeTitleTags(shape)).toEqual([1])
  })

  it('finds none in the JARVIS Core plugin', () => {
    const pluginDir = resolve(__dirname, '../../../plugins/jarvis-core')

    const violations = collectTsxFiles(pluginDir).flatMap(file =>
      nativeTitleTags(readFileSync(file, 'utf-8')).map(line => `${file}:${line}`)
    )

    expect(violations, violations.join('\n')).toEqual([])
  })
})
