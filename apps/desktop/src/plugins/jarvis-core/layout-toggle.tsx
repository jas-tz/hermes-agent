/**
 * Titlebar entry for the JARVIS layout — one click from anywhere to sessions |
 * face | chat, the same apply path as picking "JARVIS" in the Layout picker.
 * Shown pressed while that layout is the active preset. A layout switch only:
 * it never touches the mic, playback, the chat or a pending prompt (the face
 * pane itself stays control-free).
 */
import { applyLayoutPreset, Button, cn, host, type LayoutNode, Tip, useValue } from '@hermes/plugin-sdk'

import { useJarvisCoreText } from './i18n'

/** Mini core glyph — the face's rings at icon size, in currentColor. */
function CoreGlyph() {
  return (
    <svg aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 16 16">
      <circle cx="8" cy="8" r="6.5" strokeDasharray="3 2" strokeWidth="1" />
      <circle cx="8" cy="8" r="4" strokeWidth="1.25" />
      <circle cx="8" cy="8" fill="currentColor" r="1.75" stroke="none" />
    </svg>
  )
}

export function JarvisLayoutToggle({ presetId, tree }: { presetId: string; tree: LayoutNode }) {
  const active = useValue(host.state.layoutPreset) === presetId
  const text = useJarvisCoreText()

  // Size, padding and radius come from the `xs` Button size; only the label's
  // tracking and the pressed tint are set here. Hover teaches what the word
  // alone doesn't (that it switches the layout), so it gets a <Tip>.
  return (
    <Tip label={active ? text.toggle.active : text.toggle.switchTo}>
      <Button
        aria-label={text.toggle.name}
        aria-pressed={active}
        className={cn(
          'font-semibold tracking-[0.18em]',
          active ? 'text-(--theme-primary)' : 'text-(--ui-text-tertiary) hover:text-(--ui-text-primary)'
        )}
        data-testid="jarvis-layout-toggle"
        onClick={() => applyLayoutPreset(presetId, tree)}
        size="xs"
        type="button"
        variant="ghost"
      >
        <CoreGlyph />
        {text.toggle.label}
      </Button>
    </Tip>
  )
}
