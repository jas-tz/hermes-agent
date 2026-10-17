import type { ReactNode } from 'react'

import { useI18n } from '@/i18n'

import { useJarvisShell } from './use-jarvis-shell'

/**
 * The main chat tab's label. The JARVIS shell names it "Conversation" (the
 * design's Conversation / Activity / Files tabs — the session's own title is in
 * the top bar); the classic shell keeps the session title, via `classic`.
 */
export function WorkspaceTabTitle({ classic }: { classic: () => ReactNode }) {
  const { t } = useI18n()

  return <>{useJarvisShell() ? t.jarvisShell.conversation : classic()}</>
}
