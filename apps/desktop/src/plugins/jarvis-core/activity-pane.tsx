/**
 * Activity — the design's chat-zone tab beside Conversation and Files: what the
 * agent backend is doing (model requests, tool calls, gateway events), as the
 * live tail of the Hermes log. Read-only, through the SDK's `host.logs` — the
 * same source as the app's own Logs panel — refreshed every few seconds while
 * the tab is open.
 */
import { host, LogView, useQuery } from '@hermes/plugin-sdk'

import { useJarvisCoreText } from './i18n'

export const ACTIVITY_LINES = 200
const REFRESH_MS = 5000

export function ActivityPane() {
  const text = useJarvisCoreText()

  const { data, isError } = useQuery({
    queryFn: () => host.logs({ lines: ACTIVITY_LINES }),
    queryKey: ['jarvis-core', 'activity'],
    refetchInterval: REFRESH_MS
  })

  const lines = data?.lines ?? []

  return (
    <div className="h-full min-h-0 p-3" data-testid="jarvis-activity">
      {isError ? (
        <p className="text-sm text-(--ui-text-secondary)">{text.activity.error}</p>
      ) : lines.length === 0 ? (
        <p className="text-sm text-(--ui-text-secondary)">{text.activity.empty}</p>
      ) : (
        <LogView className="h-full">{lines.join('\n')}</LogView>
      )}
    </div>
  )
}
