import { useI18n } from '@/i18n'

import { useJarvisShell } from './use-jarvis-shell'
import { usePublishedVoiceProvider } from './voice-provider'

/** The agent's text-to-speech tool (tools/tts_tool.py). */
export const TTS_TOOL = 'text_to_speech'

/**
 * JARVIS shell transcript chips (design: "Spoke · Piper"): the text-to-speech
 * tool row reads "Spoke" with the configured voice provider in place of its
 * generic title and chunk count. Null outside the JARVIS shell or for any
 * other tool, so the classic rows are untouched.
 */
export function useSpokeChip(toolName: string | undefined): null | { meta: null | string; title: string } {
  const { t } = useI18n()
  const jarvisShell = useJarvisShell()
  const provider = usePublishedVoiceProvider()

  if (!jarvisShell || toolName !== TTS_TOOL) {
    return null
  }

  return { meta: provider, title: t.jarvisShell.spoke }
}
