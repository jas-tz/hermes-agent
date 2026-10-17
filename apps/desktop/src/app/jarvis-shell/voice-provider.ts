import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { atom } from 'nanostores'
import { useEffect } from 'react'

import { getHermesConfig } from '@/hermes'

/** "piper" → "Piper"; null when no TTS provider is configured. */
export function voiceProviderLabel(config: unknown): null | string {
  const provider = (config as { tts?: { provider?: unknown } } | null)?.tts?.provider

  if (typeof provider !== 'string' || !provider.trim()) {
    return null
  }

  const name = provider.trim()

  return name.charAt(0).toUpperCase() + name.slice(1)
}

/** The configured TTS provider's display name, published by the top bar for
 *  surfaces that shouldn't own a config query (transcript chips). */
export const $voiceProvider = atom<null | string>(null)

/** Read the configured TTS provider (the top bar's Voice pill) and publish it
 *  to $voiceProvider. */
export function useVoiceProvider(): null | string {
  const config = useQuery({
    queryFn: () => getHermesConfig(),
    queryKey: ['jarvis-shell', 'config'],
    staleTime: 60_000
  })

  const label = voiceProviderLabel(config.data)

  useEffect(() => $voiceProvider.set(label), [label])

  return label
}

/** The last published TTS provider, without a query of its own. */
export function usePublishedVoiceProvider(): null | string {
  return useStore($voiceProvider)
}
