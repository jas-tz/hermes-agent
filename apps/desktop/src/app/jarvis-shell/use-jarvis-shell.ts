import { useTheme } from '@/themes/context'

/** The skin whose look is the JARVIS desktop (themes/presets.ts). */
export const JARVIS_SKIN = 'jarvis'

/**
 * True while the JARVIS theme is applied: the JARVIS shell (top bar, icon rail)
 * replaces the classic Hermes chrome. Any other skin — Nous is the fallback —
 * keeps the classic shell untouched, so switching theme in Settings ▸
 * Appearance is the way back.
 */
export function useJarvisShell(): boolean {
  return useTheme().themeName === JARVIS_SKIN
}
