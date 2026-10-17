/**
 * Plugin-scoped i18n for JARVIS Core — bundles registered under the plugin id
 * via `ctx.i18n.register`, never touching core `en.ts` (same pattern as
 * kanban). `useJarvisCoreText()` binds the stringly `t(key)` to the message
 * SHAPE so components read typed `text.states.idle`. Terms follow the app's own
 * locale files (e.g. "speaking"/"listening"/"layout") so the face and the rest
 * of the desktop agree.
 */

import { type PluginLocaleBundles, type PluginTranslate, usePluginI18n } from '@hermes/plugin-sdk'
import { useMemo } from 'react'

import type { CoreState } from './core-state'

type Messages = {
  states: Record<CoreState, string>
  /** One line under the mood label: what is happening / what to do next. */
  captions: Record<CoreState, string>
  rows: { agent: string; link: string; microphone: string; responseTime: string; voice: string }
  agent: { idle: string; waiting: string; working: string }
  voice: { idle: string; preparing: string; speaking: string }
  voiceInput: { dictating: string; idle: string; listening: string; transcribing: string }
  link: { connecting: string; offline: string; online: string }
  /** Status-card sub-lines. */
  cardSub: {
    agent: { idle: string; waiting: string; working: string }
    microphone: { dictating: string; idle: string; listening: string; transcribing: string }
    voice: { idle: string; preparing: string; speaking: string }
    link: { connecting: string; offline: string; online: string }
    responseTime: { none: string; last: (count: number) => string }
  }
  toggle: { label: string; name: string; switchTo: string; active: string }
  activity: { title: string; empty: string; error: string }
}

const en: Messages = {
  states: {
    dictating: 'Dictating',
    idle: 'Standing by',
    listening: 'Listening',
    'needs-input': 'Needs you',
    offline: 'Offline',
    preparing: 'Preparing voice',
    speaking: 'Speaking',
    thinking: 'Thinking',
    transcribing: 'Transcribing'
  },
  captions: {
    dictating: 'Recording your dictation.',
    idle: 'Start a voice conversation or type in the chat.',
    listening: "I'm listening — go ahead.",
    'needs-input': 'Respond in the chat to continue.',
    offline: "Can't reach the Hermes backend.",
    preparing: 'Preparing the spoken reply…',
    speaking: 'Reading the reply aloud.',
    thinking: 'Working on it…',
    transcribing: 'Turning your speech into text…'
  },
  rows: { agent: 'Agent', link: 'Link', microphone: 'Microphone', responseTime: 'Response time', voice: 'Voice' },
  agent: { idle: 'Idle', waiting: 'Waiting for you', working: 'Working' },
  voice: { idle: 'Idle', preparing: 'Synthesizing', speaking: 'Playing' },
  voiceInput: { dictating: 'Recording', idle: 'Off', listening: 'Listening', transcribing: 'Transcribing' },
  link: { connecting: 'Connecting', offline: 'Offline', online: 'Online' },
  cardSub: {
    agent: { idle: 'Ready', waiting: 'Answer in the chat', working: 'Running a turn' },
    microphone: {
      dictating: 'Dictation',
      idle: 'Mic closed',
      listening: 'Voice conversation',
      transcribing: 'Speech to text'
    },
    voice: { idle: 'Spoken replies', preparing: 'Generating speech', speaking: 'Reply audio' },
    link: { connecting: 'Reaching Hermes…', offline: 'Backend unreachable', online: 'Hermes backend' },
    responseTime: { none: 'No replies yet', last: count => `last ${count} replies` }
  },
  toggle: {
    label: 'JARVIS',
    name: 'JARVIS layout',
    switchTo: 'Switch to the JARVIS layout',
    active: 'JARVIS layout (active)'
  },
  activity: { title: 'Activity', empty: 'No activity yet', error: 'Activity unavailable' }
}

const ja: Messages = {
  states: {
    dictating: '音声入力中',
    idle: '待機中',
    listening: '聴いています',
    'needs-input': '対応が必要です',
    offline: 'オフライン',
    preparing: '音声を準備中',
    speaking: '話しています',
    thinking: '考え中',
    transcribing: '文字起こし中'
  },
  captions: {
    dictating: '音声入力を録音しています。',
    idle: '音声会話を始めるか、チャットに入力してください。',
    listening: '聴いています。どうぞ話してください。',
    'needs-input': '続けるにはチャットで応答してください。',
    offline: 'Hermes バックエンドに接続できません。',
    preparing: '音声の返答を準備しています…',
    speaking: '返答を読み上げています。',
    thinking: '作業中です…',
    transcribing: '音声をテキストに変換しています…'
  },
  rows: { agent: 'エージェント', link: '接続', microphone: 'マイク', responseTime: '応答時間', voice: '音声' },
  agent: { idle: '待機', waiting: 'あなたの応答待ち', working: '作業中' },
  voice: { idle: '待機', preparing: '合成中', speaking: '再生中' },
  voiceInput: { dictating: '録音中', idle: 'オフ', listening: '聴いています', transcribing: '文字起こし中' },
  link: { connecting: '接続中', offline: 'オフライン', online: 'オンライン' },
  cardSub: {
    agent: { idle: '準備完了', waiting: 'チャットで回答してください', working: 'ターンを実行中' },
    microphone: {
      dictating: '音声入力',
      idle: 'マイクはオフ',
      listening: '音声会話',
      transcribing: '音声をテキストに変換'
    },
    voice: { idle: '音声での返答', preparing: '音声を生成中', speaking: '返答の音声' },
    link: { connecting: 'Hermes に接続中…', offline: 'バックエンドに接続できません', online: 'Hermes バックエンド' },
    responseTime: { none: 'まだ返答はありません', last: count => `直近 ${count} 件の返答` }
  },
  toggle: {
    label: 'JARVIS',
    name: 'JARVIS レイアウト',
    switchTo: 'JARVIS レイアウトに切り替え',
    active: 'JARVIS レイアウト（使用中）'
  },
  activity: {
    title: 'アクティビティ',
    empty: 'まだアクティビティはありません',
    error: 'アクティビティを取得できません'
  }
}

const zh: Messages = {
  states: {
    dictating: '听写中',
    idle: '待命',
    listening: '聆听中',
    'needs-input': '需要你处理',
    offline: '离线',
    preparing: '正在准备语音',
    speaking: '讲话中',
    thinking: '思考中',
    transcribing: '转写中'
  },
  captions: {
    dictating: '正在录制你的听写。',
    idle: '开始语音对话，或在聊天中输入。',
    listening: '我在听，请说。',
    'needs-input': '请在聊天中回复以继续。',
    offline: '无法连接到 Hermes 后端。',
    preparing: '正在准备语音回复…',
    speaking: '正在朗读回复。',
    thinking: '正在处理…',
    transcribing: '正在将语音转为文字…'
  },
  rows: { agent: '代理', link: '连接', microphone: '麦克风', responseTime: '响应时间', voice: '语音' },
  agent: { idle: '空闲', waiting: '等待你的回复', working: '工作中' },
  voice: { idle: '空闲', preparing: '合成中', speaking: '播放中' },
  voiceInput: { dictating: '录音中', idle: '关闭', listening: '聆听中', transcribing: '转写中' },
  link: { connecting: '连接中', offline: '离线', online: '在线' },
  cardSub: {
    agent: { idle: '就绪', waiting: '请在聊天中回答', working: '正在执行' },
    microphone: { dictating: '听写', idle: '麦克风已关闭', listening: '语音对话', transcribing: '语音转文字' },
    voice: { idle: '语音回复', preparing: '正在生成语音', speaking: '回复音频' },
    link: { connecting: '正在连接 Hermes…', offline: '无法连接后端', online: 'Hermes 后端' },
    responseTime: { none: '暂无回复', last: count => `最近 ${count} 次回复` }
  },
  toggle: {
    label: 'JARVIS',
    name: 'JARVIS 布局',
    switchTo: '切换到 JARVIS 布局',
    active: 'JARVIS 布局（当前）'
  },
  activity: { title: '活动', empty: '暂无活动', error: '活动不可用' }
}

const zhHant: Messages = {
  states: {
    dictating: '聽寫中',
    idle: '待命',
    listening: '聆聽中',
    'needs-input': '需要你處理',
    offline: '離線',
    preparing: '正在準備語音',
    speaking: '說話中',
    thinking: '思考中',
    transcribing: '轉寫中'
  },
  captions: {
    dictating: '正在錄製你的聽寫。',
    idle: '開始語音對話，或在聊天中輸入。',
    listening: '我在聽，請說。',
    'needs-input': '請在聊天中回覆以繼續。',
    offline: '無法連線到 Hermes 後端。',
    preparing: '正在準備語音回覆…',
    speaking: '正在朗讀回覆。',
    thinking: '正在處理…',
    transcribing: '正在將語音轉為文字…'
  },
  rows: { agent: '代理', link: '連線', microphone: '麥克風', responseTime: '回應時間', voice: '語音' },
  agent: { idle: '閒置', waiting: '等待你的回覆', working: '工作中' },
  voice: { idle: '閒置', preparing: '合成中', speaking: '播放中' },
  voiceInput: { dictating: '錄音中', idle: '關閉', listening: '聆聽中', transcribing: '轉寫中' },
  link: { connecting: '連線中', offline: '離線', online: '線上' },
  cardSub: {
    agent: { idle: '就緒', waiting: '請在聊天中回答', working: '正在執行' },
    microphone: { dictating: '聽寫', idle: '麥克風已關閉', listening: '語音對話', transcribing: '語音轉文字' },
    voice: { idle: '語音回覆', preparing: '正在產生語音', speaking: '回覆音訊' },
    link: { connecting: '正在連線 Hermes…', offline: '無法連線後端', online: 'Hermes 後端' },
    responseTime: { none: '尚無回覆', last: count => `最近 ${count} 次回覆` }
  },
  toggle: {
    label: 'JARVIS',
    name: 'JARVIS 版面配置',
    switchTo: '切換到 JARVIS 版面配置',
    active: 'JARVIS 版面配置（目前）'
  },
  activity: { title: '活動', empty: '尚無活動', error: '活動無法使用' }
}

export const JARVIS_CORE_LOCALES: PluginLocaleBundles = { en, ja, zh, 'zh-hant': zhHant }

/** The locales this plugin ships — the four the desktop design system requires. */
export const JARVIS_CORE_LOCALE_IDS = ['en', 'ja', 'zh', 'zh-hant'] as const

function bind<T extends object>(t: PluginTranslate, template: T, prefix = ''): T {
  const out = {} as Record<string, unknown>

  for (const [key, value] of Object.entries(template)) {
    const path = prefix ? `${prefix}.${key}` : key
    out[key] =
      typeof value === 'function'
        ? (...args: unknown[]) => t(path, ...args)
        : value && typeof value === 'object'
          ? bind(t, value as object, path)
          : t(path)
  }

  return out as T
}

/** Typed, reactive JARVIS Core copy for the active app locale (falls back to English). */
export function useJarvisCoreText(): Messages {
  const t = usePluginI18n('jarvis-core')

  return useMemo(() => bind(t, en), [t])
}
