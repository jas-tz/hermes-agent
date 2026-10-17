/**
 * The JARVIS Memory card at the bottom of the sessions sidebar
 * (docs/jarvis-design/references/01-main-screen.jpg): an eyebrow, the top
 * goals from continuity (or the goal / task / decision counts when there are
 * none), and "Open memory vault →" to this plugin's page. Read-only: it shares
 * the page's own status / continuity queries, so opening the page reuses the
 * cache and nothing is fetched twice.
 */
import './memory-card.css'

import { Button, host, type PluginLocaleBundles, usePluginI18n, useQuery } from '@hermes/plugin-sdk'

import { continuityKey, fetchContinuity, fetchStatus, statusKey } from './api'

const QUERY_OPTIONS = { refetchOnWindowFocus: false, retry: false, staleTime: 60_000 } as const
/** How many goal titles the card lists. */
const GOAL_LIMIT = 3

/** This plugin's own strings (Memory card + Active tasks bar), en/ja/zh/zh-hant. */
export const MEMORY_CARD_LOCALES: PluginLocaleBundles = {
  en: {
    tasks: {
      eyebrow: 'Active tasks',
      all: 'All tasks →',
      none: 'No active tasks',
      status: { in_progress: 'In progress', pending: 'Queued', blocked: 'Blocked' }
    },
    card: {
      eyebrow: 'Memory',
      open: 'Open memory vault →',
      counts: (goals: number, tasks: number, decisions: number) =>
        `${goals} goals · ${tasks} tasks · ${decisions} decisions`,
      unavailable: 'Memory unavailable'
    }
  },
  ja: {
    tasks: {
      eyebrow: '進行中のタスク',
      all: 'すべてのタスク →',
      none: '進行中のタスクはありません',
      status: { in_progress: '進行中', pending: '待機中', blocked: 'ブロック中' }
    },
    card: {
      eyebrow: 'メモリ',
      open: 'メモリ保管庫を開く →',
      counts: (goals: number, tasks: number, decisions: number) =>
        `目標 ${goals} · タスク ${tasks} · 決定 ${decisions}`,
      unavailable: 'メモリを利用できません'
    }
  },
  zh: {
    tasks: {
      eyebrow: '进行中的任务',
      all: '全部任务 →',
      none: '没有进行中的任务',
      status: { in_progress: '进行中', pending: '排队中', blocked: '受阻' }
    },
    card: {
      eyebrow: '记忆',
      open: '打开记忆库 →',
      counts: (goals: number, tasks: number, decisions: number) =>
        `${goals} 个目标 · ${tasks} 个任务 · ${decisions} 个决策`,
      unavailable: '记忆不可用'
    }
  },
  'zh-hant': {
    tasks: {
      eyebrow: '進行中的任務',
      all: '全部任務 →',
      none: '沒有進行中的任務',
      status: { in_progress: '進行中', pending: '排隊中', blocked: '受阻' }
    },
    card: {
      eyebrow: '記憶',
      open: '開啟記憶庫 →',
      counts: (goals: number, tasks: number, decisions: number) =>
        `${goals} 個目標 · ${tasks} 個任務 · ${decisions} 個決策`,
      unavailable: '記憶無法使用'
    }
  }
}

export function MemoryCard({ route }: { route: string }) {
  const t = usePluginI18n('jarvis')
  const status = useQuery({ queryFn: fetchStatus, queryKey: statusKey, ...QUERY_OPTIONS })
  const continuity = useQuery({ queryFn: fetchContinuity, queryKey: continuityKey, ...QUERY_OPTIONS })

  const goals =
    continuity.data?.enabled && continuity.data.goals
      ? continuity.data.goals.items
          .map(goal => goal.title.trim())
          .filter(Boolean)
          .slice(0, GOAL_LIMIT)
      : []

  const counts = status.data?.enabled && status.data.continuity.state === 'available' ? status.data.continuity : null

  let summary = t('card.unavailable')

  if (goals.length > 0) {
    summary = goals.join(' · ')
  } else if (counts) {
    summary = t('card.counts', counts.goals ?? 0, counts.tasks ?? 0, counts.decisions ?? 0)
  }

  return (
    <section className="jarvis-memory-card" data-testid="jarvis-memory-card">
      <p className="jarvis-memory-card__eyebrow">{t('card.eyebrow')}</p>
      <p className="jarvis-memory-card__summary" data-testid="jarvis-memory-card-summary">
        {summary}
      </p>
      <Button onClick={() => host.navigate(route)} size="inline" type="button" variant="link">
        {t('card.open')}
      </Button>
    </section>
  )
}
