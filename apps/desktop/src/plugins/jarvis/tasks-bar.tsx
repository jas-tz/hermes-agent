/**
 * Active tasks bar under the JARVIS face's status cards
 * (docs/jarvis-design/references/01-main-screen.jpg), contributed to the
 * JARVIS Core stage area. Lists the open continuity tasks — highest priority
 * first — with their real status. Continuity records no step counts, so the bar
 * under each task shows activity by status (a moving sweep while in progress, a
 * stub while queued) rather than an invented percentage. Read-only: it shares
 * the memory page's continuity query, and "All tasks →" opens that page.
 */
import './memory-card.css'

import { Button, host, usePluginI18n, useQuery } from '@hermes/plugin-sdk'

import { continuityKey, fetchContinuity, type Task } from './api'

const QUERY_OPTIONS = { refetchOnWindowFocus: false, retry: false, staleTime: 60_000 } as const
/** How many tasks the bar shows. */
export const TASK_LIMIT = 4
const CLOSED = new Set(['archived', 'cancelled', 'canceled', 'completed', 'done'])
const KNOWN_STATUS = new Set(['blocked', 'in_progress', 'pending'])

/** Open tasks, highest priority first (stable for equal priority). */
export function activeTasks(tasks: readonly Task[], limit = TASK_LIMIT): Task[] {
  return tasks
    .filter(task => !CLOSED.has(task.status.toLowerCase()))
    .map((task, index) => ({ index, task }))
    .sort((a, b) => (b.task.priority ?? 0) - (a.task.priority ?? 0) || a.index - b.index)
    .slice(0, limit)
    .map(({ task }) => task)
}

export function TasksBar({ route }: { route: string }) {
  const t = usePluginI18n('jarvis')
  const continuity = useQuery({ queryFn: fetchContinuity, queryKey: continuityKey, ...QUERY_OPTIONS })

  if (!continuity.data?.enabled || !continuity.data.tasks) {
    return null
  }

  const tasks = activeTasks(continuity.data.tasks.items)

  const statusLabel = (status: string) =>
    KNOWN_STATUS.has(status) ? t(`tasks.status.${status}`) : status.replace(/_/g, ' ')

  return (
    <section className="jarvis-tasks" data-testid="jarvis-tasks-bar">
      <div className="jarvis-tasks__head">
        <p className="jarvis-memory-card__eyebrow">{t('tasks.eyebrow')}</p>
        <Button onClick={() => host.navigate(route)} size="inline" type="button" variant="link">
          {t('tasks.all')}
        </Button>
      </div>
      {tasks.length === 0 ? (
        <p className="jarvis-tasks__empty">{t('tasks.none')}</p>
      ) : (
        <ul className="jarvis-tasks__grid">
          {tasks.map(task => (
            <li className="jarvis-tasks__task" data-status={task.status} data-testid="jarvis-task" key={task.id}>
              <div className="jarvis-tasks__row">
                <span className="jarvis-tasks__title">{task.title}</span>
                <span className="jarvis-tasks__status">{statusLabel(task.status)}</span>
              </div>
              <div aria-hidden="true" className="jarvis-tasks__track">
                <i />
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
