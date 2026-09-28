import type { ReactNode } from 'react'
import type { Status } from './types'

export const COLORS = {
  model: 'var(--series-1)',
  fixed: 'var(--series-2)',
  sensor: 'var(--series-3)',
  neutral: 'var(--neutral-mark)',
  good: 'var(--good)',
  warning: 'var(--warning)',
  critical: 'var(--critical)',
  grid: 'var(--grid)',
  axis: 'var(--axis)',
  muted: 'var(--text-muted)',
  band: 'var(--band)',
}

export const AXIS_PROPS = {
  stroke: COLORS.axis,
  tick: { fill: COLORS.muted, fontSize: 12 },
  tickLine: false,
}

export const fmt = (n: number, digits = 1) =>
  n.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })

export function Section(props: { id: string; num: string; title: string; lede: ReactNode; children: ReactNode }) {
  return (
    <section className="section" id={props.id}>
      <div className="section-num">{props.num}</div>
      <h2>{props.title}</h2>
      <p className="section-lede">{props.lede}</p>
      {props.children}
    </section>
  )
}

export function Tile(props: { label: string; value: ReactNode; note?: ReactNode }) {
  return (
    <div className="tile">
      <div className="tile-label">{props.label}</div>
      <div className="tile-value">{props.value}</div>
      {props.note && <div className="tile-note">{props.note}</div>}
    </div>
  )
}

export function Legend(props: { items: { label: string; color: string; kind?: 'rect' | 'line' | 'dash' }[] }) {
  return (
    <div className="legend">
      {props.items.map((it) => (
        <span className="legend-item" key={it.label}>
          {it.kind === 'rect' || !it.kind ? (
            <span className="key-rect" style={{ background: it.color }} />
          ) : (
            <span className="key-line" style={{ borderTopColor: it.color, borderTopStyle: it.kind === 'dash' ? 'dashed' : 'solid' }} />
          )}
          {it.label}
        </span>
      ))}
    </div>
  )
}

const STATUS_META: Record<Status, { color: string; icon: string; label: string }> = {
  OK: { color: COLORS.good, icon: '✓', label: 'OK' },
  WATCH: { color: COLORS.warning, icon: '!', label: 'Watch' },
  ALERT: { color: COLORS.critical, icon: '✕', label: 'Alert' },
}

export function StatusBadge({ status }: { status: Status }) {
  const m = STATUS_META[status]
  return (
    <span className="status">
      <span className="status-dot" style={{ background: m.color }} aria-hidden>
        {m.icon}
      </span>
      {m.label}
    </span>
  )
}

export const statusColor = (s: Status) => STATUS_META[s].color

/** Tooltip body: value first (strong), series name second, keyed by a short line of the series color. */
export function TooltipBox(props: { title: ReactNode; rows: { color: string; value: ReactNode; label: string }[] }) {
  return (
    <div className="tooltip">
      <div className="tooltip-title">{props.title}</div>
      {props.rows.map((r) => (
        <div className="tooltip-row" key={r.label}>
          <span className="key-line" style={{ borderTopColor: r.color }} />
          <strong>{r.value}</strong>
          <span>{r.label}</span>
        </div>
      ))}
    </div>
  )
}
