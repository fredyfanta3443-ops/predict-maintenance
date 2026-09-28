import { useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { DashboardData } from '../types'
import { AXIS_PROPS, COLORS, Legend, Section, Tile, TooltipBox, fmt } from '../ui'

const BIN = 20

export function Problem({ data }: { data: DashboardData }) {
  const [interval, setInterval] = useState(150)
  const life = data.life

  const stats = useMemo(() => {
    const failed = life.filter((T) => T <= interval)
    const survived = life.filter((T) => T > interval)
    const flown = failed.reduce((a, b) => a + b, 0) + survived.length * interval
    const cost = failed.length * data.failure_cost + survived.length
    const wasted = survived.length ? survived.reduce((a, T) => a + (T - interval), 0) / survived.length : 0
    return { failed: failed.length, wasted, costPer1000: (1000 * cost) / flown }
  }, [life, interval, data.failure_cost])

  const bins = useMemo(() => {
    const lo = Math.floor(Math.min(...life) / BIN) * BIN
    const hi = Math.ceil((Math.max(...life) + 1) / BIN) * BIN
    const out = []
    for (let b = lo; b < hi; b += BIN) {
      const inBin = life.filter((T) => T >= b && T < b + BIN)
      out.push({
        label: `${b}–${b + BIN - 1}`,
        mid: b + BIN / 2,
        failed: inBin.filter((T) => T <= interval).length,
        serviced: inBin.filter((T) => T > interval).length,
      })
    }
    return out
  }, [life, interval])

  const shortest = Math.min(...life)
  const longest = Math.max(...life)
  const avg = life.reduce((a, b) => a + b, 0) / life.length

  return (
    <Section
      id="problem"
      num="01 · The problem"
      title="Engines don't wear out on schedule"
      lede={
        <>
          The client services every engine every N flights. But across 100 engines run until they failed, lifetimes ranged
          from <strong>{shortest}</strong> to <strong>{longest}</strong> flights (average {fmt(avg, 0)}). No single N
          works: set it low and you throw away healthy life, set it high and engines fail in service.
        </>
      }
    >
      <div className="card">
        <h3>How long 100 engines lasted, and what a fixed schedule does to them</h3>
        <p className="card-sub">Drag the slider to change the service interval.</p>
        <div className="controls">
          <label htmlFor="interval">Service every</label>
          <input id="interval" type="range" min={80} max={260} value={interval} onChange={(e) => setInterval(+e.target.value)} />
          <output htmlFor="interval">{interval}</output> flights
        </div>
        <div className="tiles">
          <Tile label="Engines that fail before service" value={`${stats.failed} / 100`} note="each costs ~10× a planned service" />
          <Tile label="Life thrown away per serviced engine" value={`${fmt(stats.wasted, 0)} flights`} note="engine was still healthy" />
          <Tile label="Cost per 1,000 flights" value={fmt(stats.costPer1000, 2)} note="planned service = 1, failure = 10" />
        </div>
        <div className="chart">
          <ResponsiveContainer>
            <BarChart data={bins} margin={{ top: 16, right: 8, bottom: 8, left: 0 }} barCategoryGap={2}>
              <CartesianGrid vertical={false} stroke={COLORS.grid} />
              <XAxis dataKey="label" {...AXIS_PROPS} interval="preserveStartEnd" minTickGap={16} tickFormatter={(v: string) => v.split("–")[0]} label={{ value: 'flights until failure', position: 'insideBottom', offset: -4, fill: COLORS.muted, fontSize: 12 }} height={40} />
              <YAxis {...AXIS_PROPS} allowDecimals={false} width={32} />
              <Tooltip
                cursor={{ fill: 'var(--wash)' }}
                content={(p: any) =>
                  p.active && p.payload?.length ? (
                    <TooltipBox
                      title={`Lifetime ${p.label} flights`}
                      rows={[
                        { color: COLORS.critical, value: p.payload[0].payload.failed, label: 'fail before service' },
                        { color: COLORS.neutral, value: p.payload[0].payload.serviced, label: 'serviced with life left' },
                      ]}
                    />
                  ) : null
                }
              />
              <Bar dataKey="failed" stackId="a" fill={COLORS.critical} stroke="var(--surface-1)" strokeWidth={1} maxBarSize={24} isAnimationActive={false} />
              <Bar dataKey="serviced" stackId="a" fill={COLORS.neutral} stroke="var(--surface-1)" strokeWidth={1} radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false} />
              <ReferenceLine
                x={bins.find((b) => interval >= b.mid - BIN / 2 && interval < b.mid + BIN / 2)?.label}
                stroke={COLORS.fixed}
                strokeWidth={2}
                label={{ value: `service at ${interval}`, position: 'top', fill: 'var(--text-secondary)', fontSize: 12 }}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <Legend
          items={[
            { label: 'Fails before its service date (unplanned)', color: COLORS.critical },
            { label: 'Serviced while still healthy (wasted life)', color: COLORS.neutral },
            { label: 'Service interval', color: COLORS.fixed, kind: 'line' },
          ]}
        />
        <p className="callout">
          Try <strong>{shortest - 1}</strong>: no failures, but the average engine loses{' '}
          {fmt(life.reduce((a, T) => a + T - (shortest - 1), 0) / life.length, 0)} flights of life. Try <strong>200</strong>:{' '}
          {life.filter((T) => T <= 200).length} of 100 engines fail in service.{' '}
          <strong>The goal: service each engine just before it wears out.</strong>
        </p>
      </div>
    </Section>
  )
}
