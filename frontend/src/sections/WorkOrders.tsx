import { useEffect, useState } from 'react'
import { CartesianGrid, ReferenceArea, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import type { DashboardData } from '../types'
import { AXIS_PROPS, COLORS, Legend, Section, Tile, TooltipBox, fmt } from '../ui'

interface LiveOrder {
  id: number
  unit: number
  predicted_rul: number
  last_cycle: number
  status: string
  created_at: string
}

type Live = { state: 'loading' } | { state: 'offline' } | { state: 'ok'; orders: LiveOrder[] }

export function WorkOrders({ data }: { data: DashboardData }) {
  const [live, setLive] = useState<Live>({ state: 'loading' })

  useEffect(() => {
    // Proxied to the FastAPI service by vite.config.ts; absent API just means "offline".
    fetch('/api/work-orders?status_filter=open')
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((orders: LiveOrder[]) => setLive({ state: 'ok', orders }))
      .catch(() => setLive({ state: 'offline' }))
  }, [])

  if (!data.fleet) return null
  const k = data.deployed_k
  const engines = data.fleet.engines
  const due = engines.filter((e) => e.due).sort((a, b) => a.pred - b.pred)
  const others = engines.filter((e) => !e.due)
  const correct = due.filter((e) => e.true < k).length
  const early = due.filter((e) => e.true >= k).map((e) => e.true)
  const minMissed = Math.min(...others.map((e) => e.true))
  const missedUrgent = others.filter((e) => e.true < 8).length

  return (
    <Section
      id="orders"
      num="07 · In production"
      title="Tonight's run: 100 engines scored, work orders raised"
      lede={`A nightly job scores every engine in the fleet and posts a work order for each one predicted to have fewer than ${k} flights left. Here the "fleet" is NASA's 100 test engines, caught mid-life, so we can check the job against what really happened.`}
    >
      <div className="tiles">
        <Tile label="Engines scored" value={engines.length} note={`run ${data.fleet.run_date}`} />
        <Tile label="Work orders raised" value={due.length} note={`predicted < ${k} flights left`} />
        <Tile label="Truly due (< 15 left)" value={`${correct} of ${due.length}`} note={early.length ? `the rest had ${Math.min(...early)}–${Math.max(...early)} left: early, not wrong` : 'all correct'} />
        <Tile label="Urgent engines missed (< 8 left)" value={missedUrgent} note={`fewest flights left among unflagged: ${minMissed}`} />
      </div>
      <div className="grid-2">
        <div className="card">
          <h3>Predicted vs. actual flights left, every engine</h3>
          <p className="card-sub">Points on the diagonal are perfect predictions.</p>
          <div className="chart">
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 16, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid stroke={COLORS.grid} />
                <XAxis type="number" dataKey="true" name="actual" {...AXIS_PROPS} domain={[0, 150]} label={{ value: 'actual flights left', position: 'insideBottom', offset: -4, fill: COLORS.muted, fontSize: 12 }} height={40} />
                <YAxis type="number" dataKey="pred" name="predicted" {...AXIS_PROPS} domain={[0, 150]} width={36} label={{ value: 'predicted', angle: -90, position: 'insideLeft', fill: COLORS.muted, fontSize: 12 }} />
                <ReferenceArea y1={0} y2={k} fill={COLORS.band} />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 150, y: 150 }]} stroke={COLORS.axis} />
                <ReferenceLine y={k} stroke={COLORS.model} strokeDasharray="4 3" label={{ value: `work order below ${k}`, position: 'insideBottomRight', fill: 'var(--text-secondary)', fontSize: 11 }} />
                <Tooltip
                  cursor={false}
                  content={(p: any) =>
                    p.active && p.payload?.length ? (
                      <TooltipBox
                        title={`Engine #${p.payload[0].payload.unit}${p.payload[0].payload.due ? ' · work order' : ''}`}
                        rows={[
                          { color: COLORS.model, value: fmt(p.payload[0].payload.pred), label: 'predicted' },
                          { color: COLORS.neutral, value: p.payload[0].payload.true, label: 'actual' },
                        ]}
                      />
                    ) : null
                  }
                />
                <Scatter data={others} fill={COLORS.neutral} stroke="var(--surface-1)" strokeWidth={2} isAnimationActive={false} />
                <Scatter data={due} fill={COLORS.model} stroke="var(--surface-1)" strokeWidth={2} isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <Legend
            items={[
              { label: 'Work order raised', color: COLORS.model },
              { label: 'No action tonight', color: COLORS.neutral },
            ]}
          />
        </div>
        <div className="card">
          <h3>Work orders</h3>
          <p className="card-sub">
            {live.state === 'ok'
              ? `Live from the work-order API: ${live.orders.length} open.`
              : live.state === 'loading'
                ? 'Checking the work-order API…'
                : 'From the last batch run. Start the API to see live orders.'}
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Engine</th>
                  <th className="num">Last flight</th>
                  <th className="num">Predicted left</th>
                  <th className="num">Actually left</th>
                  {live.state === 'ok' && <th>Order</th>}
                </tr>
              </thead>
              <tbody>
                {due.map((e) => {
                  const order = live.state === 'ok' ? live.orders.find((o) => o.unit === e.unit) : undefined
                  return (
                    <tr key={e.unit}>
                      <td>#{e.unit}</td>
                      <td className="num">{e.last_cycle}</td>
                      <td className="num">{fmt(e.pred)}</td>
                      <td className="num">{e.true}</td>
                      {live.state === 'ok' && <td>{order ? `#${order.id} ${order.status}` : <span className="muted">none</span>}</td>}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          {live.state === 'offline' && (
            <details>
              <summary>How to connect the live API</summary>
              <p style={{ marginTop: 8 }}>
                Run <code>uv run uvicorn api:app --app-dir src</code> then <code>uv run python src/batch.py</code> from the project
                folder, and reload this page.
              </p>
            </details>
          )}
        </div>
      </div>
    </Section>
  )
}
