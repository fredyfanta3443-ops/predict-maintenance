import { useState } from 'react'
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { DashboardData } from '../types'
import { AXIS_PROPS, COLORS, Legend, Section, Tile, TooltipBox, fmt } from '../ui'

const POLICY_ROWS: { key: string; label: string; color: string; setting: (knob: string) => string }[] = [
  { key: 'Run to failure', label: 'Never service early', color: COLORS.neutral, setting: () => '–' },
  { key: 'Fixed schedule', label: 'Fixed schedule (today)', color: COLORS.fixed, setting: (k) => `every ${parseFloat(k)} flights` },
  { key: 'One-sensor rule (s11)', label: 'One-sensor rule', color: COLORS.sensor, setting: (k) => `s11 > ${k}` },
  { key: 'Predictive (linear)', label: 'Straight-line model', color: COLORS.neutral, setting: (k) => `predicted < ${parseFloat(k)}` },
  { key: 'Predictive (LightGBM)', label: 'LightGBM model', color: COLORS.model, setting: (k) => `predicted < ${parseFloat(k)}` },
  { key: 'Perfect foresight', label: 'Perfect foresight (floor)', color: COLORS.neutral, setting: () => '–' },
]

export function Money({ data }: { data: DashboardData }) {
  const [k, setK] = useState(data.deployed_k)
  const lv = data.policy_levels
  const fixed = lv['Fixed schedule'].cost
  const sensor = lv['One-sensor rule (s11)'].cost
  const oracle = lv['Perfect foresight'].cost
  const at = data.cost_curve.find((p) => p.k === k)!
  const yMax = Math.ceil(fixed * 1.35)

  return (
    <Section
      id="money"
      num="04 · The money"
      title="Service when predicted flights left drop below k"
      lede={
        <>
          Each policy is replayed on all 100 engines. A planned service costs 1; an unplanned failure (with downtime) is
          assumed to cost <strong>{data.failure_cost}</strong> — an assumption the client must confirm. Cost is per 1,000
          flights, so servicing too early is penalised too (you service more often).
        </>
      }
    >
      <div className="card">
        <h3>Total maintenance cost by threshold k</h3>
        <p className="card-sub">Too low and engines fail before service; too high and healthy life is wasted.</p>
        <div className="controls">
          <label htmlFor="k">Threshold k</label>
          <input id="k" type="range" min={1} max={80} value={k} onChange={(e) => setK(+e.target.value)} />
          <output htmlFor="k">{k}</output>
          <span className="muted">deployed: k = {data.deployed_k}</span>
        </div>
        <div className="tiles">
          <Tile label={`Cost per 1,000 flights at k = ${k}`} value={fmt(at.lgbm, 2)} note={`fixed schedule: ${fmt(fixed, 2)}`} />
          <Tile label="Engines failing in service" value={`${at.failures} / 100`} note={at.failures ? 'k too low' : 'none'} />
          <Tile label="Average flights per engine" value={fmt(at.avg_flights, 0)} note={`fixed schedule: ${fmt(lv['Fixed schedule'].avg_flights, 0)}`} />
        </div>
        <div className="chart" style={{ height: 340 }}>
          <ResponsiveContainer>
            <ComposedChart data={data.cost_curve} margin={{ top: 16, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid vertical={false} stroke={COLORS.grid} />
              <XAxis dataKey="k" type="number" domain={[1, 80]} {...AXIS_PROPS} label={{ value: 'service when predicted flights left < k', position: 'insideBottom', offset: -4, fill: COLORS.muted, fontSize: 12 }} height={40} />
              <YAxis {...AXIS_PROPS} width={36} domain={[Math.floor(oracle * 0.9), yMax]} allowDataOverflow />
              <Tooltip
                cursor={{ stroke: COLORS.axis }}
                content={(p: any) =>
                  p.active && p.payload?.length ? (
                    <TooltipBox
                      title={`k = ${p.label} · ${p.payload[0].payload.failures} failures / 100`}
                      rows={[
                        { color: COLORS.model, value: fmt(p.payload[0].payload.lgbm, 2), label: 'LightGBM' },
                        { color: COLORS.neutral, value: fmt(p.payload[0].payload.linear, 2), label: 'straight-line model' },
                        { color: COLORS.fixed, value: fmt(fixed, 2), label: 'fixed schedule' },
                        { color: COLORS.sensor, value: fmt(sensor, 2), label: 'one-sensor rule' },
                      ]}
                    />
                  ) : null
                }
              />
              <Area dataKey={(d: any) => [d.lo, d.hi]} stroke="none" fill={COLORS.band} isAnimationActive={false} />
              <ReferenceLine y={fixed} stroke={COLORS.fixed} strokeWidth={2} label={{ value: `fixed schedule ${fmt(fixed, 2)}`, position: 'insideTopRight', fill: 'var(--text-secondary)', fontSize: 11 }} />
              <ReferenceLine y={sensor} stroke={COLORS.sensor} strokeWidth={2} strokeDasharray="6 4" label={{ value: `one-sensor rule ${fmt(sensor, 2)}`, position: 'insideTopRight', fill: 'var(--text-secondary)', fontSize: 11 }} />
              <ReferenceLine y={oracle} stroke={COLORS.neutral} label={{ value: `perfect foresight ${fmt(oracle, 2)}`, position: 'insideBottomRight', fill: 'var(--text-muted)', fontSize: 11 }} />
              <ReferenceLine x={k} stroke={COLORS.axis} strokeWidth={1} />
              <Line dataKey="linear" stroke={COLORS.neutral} strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
              <Line dataKey="lgbm" stroke={COLORS.model} strokeWidth={2} dot={false} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <Legend
          items={[
            { label: 'LightGBM policy', color: COLORS.model, kind: 'line' },
            { label: '95% range (resampled engines)', color: COLORS.band },
            { label: 'Straight-line model', color: COLORS.neutral, kind: 'dash' },
            { label: 'Fixed schedule (best interval)', color: COLORS.fixed, kind: 'line' },
            { label: 'One-sensor rule (best threshold)', color: COLORS.sensor, kind: 'dash' },
          ]}
        />
        <p className="callout">
          <strong>The cliff at k ≈ 8:</strong> below it, engines start failing and cost shoots up. We deploy at{' '}
          <strong>k = {data.deployed_k}</strong>: about 3% more cost, bought as a safety margin.
        </p>
      </div>

      <div className="grid-2">
        <div className="card">
          <h3>Every policy, tuned honestly</h3>
          <p className="card-sub">Each setting chosen on 80 engines and scored on the other 20, repeated 100×.</p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Policy</th>
                  <th>Setting</th>
                  <th className="num">Cost / 1k flights</th>
                  <th className="num">Failures / 100</th>
                </tr>
              </thead>
              <tbody>
                {POLICY_ROWS.map((r) => (
                  <tr key={r.key} className={r.key === 'Predictive (LightGBM)' ? 'highlight' : ''}>
                    <td>
                      <span className="legend-item">
                        <span className="key-rect" style={{ background: r.color }} />
                        {r.label}
                      </span>
                    </td>
                    <td className="muted">{r.setting(lv[r.key].knob)}</td>
                    <td className="num">{fmt(lv[r.key].cost, 2)}</td>
                    <td className="num">{fmt(lv[r.key].failures, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="card">
          <h3>What if a failure costs more or less than {data.failure_cost}×?</h3>
          <p className="card-sub">Saving of the LightGBM policy, every policy re-tuned at each cost.</p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Failure cost (× planned)</th>
                  <th className="num">Saving vs fixed schedule</th>
                  <th className="num">Saving vs one-sensor rule</th>
                </tr>
              </thead>
              <tbody>
                {data.sensitivity.map((s) => (
                  <tr key={s.failure_cost} className={s.failure_cost === data.failure_cost ? 'highlight' : ''}>
                    <td>{s.failure_cost}×</td>
                    <td className="num up-good">{fmt(s.vs_fixed)}%</td>
                    <td className="num">{fmt(s.vs_sensor)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="callout">
            The headline barely moves: whatever the true cost ratio, moving off the fixed schedule saves{' '}
            <strong>
              {fmt(Math.min(...data.sensitivity.map((s) => s.vs_fixed)), 0)}–{fmt(Math.max(...data.sensitivity.map((s) => s.vs_fixed)), 0)}%
            </strong>
            .
          </p>
        </div>
      </div>
    </Section>
  )
}
