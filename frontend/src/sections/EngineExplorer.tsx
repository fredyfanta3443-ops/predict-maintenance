import { useMemo, useState } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { DashboardData } from '../types'
import { AXIS_PROPS, COLORS, Legend, Section, Tile, TooltipBox, fmt } from '../ui'

const CAP = 125

export function EngineExplorer({ data }: { data: DashboardData }) {
  const [unit, setUnit] = useState('3')
  const e = data.engines[unit]
  const k = data.deployed_k
  const s11Threshold = parseFloat(data.policy_levels['One-sensor rule (s11)'].knob)

  const rows = useMemo(
    () => e.cycle.map((c, i) => ({ cycle: c, true: e.true[i], cappedTrue: Math.min(e.true[i], CAP), pred: e.pred[i], s11: e.s11[i] })),
    [e],
  )
  const T = e.cycle[e.cycle.length - 1]
  // Acting on the failure flight itself is too late, same as the Python simulation.
  const modelIdx = e.pred.slice(0, -1).findIndex((p) => p < k)
  const sensorIdx = e.s11.slice(0, -1).findIndex((s) => s > s11Threshold)
  const trigger = (i: number) => (i >= 0 ? { cycle: e.cycle[i], left: T - e.cycle[i] } : null)
  const model = trigger(modelIdx)
  const sensor = trigger(sensorIdx)
  const rmse = Math.sqrt(rows.reduce((a, r) => a + (r.pred - r.cappedTrue) ** 2, 0) / rows.length)

  return (
    <Section
      id="engine"
      num="02 · How the model sees wear"
      title="Reading an engine's sensors to estimate flights left"
      lede="Sensors don't say how long an engine has left. The model learns it from engines that ran to failure: how 14 sensors drift, averaged over the last 5, 10 and 30 flights. Every prediction below was made blind, by a model that never saw this engine."
    >
      <div className="card">
        <div className="controls" style={{ marginTop: 0 }}>
          <label htmlFor="unit">Engine</label>
          <select id="unit" value={unit} onChange={(ev) => setUnit(ev.target.value)}>
            {Object.keys(data.engines).map((u) => (
              <option key={u} value={u}>
                #{u} (failed on flight {data.engines[u].cycle.at(-1)})
              </option>
            ))}
          </select>
        </div>
        <div className="tiles">
          <Tile label="Failed on flight" value={T} />
          <Tile
            label={`Model work order (predicted < ${k})`}
            value={model ? `flight ${model.cycle}` : 'missed'}
            note={model ? `${model.left} flights before failure` : 'engine would fail in service'}
          />
          <Tile
            label={`One-sensor rule (s11 > ${s11Threshold})`}
            value={sensor ? `flight ${sensor.cycle}` : 'missed'}
            note={sensor ? `${sensor.left} flights before failure` : 'engine would fail in service'}
          />
          <Tile label="Typical error, this engine" value={`${fmt(rmse, 1)} flights`} note="vs. true flights left, capped at 125" />
        </div>

        <h3 style={{ marginTop: 20 }}>Flights left: true vs. predicted</h3>
        <div className="chart">
          <ResponsiveContainer>
            <LineChart data={rows} margin={{ top: 16, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid vertical={false} stroke={COLORS.grid} />
              <XAxis dataKey="cycle" type="number" domain={[1, T]} {...AXIS_PROPS} label={{ value: 'flight number', position: 'insideBottom', offset: -4, fill: COLORS.muted, fontSize: 12 }} height={40} />
              <YAxis {...AXIS_PROPS} width={36} domain={[0, 'dataMax']} />
              <Tooltip
                cursor={{ stroke: COLORS.axis }}
                content={(p: any) =>
                  p.active && p.payload?.length ? (
                    <TooltipBox
                      title={`Flight ${p.label}`}
                      rows={[
                        { color: COLORS.model, value: fmt(p.payload[0].payload.pred, 0), label: 'predicted flights left' },
                        { color: COLORS.neutral, value: p.payload[0].payload.true, label: 'true flights left' },
                      ]}
                    />
                  ) : null
                }
              />
              <ReferenceLine y={CAP} stroke={COLORS.axis} label={{ value: 'cap: 125 = "healthy"', position: 'insideTopLeft', fill: COLORS.muted, fontSize: 11 }} />
              <ReferenceLine y={k} stroke={COLORS.model} strokeDasharray="4 3" label={{ value: `service threshold k = ${k}`, position: 'insideBottomLeft', fill: 'var(--text-secondary)', fontSize: 11 }} />
              {model && <ReferenceLine x={model.cycle} stroke={COLORS.model} label={{ value: 'work order', position: 'top', fill: 'var(--text-secondary)', fontSize: 11 }} />}
              <Line dataKey="true" stroke={COLORS.neutral} strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
              <Line dataKey="pred" stroke={COLORS.model} strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <Legend
          items={[
            { label: 'Model prediction', color: COLORS.model, kind: 'line' },
            { label: 'True flights left', color: COLORS.neutral, kind: 'dash' },
          ]}
        />

        <h3 style={{ marginTop: 24 }}>Sensor s11 (5-flight average): what the simple rule watches</h3>
        <div className="chart short">
          <ResponsiveContainer>
            <LineChart data={rows} margin={{ top: 16, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid vertical={false} stroke={COLORS.grid} />
              <XAxis dataKey="cycle" type="number" domain={[1, T]} {...AXIS_PROPS} height={24} />
              <YAxis {...AXIS_PROPS} width={48} domain={['auto', 'auto']} tickFormatter={(v) => v.toFixed(1)} />
              <Tooltip
                cursor={{ stroke: COLORS.axis }}
                content={(p: any) =>
                  p.active && p.payload?.length ? (
                    <TooltipBox title={`Flight ${p.label}`} rows={[{ color: COLORS.sensor, value: p.payload[0].payload.s11.toFixed(2), label: 's11' }]} />
                  ) : null
                }
              />
              <ReferenceLine y={s11Threshold} stroke={COLORS.sensor} strokeDasharray="4 3" label={{ value: `rule threshold ${s11Threshold}`, position: 'insideTopLeft', fill: 'var(--text-secondary)', fontSize: 11 }} />
              {sensor && <ReferenceLine x={sensor.cycle} stroke={COLORS.sensor} />}
              <Line dataKey="s11" stroke={COLORS.sensor} strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="callout">
          The prediction sits flat at ~125 while the engine is healthy, then counts down as wear shows. Early life is
          capped at 125 because the sensors look the same at 300 or 150 flights left.
        </p>
      </div>
    </Section>
  )
}
