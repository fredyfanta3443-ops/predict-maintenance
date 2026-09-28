import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { DashboardData } from '../types'
import { AXIS_PROPS, COLORS, Section, Tile, TooltipBox, fmt } from '../ui'

const colorFor = (policy: string) =>
  policy.includes('model') ? COLORS.model : policy.includes('s11') ? COLORS.sensor : COLORS.fixed

const labelFor = (policy: string) =>
  policy.includes('model') ? 'LightGBM (k = 15)' : policy.includes('s11') ? 'One-sensor rule' : 'Fixed schedule'

export function Robustness({ data }: { data: DashboardData }) {
  const g = data.generalize
  const acc = (tr: string, te: string) => g.accuracy.find((a) => a['trained on'] === tr && a['tested on'] === te)!
  const same = acc('FD001', 'FD001')
  const shift = acc('FD001', 'FD003 (shift)')
  const retrained = acc('FD003', 'FD003')
  const asIs = g.as_is.map((a) => ({ ...a, name: labelFor(a.policy), color: colorFor(a.policy) }))

  return (
    <Section
      id="robustness"
      num="05 · Stress test"
      title="What happens when engines start failing a new way?"
      lede="Everything so far used FD001, where every engine fails the same way (compressor wear). FD003 adds a second failure type (fan wear). We deployed each policy onto FD003 exactly as tuned on FD001, with no new data, as a client would on day one."
    >
      <div className="tiles">
        <Tile label="Model error, same conditions" value={`${fmt(same.test_rmse)} flights`} />
        <Tile label="Model error, new failure type" value={`${fmt(shift.test_rmse)} flights`} note={`predicts too late on ${fmt(g.late_share * 100, 0)}% of engines`} />
        <Tile label="After retraining on FD003" value={`${fmt(retrained.test_rmse)} flights`} note="recovers once it sees the new failure type" />
      </div>
      <div className="grid-2">
        <div className="card">
          <h3>Engines failing in service, per 100 (FD003, policies unchanged)</h3>
          <div className="chart short">
            <ResponsiveContainer>
              <BarChart data={asIs} layout="vertical" margin={{ top: 8, right: 48, bottom: 8, left: 8 }}>
                <CartesianGrid horizontal={false} stroke={COLORS.grid} />
                <XAxis type="number" {...AXIS_PROPS} domain={[0, 50]} />
                <YAxis type="category" dataKey="name" {...AXIS_PROPS} width={160} />
                <Tooltip
                  cursor={{ fill: 'var(--wash)' }}
                  content={(p: any) =>
                    p.active && p.payload?.length ? (
                      <TooltipBox
                        title={p.payload[0].payload.name}
                        rows={[
                          { color: p.payload[0].payload.color, value: `${p.payload[0].payload.failures} / 100`, label: 'failures' },
                          { color: p.payload[0].payload.color, value: fmt(p.payload[0].payload.cost, 2), label: 'cost / 1k flights' },
                        ]}
                      />
                    ) : null
                  }
                />
                <Bar dataKey="failures" radius={[0, 4, 4, 0]} maxBarSize={24} minPointSize={2} isAnimationActive={false}>
                  {asIs.map((a) => (
                    <Cell key={a.policy} fill={a.color} />
                  ))}
                  <LabelList dataKey="failures" position="right" fill="var(--text-primary)" fontSize={12} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="callout">
            <strong>The ML model is fragile:</strong> it only ever saw compressor wear. Most likely, fan wear looks "healthy"
            to it, so it acts too late. The simple rules survive. This is why the drift monitor (next section) exists.
          </p>
        </div>
        <div className="card">
          <h3>Does ML earn its place? Savings of the LightGBM policy</h3>
          <p className="card-sub">Each model trained on its own dataset. 95% range from resampling engines.</p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Compared with</th>
                  <th className="num">FD001</th>
                  <th className="num">FD003</th>
                </tr>
              </thead>
              <tbody>
                {g.savings.FD001.map((s, i) => {
                  const s3 = g.savings.FD003[i]
                  const cell = (x: typeof s) => (
                    <>
                      <span className={x.lo > 0 ? 'up-good' : ''}>{fmt(x.saving)}%</span>
                      <div className="muted" style={{ fontSize: 12 }}>
                        {fmt(x.lo, 0)} to {fmt(x.hi, 0)}% {x.lo > 0 ? '✓ proven' : '– not proven'}
                      </div>
                    </>
                  )
                  return (
                    <tr key={s.vs}>
                      <td>{s.vs.replace(' (s11 up)', ' (s11)').replace('Best one-sensor rule', 'Best single sensor')}</td>
                      <td className="num">{cell(s)}</td>
                      <td className="num">{cell(s3)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="callout">
            <strong>Honest answer:</strong> the big, proven saving comes from moving off the fixed schedule. ML's edge over a
            well-chosen single sensor is within the noise so far.
          </p>
        </div>
      </div>
    </Section>
  )
}
