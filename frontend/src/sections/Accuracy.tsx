import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { DashboardData } from '../types'
import { AXIS_PROPS, COLORS, Section, TooltipBox, fmt } from '../ui'

export function Accuracy({ data }: { data: DashboardData }) {
  const get = (name: string) => data.accuracy.find((a) => a.model === name)!
  const lin = get('linear')
  const lgbm = get('lightgbm')
  const leaky = get('lightgbm-LEAKY-row-split')

  const rows = [
    { name: 'Straight-line baseline', value: lin.test_rmse!, color: COLORS.neutral, note: 'unseen engines', leaky: false },
    { name: 'LightGBM', value: lgbm.test_rmse!, color: COLORS.model, note: 'unseen engines', leaky: false },
    { name: 'LightGBM, row split', value: leaky.cv_rmse, color: COLORS.model, note: 'leaky: same engine in train and test', leaky: true },
  ]

  return (
    <Section
      id="accuracy"
      num="03 · How accurate"
      title="Typically about 12 flights off, on engines it has never seen"
      lede="Error measured on NASA's 100 held-out test engines (RMSE, in flights; lower is better). The last bar is a warning: split the data by row instead of by engine and the model looks twice as good. That number is fake."
    >
      <div className="grid-2">
        <div className="card">
          <h3>Prediction error (flights)</h3>
          <div className="chart short">
            <ResponsiveContainer>
              <BarChart data={rows} layout="vertical" margin={{ top: 8, right: 56, bottom: 8, left: 8 }}>
                <defs>
                  <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                    <rect width="6" height="6" fill="var(--band)" />
                    <line x1="0" y1="0" x2="0" y2="6" stroke="var(--series-1)" strokeWidth="2" />
                  </pattern>
                </defs>
                <CartesianGrid horizontal={false} stroke={COLORS.grid} />
                <XAxis type="number" {...AXIS_PROPS} domain={[0, 18]} />
                <YAxis type="category" dataKey="name" {...AXIS_PROPS} width={150} />
                <Tooltip
                  cursor={{ fill: 'var(--wash)' }}
                  content={(p: any) =>
                    p.active && p.payload?.length ? (
                      <TooltipBox
                        title={p.payload[0].payload.name}
                        rows={[{ color: p.payload[0].payload.color, value: `${fmt(p.payload[0].payload.value)} flights`, label: p.payload[0].payload.note }]}
                      />
                    ) : null
                  }
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={24} isAnimationActive={false}>
                  {rows.map((r) => (
                    <Cell key={r.name} fill={r.leaky ? 'url(#hatch)' : r.color} />
                  ))}
                  <LabelList
                    dataKey="value"
                    position="right"
                    fill="var(--text-primary)"
                    fontSize={12}
                    formatter={(v: any) => fmt(Number(v))}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="card-sub">Hatched bar: invalid test (engine leakage). Shown only as a warning.</p>
        </div>
        <div className="card">
          <h3>What these numbers mean</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th className="num">Error</th>
                  <th className="num">PHM08 score</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Straight-line baseline</td>
                  <td className="num">{fmt(lin.test_rmse!)}</td>
                  <td className="num">{fmt(lin.test_phm08!, 0)}</td>
                </tr>
                <tr className="highlight">
                  <td>LightGBM</td>
                  <td className="num">{fmt(lgbm.test_rmse!)}</td>
                  <td className="num">{fmt(lgbm.test_phm08!, 0)}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="callout">
            <strong>PHM08</strong> is NASA's score that punishes predicting <em>too late</em> (engine fails first) harder than
            too early. LightGBM nearly halves it: {fmt(lin.test_phm08!, 0)} → {fmt(lgbm.test_phm08!, 0)}.
          </p>
          <p className="callout">
            <strong>Why the split matters:</strong> with a row split the model partly memorises each engine, scoring{' '}
            {fmt(leaky.cv_rmse)}. On a genuinely new engine it would get ~{fmt(lgbm.cv_rmse)}. This is the most common way these
            projects oversell.
          </p>
        </div>
      </div>
    </Section>
  )
}
