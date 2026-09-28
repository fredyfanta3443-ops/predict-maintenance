import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { DashboardData, DriftFleet } from '../types'
import { AXIS_PROPS, COLORS, Legend, Section, StatusBadge, TooltipBox, statusColor } from '../ui'

const ALERT = 0.2

function FleetPanel({ name, fleet, yMax }: { name: string; fleet: DriftFleet; yMax: number }) {
  const rows = [...fleet.sensors].sort((a, b) => +a.sensor.slice(1) - +b.sensor.slice(1))
  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'baseline', flexWrap: 'wrap' }}>
        <h3>{name}</h3>
        <StatusBadge status={fleet.overall} />
      </div>
      <p className="card-sub">
        {fleet.n_alert} of {rows.length} sensors shifted · fleet classifier AUC {fleet.auc.toFixed(2)} (0.5 = identical)
      </p>
      <div className="chart short">
        <ResponsiveContainer>
          <BarChart data={rows} margin={{ top: 16, right: 8, bottom: 8, left: 0 }} barCategoryGap={2}>
            <CartesianGrid vertical={false} stroke={COLORS.grid} />
            <XAxis dataKey="sensor" {...AXIS_PROPS} interval={0} fontSize={11} />
            <YAxis {...AXIS_PROPS} width={36} domain={[0, yMax]} />
            <Tooltip
              cursor={{ fill: 'var(--wash)' }}
              content={(p: any) =>
                p.active && p.payload?.length ? (
                  <TooltipBox
                    title={`Sensor ${p.label}`}
                    rows={[{ color: statusColor(p.payload[0].payload.status), value: p.payload[0].payload.psi.toFixed(3), label: `PSI · ${p.payload[0].payload.status}` }]}
                  />
                ) : null
              }
            />
            <ReferenceLine y={ALERT} stroke={COLORS.critical} strokeDasharray="4 3" label={{ value: 'alert 0.2', position: 'insideTopRight', fill: 'var(--text-secondary)', fontSize: 11 }} />
            <Bar dataKey="psi" radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false}>
              {rows.map((r) => (
                <Cell key={r.sensor} fill={statusColor(r.status)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export function Drift({ data }: { data: DashboardData }) {
  const fleets = Object.entries(data.drift)
  const yMax = Math.ceil(Math.max(...fleets.flatMap(([, f]) => f.sensors.map((s) => s.psi))) * 10) / 10
  return (
    <Section
      id="drift"
      num="06 · Early warning"
      title="A nightly check: does the fleet still look like the training data?"
      lede="Real failure records take weeks to arrive, too late to protect engines. So every night we compare incoming sensor data to what the model learned from, matching engines by age (young engines naturally read healthier). PSI measures how far each sensor's values have moved."
    >
      <div className="grid-2">
        {fleets.map(([name, f]) => (
          <FleetPanel key={name} name={name} fleet={f} yMax={yMax} />
        ))}
      </div>
      <Legend
        items={[
          { label: 'OK: PSI below 0.1', color: COLORS.good },
          { label: 'Watch: 0.1 to 0.2', color: COLORS.warning },
          { label: 'Alert: above 0.2', color: COLORS.critical },
        ]}
      />
      <p className="callout">
        <strong>On ALERT:</strong> stop trusting the model, fall back to the one-sensor rule, collect technician inspection
        findings, and retrain. Had this been running, it would have flagged the FD003 fleet as unlike the training data, the
        warning the model needed before it let 44 engines fail.
      </p>
    </Section>
  )
}
