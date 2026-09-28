import { useEffect, useState } from 'react'
import { Accuracy } from './sections/Accuracy'
import { Drift } from './sections/Drift'
import { EngineExplorer } from './sections/EngineExplorer'
import { Money } from './sections/Money'
import { Problem } from './sections/Problem'
import { Robustness } from './sections/Robustness'
import { WorkOrders } from './sections/WorkOrders'
import type { DashboardData } from './types'
import { Section, fmt } from './ui'

const NAV = [
  ['problem', 'Problem'],
  ['engine', 'Model'],
  ['accuracy', 'Accuracy'],
  ['money', 'Money'],
  ['robustness', 'Stress test'],
  ['drift', 'Drift'],
  ['orders', 'Work orders'],
  ['verdict', 'Verdict'],
]

function ThemeToggle() {
  const [theme, setTheme] = useState<'light' | 'dark' | null>(null)
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme
  }, [theme])
  const isDark = theme ? theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches
  return (
    <button className="theme-btn" onClick={() => setTheme(isDark ? 'light' : 'dark')}>
      {isDark ? 'Light mode' : 'Dark mode'}
    </button>
  )
}

export default function App() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    fetch('/data.json')
      .then((r) => r.json())
      .then(setData)
      .catch(() => setError(true))
  }, [])

  if (error)
    return <div className="page hero">Could not load data.json. Run <code>uv run python src/export_dashboard.py</code> first.</div>
  if (!data) return <div className="page hero muted">Loading results…</div>

  const lv = data.policy_levels
  const h = data.headline

  return (
    <>
      <nav className="nav">
        <div className="nav-inner">
          <span className="brand">Predictive maintenance</span>
          {NAV.map(([id, label]) => (
            <a key={id} href={`#${id}`}>
              {label}
            </a>
          ))}
          <ThemeToggle />
        </div>
      </nav>
      <main className="page">
        <header className="hero">
          <div className="eyebrow">Turbofan engine fleet · NASA C-MAPSS simulation</div>
          <h1>Service engines when they need it, not when the calendar says so</h1>
          <p className="hero-lede">
            An engine maintenance operation services every engine on a fixed schedule. Some fail before their date; others
            are serviced while healthy. We used sensor data to predict how many flights each engine has left, and priced what
            that is worth.
          </p>
          <div className="hero-figure">{fmt(h.saving, 0)}% lower</div>
          <p className="hero-caption">
            total maintenance cost vs. today's fixed schedule (95% range {fmt(h.lo, 0)}–{fmt(h.hi, 0)}%), assuming a failure
            costs {data.failure_cost}× a planned service.
          </p>
          <div className="tiles">
            <div className="tile">
              <div className="tile-label">Flights per engine before service</div>
              <div className="tile-value">
                {fmt(lv['Fixed schedule'].avg_flights, 0)} → {fmt(lv['Predictive (LightGBM)'].avg_flights, 0)}
              </div>
              <div className="tile-note">fixed schedule → model</div>
            </div>
            <div className="tile">
              <div className="tile-label">Failures per 100 engines</div>
              <div className="tile-value">
                {fmt(lv['Fixed schedule'].failures, 1)} → {fmt(lv['Predictive (LightGBM)'].failures, 1)}
              </div>
              <div className="tile-note">no extra risk</div>
            </div>
            <div className="tile">
              <div className="tile-label">Prediction error</div>
              <div className="tile-value">~12 flights</div>
              <div className="tile-note">on engines never seen in training</div>
            </div>
            <div className="tile">
              <div className="tile-label">vs. a simple one-sensor rule</div>
              <div className="tile-value">Not proven</div>
              <div className="tile-note">ML's extra edge is within the noise</div>
            </div>
          </div>
        </header>

        <Problem data={data} />
        <EngineExplorer data={data} />
        <Accuracy data={data} />
        <Money data={data} />
        <Robustness data={data} />
        <Drift data={data} />
        <WorkOrders data={data} />

        <Section
          id="verdict"
          num="08 · Verdict"
          title="How much will this save, and how sure are we?"
          lede="The answer to the client's question, and what we recommend doing next."
        >
          <div className="verdict">
            <div className="verdict-item">
              <span className="status-dot" style={{ background: 'var(--good)' }} aria-hidden>✓</span>
              <div>
                <strong>Condition-based maintenance saves about {fmt(h.saving, 0)}%</strong>
                <p>Confident: 95% range {fmt(h.lo, 0)}–{fmt(h.hi, 0)}%, holds whether a failure costs 3× or 50× a planned service, and on both datasets.</p>
              </div>
            </div>
            <div className="verdict-item">
              <span className="status-dot" style={{ background: 'var(--warning)' }} aria-hidden>!</span>
              <div>
                <strong>ML is not yet proven better than one well-chosen sensor</strong>
                <p>The s11 rule captures most of the saving and survives a new failure type. The model's extra few percent is inside the noise.</p>
              </div>
            </div>
            <div className="verdict-item">
              <span className="status-dot" style={{ background: 'var(--critical)' }} aria-hidden>✕</span>
              <div>
                <strong>The model breaks when conditions change</strong>
                <p>Deployed unchanged on a new failure type it let 44 of 100 engines fail. The nightly drift check catches this.</p>
              </div>
            </div>
          </div>
          <div className="card">
            <h3>Recommended rollout</h3>
            <table>
              <tbody>
                <tr><td>1</td><td>Make service decisions with the <strong>one-sensor rule</strong>: cheap, explainable, robust.</td></tr>
                <tr><td>2</td><td>Run the ML model in <strong>shadow mode</strong> alongside it, with the nightly drift report.</td></tr>
                <tr><td>3</td><td>Record <strong>technician inspection findings</strong> at every service: they become the new labels.</td></tr>
                <tr><td>4</td><td>Switch to the model only once it beats the rule on the client's own engines.</td></tr>
              </tbody>
            </table>
            <h3 style={{ marginTop: 20 }}>Assumptions the client must confirm</h3>
            <table>
              <tbody>
                <tr><td>Failure cost</td><td>An unplanned failure (with downtime) costs ~{data.failure_cost}× a planned service.</td></tr>
                <tr><td>Timing</td><td>A work order raised overnight is actioned before the engine's next flight.</td></tr>
                <tr><td>Data</td><td>Results come from NASA simulations; every number must be re-measured on the client's real engines.</td></tr>
              </tbody>
            </table>
          </div>
        </Section>
      </main>
    </>
  )
}
