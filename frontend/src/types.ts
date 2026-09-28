// Shape of public/data.json, written by src/export_dashboard.py.

export interface EngineSeries {
  cycle: number[]
  true: number[] // real flights left (uncapped)
  pred: number[] // model's blind prediction (out-of-fold)
  s11: number[] // sensor s11, 5-flight average
}

export interface CurvePoint {
  k: number
  lgbm: number
  lo: number
  hi: number
  linear: number
  failures: number
  avg_flights: number
}

export interface Level {
  cost: number
  failures: number
  avg_flights: number
  knob: string
}

export type Status = 'OK' | 'WATCH' | 'ALERT'

export interface DriftFleet {
  overall: Status
  auc: number
  n_alert: number
  sensors: { sensor: string; psi: number; status: Status }[]
}

export interface Saving {
  vs: string
  saving: number
  lo: number
  hi: number
  p_positive: number
}

export interface DashboardData {
  failure_cost: number
  deployed_k: number
  headline: { saving: number; lo: number; hi: number }
  life: number[]
  engines: Record<string, EngineSeries>
  accuracy: { model: string; cv_rmse: number; test_rmse: number | null; test_phm08: number | null }[]
  cost_curve: CurvePoint[]
  policy_levels: Record<string, Level>
  sensitivity: { failure_cost: number; vs_fixed: number; vs_sensor: number }[]
  generalize: {
    accuracy: { 'trained on': string; 'tested on': string; cv_rmse: number; test_rmse: number; test_phm08: number }[]
    late_share: number
    late_20_share: number
    policies: Record<string, { policy: string; knob: string; cost: number; failures: number; avg_flights: number }[]>
    as_is: { policy: string; cost: number; failures: number }[]
    savings: Record<string, Saving[]>
  }
  drift: Record<string, DriftFleet>
  fleet: {
    run_date: string
    engines: { unit: number; last_cycle: number; pred: number; true: number; due: boolean }[]
  } | null
}
