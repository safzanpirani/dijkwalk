import { useMemo } from 'react'
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import type { Experiment } from '../types'
import { SERIES, fmt } from '../lib'

// first three categorical slots — all-pairs CVD-validated in dark mode
const COLORS = [SERIES.s1, SERIES.s2, SERIES.s3]

interface MetricSpec {
  key: string
  title: string
  note: string
  fmt: (x: number) => string
  domain?: [number | 'auto', number | 'auto']
}

const METRICS: MetricSpec[] = [
  {
    key: 'environment/success_rate',
    title: 'success rate',
    note: 'fraction of episodes that reach the target',
    fmt: fmt.pct,
    domain: [0, 1],
  },
  {
    key: 'environment/opt_gap',
    title: 'optimality gap',
    note: 'path cost ÷ φ(src) on successes — 1.0 is Dijkstra',
    fmt: (x) => fmt.num(x, 2),
    domain: [1, 'auto'],
  },
  {
    key: 'environment/backward_rate',
    title: 'backward-step rate',
    note: 'fraction of moves where φ increased (wrong direction)',
    fmt: fmt.pct,
    domain: [0, 'auto'],
  },
  {
    key: 'environment/episode_length',
    title: 'episode length',
    note: 'mean steps per episode (curriculum grows n over time)',
    fmt: (x) => fmt.num(x, 1),
  },
  {
    key: 'losses/entropy',
    title: 'policy entropy',
    note: 'over valid slots only — masking keeps invalid mass at 0',
    fmt: (x) => fmt.num(x, 2),
  },
  {
    key: 'environment/n_max_cur',
    title: 'curriculum n_max',
    note: 'graph size cap, expands at >90% rolling success',
    fmt: (x) => fmt.num(x, 0),
  },
]

/** Merge experiment curves into one recharts row set keyed by agent_steps. */
function mergeCurves(exps: Experiment[], metric: string) {
  const out = new Map<number, Record<string, number>>()
  for (const e of exps) {
    for (const pt of e.curve) {
      const v = (pt as unknown as Record<string, number>)[metric]
      if (v === undefined) continue
      const s = pt.agent_steps
      if (!out.has(s)) out.set(s, { agent_steps: s })
      out.get(s)![e.name] = v
    }
  }
  return [...out.values()].sort((a, b) => a.agent_steps - b.agent_steps)
}

function MetricChart({
  exps,
  spec,
}: {
  exps: Experiment[]
  spec: MetricSpec
}) {
  const data = useMemo(() => mergeCurves(exps, spec.key), [exps, spec.key])
  return (
    <div className="rounded-lg border border-ring bg-surface p-4 fade-up">
      <div className="mb-1 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-ink">{spec.title}</h3>
      </div>
      <p className="mb-3 text-[11px] leading-snug text-ink-3">{spec.note}</p>
      <div className="h-52">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -14 }}>
            <CartesianGrid stroke="#2c2c2a" strokeDasharray="0" vertical={false} />
            <XAxis
              dataKey="agent_steps"
              tickFormatter={fmt.steps}
              stroke="#383835"
              tick={{ fill: '#898781', fontSize: 11, fontFamily: 'IBM Plex Mono' }}
              tickLine={false}
              type="number"
              domain={['dataMin', 'dataMax']}
            />
            <YAxis
              stroke="#383835"
              tick={{ fill: '#898781', fontSize: 11, fontFamily: 'IBM Plex Mono' }}
              tickLine={false}
              tickFormatter={(v: number) => spec.fmt(v)}
              domain={spec.domain}
              width={62}
            />
            <Tooltip
              contentStyle={{
                background: '#222220',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 6,
                fontFamily: 'IBM Plex Mono',
                fontSize: 12,
              }}
              labelStyle={{ color: '#898781' }}
              labelFormatter={(v) => `${fmt.steps(Number(v))} steps`}
              formatter={(v) => spec.fmt(Number(v))}
            />
            {exps.map((e, i) => (
              <Line
                key={e.name}
                dataKey={e.name}
                stroke={COLORS[i % COLORS.length]}
                strokeWidth={2}
                dot={false}
                connectNulls
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default function Curves({
  experiments,
  names,
  title,
  subtitle,
  labels,
}: {
  experiments: Experiment[]
  names: string[]
  title: string
  subtitle: string
  labels: Record<string, string>
}) {
  const exps = names
    .map((n) => experiments.find((e) => e.name === n))
    .filter((e): e is Experiment => !!e)
  if (!exps.length) return null
  return (
    <section className="mb-12">
      <div className="mb-4 flex flex-wrap items-baseline gap-x-6 gap-y-1">
        <h2 className="font-display text-3xl italic text-ink">{title}</h2>
        <p className="text-ink-3">{subtitle}</p>
      </div>
      {/* legend — color follows the entity across every chart below */}
      <div className="mb-4 flex flex-wrap gap-5">
        {exps.map((e, i) => (
          <span key={e.name} className="flex items-center gap-2 text-ink-2">
            <span
              className="inline-block h-[3px] w-6 rounded-full"
              style={{ background: COLORS[i % COLORS.length] }}
            />
            {labels[e.name] ?? e.name}
            <span className="text-ink-3">
              {(e.n_params / 1000).toFixed(0)}k params
            </span>
          </span>
        ))}
      </div>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {METRICS.map((m) => (
          <MetricChart key={m.key} exps={exps} spec={m} />
        ))}
      </div>
    </section>
  )
}
