import { useEffect, useMemo, useState } from 'react'
import Curves from './components/Curves'
import GraphReplay, { usePlayback } from './components/GraphReplay'
import PhaseDiagram from './components/PhaseDiagram'
import StepPanel from './components/StepPanel'
import type { Experiment, Q2PhaseDiagram, TraceFile } from './types'
import { fmt } from './lib'

const TRACE_SOURCES: { id: string; file: string; label: string }[] = [
  { id: 'beta1', file: 'traces-shaping-beta1.json', label: 'LSTM · β=1 shaped' },
  { id: 'sparse', file: 'traces-shaping-sparse.json', label: 'LSTM · sparse (β=0)' },
  { id: 'binary', file: 'traces-shaping-binary.json', label: 'LSTM · binary penalty' },
  { id: 'transformer', file: 'traces-policy-transformer.json', label: 'Transformer · β=1' },
]

const EXP_LABELS: Record<string, string> = {
  'shaping-beta1': 'β=1 graded shaping',
  'shaping-sparse': 'β=0 sparse',
  'shaping-binary': 'binary penalty',
  'policy-mlp': 'MLP',
  'policy-transformer': 'Transformer',
}

function useJson<T>(url: string | null): T | null | 'missing' {
  const [data, setData] = useState<T | null | 'missing'>(null)
  useEffect(() => {
    if (!url) return
    let live = true
    setData(null)
    fetch(url)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((d) => live && setData(d))
      .catch(() => live && setData('missing'))
    return () => {
      live = false
    }
  }, [url])
  return data
}

function Replay() {
  const [sourceId, setSourceId] = useState(TRACE_SOURCES[0].id)
  const source = TRACE_SOURCES.find((s) => s.id === sourceId)!
  const traces = useJson<TraceFile>(`data/${source.file}`)
  const [epIdx, setEpIdx] = useState(0)

  const episode =
    traces && traces !== 'missing' ? (traces.episodes[epIdx] ?? null) : null
  const numFrames = episode ? episode.steps.length : 0
  const { frame, setFrame, playing, setPlaying, speed, setSpeed } =
    usePlayback(numFrames)

  useEffect(() => setEpIdx(0), [sourceId])

  const walkCost = useMemo(() => {
    if (!episode) return 0
    return episode.steps
      .slice(0, Math.min(frame, episode.steps.length))
      .reduce((a, s) => a + (s.v >= 0 ? s.w : 0), 0)
  }, [episode, frame])

  if (traces === 'missing') {
    return (
      <div className="rounded-lg border border-ring bg-surface p-8 text-ink-2">
        <p className="mb-2 text-ink">
          No trace data yet — <code>data/{source.file}</code> is missing.
        </p>
        <p>
          Run the ablations, then export traces with{' '}
          <code className="text-s1">scripts/export_traces.py</code> into{' '}
          <code>viz/public/data/</code>.
        </p>
      </div>
    )
  }
  if (!traces || !episode) {
    return <div className="p-8 text-ink-3">loading…</div>
  }

  return (
    <div className="fade-up">
      {/* controls row */}
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <select
          value={sourceId}
          onChange={(e) => setSourceId(e.target.value)}
          className="rounded-md border border-ring bg-surface-2 px-3 py-1.5 text-ink outline-none"
        >
          {TRACE_SOURCES.map((s) => (
            <option key={s.id} value={s.id}>
              {s.label}
            </option>
          ))}
        </select>
        <div className="flex items-center gap-1">
          {traces.episodes.map((ep, i) => (
            <button
              key={i}
              onClick={() => setEpIdx(i)}
              title={`episode ${i + 1} — ${ep.success ? 'success' : 'failed'}, n=${ep.graph.n}`}
              className={`h-7 w-7 rounded-md border text-[11px] transition-colors ${
                i === epIdx
                  ? 'border-s2 bg-s2/20 text-s2'
                  : ep.success
                    ? 'border-ring text-ink-3 hover:border-ink-3'
                    : 'border-critical/40 text-critical/70 hover:border-critical'
              }`}
            >
              {i + 1}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => setPlaying(!playing)}
            className="rounded-md border border-ring bg-surface-2 px-4 py-1.5 text-ink hover:border-ink-3"
          >
            {playing ? '⏸ pause' : '▶ play'}
          </button>
          {[0.5, 1, 2, 4].map((s) => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={`rounded-md border px-2 py-1.5 ${
                speed === s
                  ? 'border-s2 text-s2'
                  : 'border-ring text-ink-3 hover:border-ink-3'
              }`}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(320px,1fr)]">
        {/* stage */}
        <div className="relative aspect-square rounded-lg border border-ring bg-surface">
          <GraphReplay episode={episode} stepIdx={frame} />
          <div className="absolute left-4 top-3 text-[11px] tracking-[0.15em] text-ink-3">
            n={episode.graph.n} · EPISODE {epIdx + 1}/{traces.episodes.length} ·{' '}
            {episode.success ? (
              <span className="text-good">SUCCESS</span>
            ) : (
              <span className="text-critical">TRUNCATED</span>
            )}
          </div>
          <div className="absolute right-4 top-3 text-[11px] text-ink-3">
            cost so far <span className="text-ink">{walkCost}</span> / optimal{' '}
            <span className="text-good">{episode.graph.optimal_cost}</span>
          </div>
        </div>

        {/* instrument panel */}
        <div className="rounded-lg border border-ring bg-surface p-4">
          <StepPanel episode={episode} stepIdx={frame} />
        </div>
      </div>

      {/* scrubber */}
      <div className="mt-4 flex items-center gap-3">
        <span className="w-24 shrink-0 text-right text-ink-3">
          step {Math.min(frame, numFrames)}/{numFrames}
        </span>
        <input
          type="range"
          min={0}
          max={numFrames}
          value={frame}
          onChange={(e) => {
            setPlaying(false)
            setFrame(Number(e.target.value))
          }}
          className="h-1 flex-1 cursor-pointer appearance-none rounded-full bg-surface-2 accent-s2"
        />
      </div>

      {/* phi legend */}
      <div className="mt-3 flex flex-wrap items-center gap-3 text-[11px] text-ink-3">
        <span>φ = distance to target:</span>
        <span
          className="h-2 w-40 rounded-full"
          style={{
            background:
              'linear-gradient(90deg,#cde2fb,#86b6ef,#3987e5,#1c5cab,#0d366b)',
          }}
        />
        <span>near → far</span>
        <span className="ml-4 inline-block h-2 w-6 rounded-full bg-s2" /> walked
        <span className="ml-2 inline-block h-2 w-6 rounded-full bg-good/30" />{' '}
        optimal path
        <span className="ml-2 text-s1">edge glow = π(a|s)</span>
      </div>
    </div>
  )
}

function Experiments() {
  const exps = useJson<Experiment[]>('data/experiments.json')
  if (exps === 'missing') {
    return (
      <div className="rounded-lg border border-ring bg-surface p-8 text-ink-2">
        No experiment curves yet — run{' '}
        <code className="text-s1">scripts/run_experiments.py</code> and copy{' '}
        <code>viz-data/*.json</code> into <code>viz/public/data/</code>.
      </div>
    )
  }
  if (!exps) return <div className="p-8 text-ink-3">loading…</div>
  return (
    <div className="fade-up">
      <Curves
        experiments={exps}
        names={['shaping-beta1', 'shaping-sparse', 'shaping-binary']}
        title="Shaping ablation"
        subtitle="same PPO + LSTM, three ways to say “wrong direction” — graded potential shaping vs nothing vs a flat slap"
        labels={EXP_LABELS}
      />
      <Curves
        experiments={exps}
        names={['policy-mlp', 'shaping-beta1', 'policy-transformer']}
        title="Policy ablation"
        subtitle="β=1 shaping held fixed — does memory or attention over neighbor slots matter?"
        labels={{ ...EXP_LABELS, 'shaping-beta1': 'LSTM' }}
      />
    </div>
  )
}

function Q2Phase() {
  const data = useJson<Q2PhaseDiagram>('data/q2-phase-diagram.json')
  if (data === 'missing') {
    return (
      <div className="rounded-lg border border-ring bg-surface p-8 text-ink-2">
        Q2 phase-diagram data is missing from{' '}
        <code>data/q2-phase-diagram.json</code>.
      </div>
    )
  }
  if (!data) return <div className="p-8 text-ink-3">loading…</div>
  return <PhaseDiagram data={data} />
}

export default function App() {
  const [tab, setTab] =
    useState<'replay' | 'experiments' | 'phase'>('replay')
  return (
    <div className="blueprint min-h-screen">
      <div className="mx-auto max-w-[1500px] px-6 py-8">
        <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="mb-1 text-[11px] tracking-[0.3em] text-s2">
              DIJKWALK · PUFFERLIB PPO · POTENTIAL-BASED SHAPING
            </div>
            <h1 className="font-display text-6xl leading-none text-ink">
              Dijkstra, <span className="italic text-s1">learned.</span>
            </h1>
            <p className="mt-3 max-w-2xl leading-relaxed text-ink-2">
              An agent walks random weighted graphs toward a target. Each step
              pays the edge and earns β·(φ(u)−φ(v)) — with β=1 the reward is
              exactly −Δ/w<sub>max</sub>, the Dijkstra regret of the move.
              Wrong directions are punished in proportion to how wrong they
              are.
            </p>
          </div>
          <nav className="flex gap-2">
            {(
              [
                ['replay', 'WALK REPLAY'],
                ['experiments', 'ABLATIONS'],
                ['phase', 'PHASE DIAGRAM'],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                className={`rounded-md border px-5 py-2.5 text-[12px] tracking-[0.15em] transition-colors ${
                  tab === id
                    ? 'border-s2 bg-s2/15 text-s2'
                    : 'border-ring text-ink-3 hover:border-ink-3 hover:text-ink-2'
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
        </header>

        {tab === 'replay' ? (
          <Replay />
        ) : tab === 'experiments' ? (
          <Experiments />
        ) : (
          <Q2Phase />
        )}

        <footer className="mt-16 border-t border-ring pt-4 text-[11px] text-ink-3">
          reward r = (−w + β·(φ(u)−φ(v)))/w_max − c · shaped per Ng et al. 1999
          · curriculum n∈[8,16]→32 · trained {fmt.steps(5_000_000)} steps per
          run on CPU at ~60–95k SPS
        </footer>
      </div>
    </div>
  )
}
