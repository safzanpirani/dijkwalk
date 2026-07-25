import { useMemo } from 'react'
import type { Episode } from '../types'
import { fmt } from '../lib'

/** Right-hand instrument panel: policy distribution over neighbor slots,
 *  reward decomposition of the current step, and the running return. */
export default function StepPanel({
  episode,
  stepIdx,
}: {
  episode: Episode
  stepIdx: number
}) {
  const arrived = stepIdx >= episode.steps.length
  const step = arrived
    ? episode.steps[episode.steps.length - 1]
    : episode.steps[stepIdx]

  const cumReturn = useMemo(
    () =>
      episode.steps
        .slice(0, Math.min(stepIdx + (arrived ? 0 : 1), episode.steps.length))
        .reduce((a, s) => a + s.reward_total, 0),
    [episode, stepIdx, arrived],
  )

  if (!step) return null

  const slots = step.slots
    .map((v, k) => ({ v, k, p: step.probs[k], delta: step.deltas[String(v)] }))
    .filter((s) => s.v >= 0)
    .sort((a, b) => b.p - a.p)

  const parts = [
    { label: '−w(u,v) edge cost', val: step.reward_edge },
    { label: 'β·(φu−φv) shaping', val: step.reward_shaping },
    { label: 'step tax', val: step.reward_step },
  ]
  if (arrived && episode.success) {
    parts.push({ label: 'goal bonus', val: 10 })
  }
  const maxAbs = Math.max(0.2, ...parts.map((p) => Math.abs(p.val)))

  return (
    <div className="flex flex-col gap-5">
      {/* policy distribution */}
      <section>
        <h3 className="mb-2 flex items-center justify-between text-[11px] tracking-[0.18em] text-ink-3">
          <span>π(a | s) — NEIGHBOR SLOTS</span>
          {!arrived && step.probs[step.action] < Math.max(...step.probs) && (
            <span
              className="rounded border border-s2/50 px-1.5 py-0.5 text-[10px] text-s2"
              title="the sampled action was not the argmax — the policy leans on stochasticity to escape 2-cycles (greedy success ≈25%, sampled ≈87%)"
            >
              SAMPLED OFF-ARGMAX
            </span>
          )}
        </h3>
        <div className="flex flex-col gap-1.5">
          {slots.map((s) => {
            const chosen = s.k === step.action
            const wrongDir = s.delta !== undefined && s.delta > 0
            return (
              <div key={s.k} className="flex items-center gap-2">
                <span
                  className={`w-12 shrink-0 text-right ${
                    chosen ? 'text-s2 font-semibold' : 'text-ink-2'
                  }`}
                >
                  →{s.v}
                </span>
                <div className="h-4 flex-1 overflow-hidden rounded-[3px] bg-surface-2">
                  <div
                    className="h-full rounded-[3px] transition-all duration-300"
                    style={{
                      width: `${Math.max(1.5, s.p * 100)}%`,
                      background: chosen ? '#d95926' : '#3987e5',
                      opacity: chosen ? 1 : 0.55,
                    }}
                  />
                </div>
                <span className="w-14 shrink-0 text-right text-ink-2">
                  {fmt.pct(s.p)}
                </span>
                <span
                  className={`w-16 shrink-0 text-right text-[11px] ${
                    wrongDir ? 'text-critical' : 'text-good'
                  }`}
                  title="Dijkstra regret Δ of this move (0 = on a shortest path)"
                >
                  Δ{s.delta !== undefined ? fmt.num(s.delta, 1) : '—'}
                </span>
              </div>
            )
          })}
        </div>
      </section>

      {/* reward decomposition */}
      <section>
        <h3 className="mb-2 text-[11px] tracking-[0.18em] text-ink-3">
          REWARD DECOMPOSITION{arrived ? ' — FINAL STEP' : ''}
        </h3>
        <div className="flex flex-col gap-1.5">
          {parts.map((p) => (
            <div key={p.label} className="flex items-center gap-2">
              <span className="w-40 shrink-0 text-[11px] text-ink-2">
                {p.label}
              </span>
              <div className="relative h-4 flex-1">
                <div className="absolute inset-y-0 left-1/2 w-px bg-axis" />
                <div
                  className="absolute inset-y-0 rounded-[3px] transition-all duration-300"
                  style={{
                    left: p.val < 0 ? `${50 - (Math.abs(p.val) / maxAbs) * 48}%` : '50%',
                    width: `${Math.max(0.8, (Math.abs(p.val) / maxAbs) * 48)}%`,
                    background: p.val >= 0 ? '#0ca30c' : '#d03b3b',
                    opacity: 0.85,
                  }}
                />
              </div>
              <span
                className={`w-14 shrink-0 text-right ${
                  p.val >= 0 ? 'text-good' : 'text-critical'
                }`}
              >
                {p.val >= 0 ? '+' : ''}
                {fmt.num(p.val)}
              </span>
            </div>
          ))}
        </div>
        <div className="mt-2 flex justify-between border-t border-ring pt-2">
          <span className="text-ink-3">r this step</span>
          <span className={step.reward_total >= 0 ? 'text-good' : 'text-critical'}>
            {step.reward_total >= 0 ? '+' : ''}
            {fmt.num(step.reward_total)}
          </span>
        </div>
      </section>

      {/* running numbers */}
      <section className="grid grid-cols-2 gap-3">
        <Stat label="RETURN Σr" value={fmt.num(cumReturn)} tone={cumReturn >= 0 ? 'good' : 'bad'} />
        <Stat label="V(s) CRITIC" value={fmt.num(step.value)} />
        <Stat
          label="φ(u) DIST LEFT"
          value={fmt.num(episode.graph.nodes[step.u]?.phi ?? 0, 0)}
        />
        <Stat
          label="OPTIMAL COST"
          value={fmt.num(episode.graph.optimal_cost, 0)}
        />
      </section>
    </div>
  )
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string
  value: string
  tone?: 'good' | 'bad'
}) {
  return (
    <div className="rounded-md border border-ring bg-surface-2 px-3 py-2">
      <div className="text-[10px] tracking-[0.15em] text-ink-3">{label}</div>
      <div
        className={`mt-0.5 text-lg ${
          tone === 'good' ? 'text-good' : tone === 'bad' ? 'text-critical' : 'text-ink'
        }`}
      >
        {value}
      </div>
    </div>
  )
}
