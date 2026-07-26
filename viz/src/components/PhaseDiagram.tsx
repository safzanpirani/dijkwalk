import { fmt } from '../lib'
import type { Q2PhaseCell, Q2PhaseDiagram } from '../types'

const BUDGETS: Q2PhaseCell['budget'][] = ['4n', '2n', '1p25n']
const REGIMES: Q2PhaseCell['regime'][] = ['small', 'cold']

function budgetLabel(budget: Q2PhaseCell['budget']) {
  return budget === '1p25n' ? '1.25n' : budget
}

function rangeLabel(range: Q2PhaseCell['graded_success']) {
  return `${fmt.pct(range.min)}–${fmt.pct(range.max)}`
}

function PhaseCell({ cell }: { cell: Q2PhaseCell }) {
  const strength = Math.min(cell.graded_minus_sparse / 0.15, 1)
  const background = `rgba(57, 135, 229, ${0.08 + strength * 0.30})`
  return (
    <div
      className={`min-h-44 rounded-lg border p-5 ${
        cell.passes_gate ? 'border-good' : 'border-ring'
      }`}
      style={{ background }}
    >
      <div className="mb-4 flex items-start justify-between gap-3">
        <div className="font-display text-4xl italic text-ink">
          +{(cell.graded_minus_sparse * 100).toFixed(1)}
          <span className="ml-1 font-mono text-sm not-italic text-ink-2">pp</span>
        </div>
        {cell.passes_gate ? (
          <span className="rounded border border-good/50 px-2 py-1 text-[10px] tracking-wider text-good">
            GATE PASS
          </span>
        ) : null}
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="mb-1 text-[10px] tracking-widest text-s1">GRADED</div>
          <div className="text-lg text-ink">{fmt.pct(cell.graded_success.mean)}</div>
          <div className="text-[10px] text-ink-3">
            seeds {rangeLabel(cell.graded_success)}
          </div>
        </div>
        <div>
          <div className="mb-1 text-[10px] tracking-widest text-ink-3">SPARSE</div>
          <div className="text-lg text-ink">{fmt.pct(cell.sparse_success.mean)}</div>
          <div className="text-[10px] text-ink-3">
            seeds {rangeLabel(cell.sparse_success)}
          </div>
        </div>
      </div>
      <div className="mt-4 border-t border-ring pt-3 text-[10px] text-ink-3">
        {cell.non_overlapping_seed_ranges
          ? 'seed ranges do not overlap'
          : 'seed ranges overlap'}
      </div>
    </div>
  )
}

export default function PhaseDiagram({ data }: { data: Q2PhaseDiagram }) {
  const cells = new Map(
    data.phase_cells.map((cell) => [`${cell.regime}/${cell.budget}`, cell]),
  )
  return (
    <section className="fade-up">
      <div className="mb-6 flex flex-wrap items-end justify-between gap-5">
        <div>
          <h2 className="font-display text-4xl italic text-ink">
            When does shaping matter?
          </h2>
          <p className="mt-2 max-w-3xl leading-relaxed text-ink-2">
            Sampled success gap (graded − sparse) after 5M LSTM/Muon steps.
            Every cell uses three training seeds and the same 1,000 held-out
            graphs per size regime.
          </p>
        </div>
        <div
          className={`rounded-md border px-4 py-3 text-xs ${
            data.gate.passed
              ? 'border-good/60 bg-good/10 text-good'
              : 'border-critical/60 bg-critical/10 text-critical'
          }`}
        >
          {data.gate.passed
            ? 'GATE PASS'
            : 'HONEST NULL · no cell reaches ≥15 pp'}
        </div>
      </div>

      <div className="overflow-x-auto pb-2">
        <div className="grid min-w-[920px] grid-cols-[145px_repeat(3,minmax(0,1fr))] gap-3">
          <div />
          {BUDGETS.map((budget) => (
            <div
              key={budget}
              className="pb-1 text-center text-xs tracking-widest text-ink-3"
            >
              {budgetLabel(budget)} BUDGET
            </div>
          ))}
          {REGIMES.flatMap((regime) => {
            const label =
              regime === 'small'
                ? ['SMALL FIXED', 'n=8–16']
                : ['COLD START', 'n=32–64']
            return [
              <div
                key={`${regime}-label`}
                className="flex flex-col justify-center pr-4 text-right"
              >
                <span className="text-sm font-semibold text-ink">{label[0]}</span>
                <span className="mt-1 text-xs text-ink-3">{label[1]}</span>
              </div>,
              ...BUDGETS.map((budget) => {
                const cell = cells.get(`${regime}/${budget}`)
                if (!cell) throw new Error(`missing Q2 cell ${regime}/${budget}`)
                return <PhaseCell key={`${regime}/${budget}`} cell={cell} />
              }),
            ]
          })}
        </div>
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-3">
        <div className="rounded-lg border border-ring bg-surface p-4">
          <div className="text-[10px] tracking-widest text-ink-3">MAIN GRID</div>
          <div className="mt-2 text-ink">goal bonus = {data.config.goal_bonus}</div>
          <div className="mt-1 text-xs text-ink-3">no curriculum</div>
        </div>
        <div className="rounded-lg border border-ring bg-surface p-4">
          <div className="text-[10px] tracking-widest text-ink-3">EVALUATION</div>
          <div className="mt-2 text-ink">
            {data.config.episodes_per_run.toLocaleString()} graphs / run
          </div>
          <div className="mt-1 text-xs text-ink-3">
            {data.config.decoding} decoding · invalid rate 0
          </div>
        </div>
        <div className="rounded-lg border border-ring bg-surface p-4">
          <div className="text-[10px] tracking-widest text-ink-3">KILL GATE</div>
          <div className="mt-2 text-ink">
            ≥{(data.gate.required_gap * 100).toFixed(0)} pp gap
          </div>
          <div className="mt-1 text-xs text-ink-3">
            plus non-overlapping seed ranges
          </div>
        </div>
      </div>
    </section>
  )
}
