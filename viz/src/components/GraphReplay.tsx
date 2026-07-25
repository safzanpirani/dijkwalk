import { useEffect, useMemo, useRef, useState } from 'react'
import type { Episode, Step } from '../types'
import { fmt, phiColor } from '../lib'

const PAD = 70
const SIZE = 1000

function px(x: number) {
  return PAD + x * (SIZE - 2 * PAD)
}

interface Props {
  episode: Episode
  stepIdx: number // index into steps; steps.length === "arrived" frame
  onHoverNode?: (id: number | null) => void
}

/** The walked node sequence up to (and including) frame `stepIdx`. */
export function walkedPath(episode: Episode, stepIdx: number): number[] {
  const path = [episode.graph.src]
  for (let i = 0; i < stepIdx && i < episode.steps.length; i++) {
    const s = episode.steps[i]
    if (s.v >= 0) path.push(s.v)
  }
  return path
}

export default function GraphReplay({ episode, stepIdx, onHoverNode }: Props) {
  const g = episode.graph
  const phiMax = useMemo(() => Math.max(...g.nodes.map((n) => n.phi)), [g])
  const arrived = stepIdx >= episode.steps.length
  const step: Step | null = arrived ? null : episode.steps[stepIdx]
  const path = useMemo(() => walkedPath(episode, stepIdx), [episode, stepIdx])
  const cur = path[path.length - 1]

  const walkedEdges = useMemo(() => {
    const set = new Set<string>()
    for (let i = 1; i < path.length; i++) {
      const [a, b] = [path[i - 1], path[i]]
      set.add(`${Math.min(a, b)}-${Math.max(a, b)}`)
    }
    return set
  }, [path])

  const optimalEdges = useMemo(() => {
    const set = new Set<string>()
    const p = g.optimal_path
    for (let i = 1; i < p.length; i++) {
      const [a, b] = [p[i - 1], p[i]]
      set.add(`${Math.min(a, b)}-${Math.max(a, b)}`)
    }
    return set
  }, [g])

  // candidate probability per neighbor node at the current frame
  const probByNode = useMemo(() => {
    const m = new Map<number, number>()
    if (step) {
      step.slots.forEach((v, k) => {
        if (v >= 0) m.set(v, step.probs[k])
      })
    }
    return m
  }, [step])

  const chosen = step && step.v >= 0 ? step.v : null

  return (
    <svg
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      className="h-full w-full select-none"
      role="img"
      aria-label="graph walk replay"
    >
      {/* base edges */}
      {g.edges.map((e) => {
        const key = `${Math.min(e.u, e.v)}-${Math.max(e.u, e.v)}`
        const a = g.nodes[e.u]
        const b = g.nodes[e.v]
        const walked = walkedEdges.has(key)
        const optimal = optimalEdges.has(key)
        return (
          <g key={key}>
            {optimal && (
              <line
                x1={px(a.x)} y1={px(a.y)} x2={px(b.x)} y2={px(b.y)}
                stroke="#0ca30c" strokeOpacity={0.16} strokeWidth={10}
                strokeLinecap="round"
              />
            )}
            <line
              x1={px(a.x)} y1={px(a.y)} x2={px(b.x)} y2={px(b.y)}
              stroke={walked ? '#d95926' : '#383835'}
              strokeOpacity={walked ? 0.9 : 0.8}
              strokeWidth={walked ? 3 : 0.75 + (e.w / 9) * 2.25}
            />
            {/* edge weight tag */}
            <text
              x={(px(a.x) + px(b.x)) / 2}
              y={(px(a.y) + px(b.y)) / 2 - 4}
              textAnchor="middle"
              fontSize={13}
              fill={walked ? '#d95926' : '#898781'}
              fillOpacity={walked ? 1 : 0.7}
              fontFamily="IBM Plex Mono, monospace"
            >
              {e.w}
            </text>
          </g>
        )
      })}

      {/* policy attention: candidate edges glow by prob */}
      {step &&
        step.slots.map((v, k) => {
          if (v < 0) return null
          const a = g.nodes[step.u]
          const b = g.nodes[v]
          const p = step.probs[k]
          const isChosen = v === chosen
          return (
            <line
              key={`cand-${k}`}
              x1={px(a.x)} y1={px(a.y)} x2={px(b.x)} y2={px(b.y)}
              stroke={isChosen ? '#d95926' : '#3987e5'}
              strokeOpacity={0.25 + p * 0.75}
              strokeWidth={2 + p * 12}
              strokeLinecap="round"
              strokeDasharray={isChosen ? '8 6' : undefined}
              className={isChosen ? 'edge-flow' : undefined}
            />
          )
        })}

      {/* nodes */}
      {g.nodes.map((n) => {
        const isTgt = n.id === g.tgt
        const isSrc = n.id === g.src
        const isCur = n.id === cur
        const p = probByNode.get(n.id)
        const r = isCur ? 15 : isTgt ? 14 : 10
        return (
          <g
            key={n.id}
            onMouseEnter={() => onHoverNode?.(n.id)}
            onMouseLeave={() => onHoverNode?.(null)}
          >
            {isCur && (
              <circle
                cx={px(n.x)} cy={px(n.y)}
                fill="none" stroke="#d95926" strokeWidth={2}
                className="agent-pulse"
              />
            )}
            <circle
              cx={px(n.x)} cy={px(n.y)} r={r}
              fill={isCur ? '#d95926' : phiColor(n.phi, phiMax)}
              stroke={
                isTgt ? '#0ca30c' : isCur ? '#ffffff' : 'rgba(255,255,255,0.25)'
              }
              strokeWidth={isTgt ? 3.5 : isCur ? 2 : 1}
            />
            {/* node id */}
            <text
              x={px(n.x)} y={px(n.y) + 4}
              textAnchor="middle" fontSize={11} fontWeight={600}
              fill={n.phi / phiMax < 0.45 || isCur ? '#0d0d0d' : '#ffffff'}
              fontFamily="IBM Plex Mono, monospace"
            >
              {n.id}
            </text>
            {/* phi under the node */}
            <text
              x={px(n.x)} y={px(n.y) + r + 15}
              textAnchor="middle" fontSize={11}
              fill="#898781"
              fontFamily="IBM Plex Mono, monospace"
            >
              φ{n.phi}
            </text>
            {(isTgt || isSrc) && (
              <text
                x={px(n.x)} y={px(n.y) - r - 8}
                textAnchor="middle" fontSize={12} fontWeight={600}
                fill={isTgt ? '#0ca30c' : '#c3c2b7'}
                fontFamily="IBM Plex Mono, monospace"
              >
                {isTgt ? 'TARGET' : 'START'}
              </text>
            )}
            {/* candidate prob tag */}
            {p !== undefined && !isCur && (
              <text
                x={px(n.x)} y={px(n.y) - r - (isTgt ? 22 : 8)}
                textAnchor="middle" fontSize={12} fontWeight={600}
                fill={n.id === chosen ? '#d95926' : '#3987e5'}
                fontFamily="IBM Plex Mono, monospace"
              >
                {fmt.pct(p)}
              </text>
            )}
          </g>
        )
      })}

      {arrived && (
        <text
          x={SIZE / 2} y={44} textAnchor="middle" fontSize={26}
          fill={episode.success ? '#0ca30c' : '#d03b3b'}
          fontFamily="Instrument Serif, serif" fontStyle="italic"
        >
          {episode.success
            ? `arrived — cost ${episode.path_cost} vs optimal ${episode.graph.optimal_cost}`
            : 'truncated — never arrived'}
        </text>
      )}
    </svg>
  )
}

/** Playback hook: owns frame index + play state for an episode. */
export function usePlayback(numFrames: number) {
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(1)
  const raf = useRef<number>(0)
  const last = useRef(0)

  useEffect(() => {
    setFrame(0)
  }, [numFrames])

  useEffect(() => {
    if (!playing) return
    const interval = 900 / speed
    const tick = (t: number) => {
      if (t - last.current >= interval) {
        last.current = t
        setFrame((f) => (f >= numFrames ? 0 : f + 1))
      }
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [playing, speed, numFrames])

  return { frame, setFrame, playing, setPlaying, speed, setSpeed }
}
