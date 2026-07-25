export interface GraphNode {
  id: number
  x: number
  y: number
  phi: number
}

export interface GraphEdge {
  u: number
  v: number
  w: number
}

export interface Graph {
  n: number
  src: number
  tgt: number
  nodes: GraphNode[]
  edges: GraphEdge[]
  optimal_path: number[]
  optimal_cost: number
}

export interface Step {
  u: number
  action: number
  v: number
  w: number
  dphi: number
  value: number
  slots: number[]
  probs: number[]
  deltas: Record<string, number>
  reward_edge: number
  reward_shaping: number
  reward_step: number
  reward_total: number
}

export interface Episode {
  graph: Graph
  steps: Step[]
  success: boolean
  path_cost: number
}

export interface TraceFile {
  checkpoint: string
  policy: string
  rnn: string | null
  beta: number
  episodes: Episode[]
}

export interface CurvePoint {
  agent_steps: number
  SPS?: number
  epoch?: number
  'environment/success_rate'?: number
  'environment/episode_length'?: number
  'environment/backward_rate'?: number
  'environment/opt_gap'?: number
  'environment/invalid_rate'?: number
  'environment/n_max_cur'?: number
  'losses/policy_loss'?: number
  'losses/value_loss'?: number
  'losses/entropy'?: number
  'losses/explained_variance'?: number
  'losses/approx_kl'?: number
}

export interface Experiment {
  name: string
  spec: { env?: Record<string, unknown>; rnn?: string | null; policy?: string }
  n_params: number
  steps: number
  num_envs: number
  curve: CurvePoint[]
}
