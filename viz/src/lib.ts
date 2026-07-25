// Sequential blue ramp (dataviz reference palette) — phi encodes distance to
// target: bright = close, dark = far. Interpolated across the ramp steps.
const RAMP = [
  '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
  '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b',
]

function hex2rgb(h: string): [number, number, number] {
  return [
    parseInt(h.slice(1, 3), 16),
    parseInt(h.slice(3, 5), 16),
    parseInt(h.slice(5, 7), 16),
  ]
}

export function phiColor(phi: number, phiMax: number): string {
  const t = phiMax <= 0 ? 0 : Math.min(1, Math.max(0, phi / phiMax))
  const f = t * (RAMP.length - 1)
  const i = Math.min(RAMP.length - 2, Math.floor(f))
  const frac = f - i
  const a = hex2rgb(RAMP[i])
  const b = hex2rgb(RAMP[i + 1])
  const c = a.map((av, k) => Math.round(av + (b[k] - av) * frac))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}

export const fmt = {
  steps(n: number): string {
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
    if (n >= 1e3) return `${(n / 1e3).toFixed(0)}k`
    return String(n)
  },
  pct(x: number): string {
    return `${(100 * x).toFixed(1)}%`
  },
  num(x: number, d = 2): string {
    return x.toFixed(d)
  },
}

export const SERIES = {
  // first three categorical slots (all-pairs CVD-validated, dark-stepped)
  s1: '#3987e5',
  s2: '#d95926',
  s3: '#199e70',
}
