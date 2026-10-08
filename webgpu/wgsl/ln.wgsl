// Построчный LayerNorm. mode 0: аффинный (W[c] * x̂ + Bv[c]); mode 1: без параметров с модуляцией x̂ * (1 + V[sc_off + c]) + V[sh_off + c].
struct P { rows: u32, C: u32, mode: u32, x_off: u32, y_off: u32, sc_off: u32, sh_off: u32, pad0: u32, eps: f32, pad1: f32, pad2: f32, pad3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
@group(0) @binding(3) var<storage, read> Bv: array<f32>;
@group(0) @binding(4) var<storage, read> V: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wg.x + wg.y * 32768u;
  if (r >= p.rows) { return; }
  let t = lid.x;
  let base = p.x_off + r * p.C;
  var s = 0.0;
  for (var c = t; c < p.C; c += 256u) { s += X[base + c]; }
  red[t] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) { if (t < st) { red[t] += red[t + st]; } workgroupBarrier(); }
  let mean = red[0] / f32(p.C);
  workgroupBarrier();
  var v = 0.0;
  for (var c = t; c < p.C; c += 256u) { let d = X[base + c] - mean; v += d * d; }
  red[t] = v;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) { if (t < st) { red[t] += red[t + st]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(red[0] / f32(p.C) + p.eps);
  let ob = p.y_off + r * p.C;
  for (var c = t; c < p.C; c += 256u) {
    let xh = (X[base + c] - mean) * inv;
    if (p.mode == 0u) { Y[ob + c] = xh * Wt[c] + Bv[c]; }
    else { Y[ob + c] = xh * (1.0 + V[p.sc_off + c]) + V[p.sh_off + c]; }
  }
}
