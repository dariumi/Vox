// RMSNorm по строкам: y[c] = w[c] * x[c] / sqrt(mean(x²) + eps).
//   main: X → Y;  ip: на месте (Y — и вход, и выход; X не используется).
// rows строк длины C; x_off/y_off — смещения в float.
struct P { rows: u32, C: u32, x_off: u32, y_off: u32, eps: f32, pad0: f32, pad1: f32, pad2: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;

fn row_id(wg: vec3<u32>) -> u32 { return wg.x + wg.y * 32768u; }

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = row_id(wg);
  if (r >= p.rows) { return; }
  let t = lid.x;
  let base = p.x_off + r * p.C;
  var s = 0.0;
  for (var c = t; c < p.C; c += 256u) { let v = X[base + c]; s += v * v; }
  red[t] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) { if (t < st) { red[t] += red[t + st]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(red[0] / f32(p.C) + p.eps);
  let ob = p.y_off + r * p.C;
  for (var c = t; c < p.C; c += 256u) { Y[ob + c] = Wt[c] * (X[base + c] * inv); }
}

@compute @workgroup_size(256)
fn ip(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = row_id(wg);
  if (r >= p.rows) { return; }
  let t = lid.x;
  let base = p.y_off + r * p.C;
  var s = 0.0;
  for (var c = t; c < p.C; c += 256u) { let v = Y[base + c]; s += v * v; }
  red[t] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) { if (t < st) { red[t] += red[t + st]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(red[0] / f32(p.C) + p.eps);
  workgroupBarrier();
  for (var c = t; c < p.C; c += 256u) { Y[base + c] = Wt[c] * (Y[base + c] * inv); }
}
