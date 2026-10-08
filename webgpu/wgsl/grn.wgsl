// GRN (ConvNeXt-V2): Gx[c] = ||x[:, c]||_2 по времени; Nx[c] = Gx[c] / (mean(Gx) + 1e-6); y = gamma * (x * Nx) + beta + x.
// Точки входа: sumsq (Gx² по столбцам), stats (Gx -> Nx, одна группа), apply.
struct P { T: u32, C: u32, x_off: u32, pad0: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Gx: array<f32>;
@group(0) @binding(3) var<storage, read> Gm: array<f32>;
@group(0) @binding(4) var<storage, read> Bt: array<f32>;
var<workgroup> part: array<f32, 256>;

// группа = 32 столбца × 8 потоков по времени
@compute @workgroup_size(256)
fn sumsq(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let cl = lid.x % 32u;
  let tl = lid.x / 32u;
  let c = wg.x * 32u + cl;
  var s = 0.0;
  if (c < p.C) {
    for (var t = tl; t < p.T; t += 8u) { let v = X[p.x_off + t * p.C + c]; s += v * v; }
  }
  part[lid.x] = s;
  workgroupBarrier();
  if (tl == 0u && c < p.C) {
    var a = 0.0;
    for (var i = 0u; i < 8u; i++) { a += part[i * 32u + cl]; }
    Gx[c] = a;
  }
}

@compute @workgroup_size(256)
fn stats(@builtin(local_invocation_id) lid: vec3<u32>) {
  var s = 0.0;
  for (var c = lid.x; c < p.C; c += 256u) { let g = sqrt(Gx[c]); Gx[c] = g; s += g; }
  part[lid.x] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) { if (lid.x < st) { part[lid.x] += part[lid.x + st]; } workgroupBarrier(); }
  let mean = part[0] / f32(p.C);
  workgroupBarrier();
  for (var c = lid.x; c < p.C; c += 256u) { Gx[c] = Gx[c] / (mean + 1e-6); }
}

@compute @workgroup_size(64)
fn apply(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * 65535u * 64u;
  if (i >= p.T * p.C) { return; }
  let c = i % p.C;
  let v = X[p.x_off + i];
  X[p.x_off + i] = Gm[c] * (v * Gx[c]) + Bt[c] + v;
}
