// Потоковые (причинные) свёртки декодера кодека. Вход = история H (ctx строк) + новый кусок X (T строк), cin каналов.
//   im2col: Y[t][k * cin + i] = ext[t + k * dil][i]       → дальше обычный gemm с весами [cout][K * cin]
//   hist  : Y[j][i] = ext[T + j][i], j < ctx               (новая история, пишется в другой буфер)
//   dw    : depthwise K = p.K: Y[t][c] = B[c] + sum_k W[c * K + k] * ext[t + k][c]   (dil = 1, ctx = K - 1)
struct P { T: u32, cin: u32, K: u32, dil: u32, ctx: u32, x_off: u32, y_off: u32, has_bias: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> H: array<f32>;
@group(0) @binding(3) var<storage, read> W: array<f32>;
@group(0) @binding(4) var<storage, read> B: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;

fn gid1(g: vec3<u32>) -> u32 { return g.x + g.y * 65535u * 64u; }
fn ext_at(s: u32, i: u32) -> f32 {
  if (s < p.ctx) { return H[s * p.cin + i]; }
  return X[p.x_off + (s - p.ctx) * p.cin + i];
}

@compute @workgroup_size(64)
fn im2col(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = gid1(g);
  let kc = p.K * p.cin;
  if (idx >= p.T * kc) { return; }
  let t = idx / kc;
  let r = idx % kc;
  let k = r / p.cin;
  let i = r % p.cin;
  Y[p.y_off + idx] = ext_at(t + k * p.dil, i);
}

@compute @workgroup_size(64)
fn hist(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = gid1(g);
  if (idx >= p.ctx * p.cin) { return; }
  let j = idx / p.cin;
  let i = idx % p.cin;
  Y[idx] = ext_at(p.T + j, i);
}

@compute @workgroup_size(64)
fn dw(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = gid1(g);
  if (idx >= p.T * p.cin) { return; }
  let t = idx / p.cin;
  let c = idx % p.cin;
  var acc = 0.0;
  if (p.has_bias != 0u) { acc = B[c]; }
  for (var k = 0u; k < p.K; k++) { acc += W[c * p.K + k] * ext_at(t + k, c); }
  Y[p.y_off + idx] = acc;
}
