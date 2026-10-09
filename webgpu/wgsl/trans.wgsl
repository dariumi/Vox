// Транспонированная свёртка (stride s, ядро K <= 2 s) в потоковом виде. TMP = gemm(x, W[K*co][cin]) → [T][K*co];
// выход p собирается из кадров t = p / s и t - 1, плюс хвост предыдущего куска TAIL [tr][co], tr = K - s.
// Позиции p < T*s идут в Y (+ bias), остальные (tr штук) — в NT (новый хвост).
struct P { T: u32, K: u32, s: u32, co: u32, tr: u32, y_off: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> TMP: array<f32>;
@group(0) @binding(2) var<storage, read> TAIL: array<f32>;
@group(0) @binding(3) var<storage, read> Bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> NT: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x + g.y * 65535u * 64u;
  let n = p.T * p.s + p.tr;
  if (idx >= n * p.co) { return; }
  let pos = idx / p.co;
  let o = idx % p.co;
  let t = pos / p.s;
  let k = pos - t * p.s;
  var v = 0.0;
  if (pos < p.tr) { v = TAIL[pos * p.co + o]; }
  if (t < p.T && k < p.K) { v = v + TMP[(t * p.K + k) * p.co + o]; }
  if (t >= 1u && k + p.s < p.K) { v = v + TMP[((t - 1u) * p.K + (k + p.s)) * p.co + o]; }
  if (pos < p.T * p.s) { Y[p.y_off + pos * p.co + o] = v + Bias[o]; }
  else { NT[(pos - p.T * p.s) * p.co + o] = v; }
}
