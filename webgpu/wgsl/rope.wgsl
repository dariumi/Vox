// RoPE в виде пар (2j, 2j+1) по таблице cs[pos][j] = (cos, sin); на месте для q и k. Строка r = последовательность * N + позиция.
struct P { rows: u32, N: u32, heads: u32, HD: u32, q_off: u32, k_off: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> Q: array<f32>;
@group(0) @binding(2) var<storage, read_write> K: array<f32>;
@group(0) @binding(3) var<storage, read> CS: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let half = p.HD / 2u;
  let total = p.rows * p.heads * half;
  let idx = gid.x + gid.y * 65535u * 64u;
  if (idx >= total) { return; }
  let j = idx % half;
  let h = (idx / half) % p.heads;
  let r = idx / (half * p.heads);
  let pos = r % p.N;
  let cs = CS[pos * half + j];
  let inner = p.heads * p.HD;
  let qi = p.q_off + r * inner + h * p.HD + 2u * j;
  let ki = p.k_off + r * inner + h * p.HD + 2u * j;
  let q0 = Q[qi]; let q1 = Q[qi + 1u];
  Q[qi] = q0 * cs.x - q1 * cs.y;
  Q[qi + 1u] = q1 * cs.x + q0 * cs.y;
  let k0 = K[ki]; let k1 = K[ki + 1u];
  K[ki] = k0 * cs.x - k1 * cs.y;
  K[ki + 1u] = k1 * cs.x + k0 * cs.y;
}
