// RoPE «rotate_half» (как в Qwen/Llama): пары (j, j + HD/2) в каждой голове; на месте.
// CS[r * half + j] = (cos, sin) для позиции строки r (таблица считается на CPU).
// rows строк, heads голов в строке, inner = heads * HD — шаг строки; off — смещение первой строки в float.
struct P { rows: u32, heads: u32, HD: u32, off: u32, pad0: u32, pad1: u32, pad2: u32, pad3: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read> CS: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let half = p.HD / 2u;
  let total = p.rows * p.heads * half;
  let idx = gid.x + gid.y * 65535u * 64u;
  if (idx >= total) { return; }
  let j = idx % half;
  let h = (idx / half) % p.heads;
  let r = idx / (half * p.heads);
  let cs = CS[r * half + j];
  let b = p.off + r * p.heads * p.HD + h * p.HD;
  let a = X[b + j];
  let c = X[b + j + half];
  X[b + j] = a * cs.x - c * cs.y;
  X[b + j + half] = c * cs.x + a * cs.y;
}
