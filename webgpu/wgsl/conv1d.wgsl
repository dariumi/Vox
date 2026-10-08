// Свёртка по времени, "same", нечётное K, группы (depthwise при cig = 1). X: [T][cin], W: [cout][cin/groups][K], Y: [T][cout].
// epi: 0 нет, 1 mish, 2 Y += mish(v).
struct P { T: u32, cin: u32, cout: u32, K: u32, groups: u32, epi: u32, x_off: u32, y_off: u32, has_bias: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;

fn mish(x: f32) -> f32 {
  var sp = x;
  if (x <= 20.0) { sp = log(1.0 + exp(x)); }
  let c = clamp(sp, -15.0, 15.0);
  let e = exp(2.0 * c);
  return x * ((e - 1.0) / (e + 1.0));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = p.T * p.cout;
  let idx = gid.x + gid.y * 65535u * 64u;
  if (idx >= total) { return; }
  let t = idx / p.cout;
  let o = idx % p.cout;
  let cig = p.cin / p.groups;
  let cog = p.cout / p.groups;
  let g = o / cog;
  let pad = p.K / 2u;
  var acc = 0.0;
  if (p.has_bias != 0u) { acc = B[o]; }
  for (var k = 0u; k < p.K; k++) {
    let tt = i32(t) + i32(k) - i32(pad);
    if (tt < 0 || tt >= i32(p.T)) { continue; }
    let xb = p.x_off + u32(tt) * p.cin + g * cig;
    let wb = (o * cig) * p.K + k;
    for (var i = 0u; i < cig; i++) { acc += W[wb + i * p.K] * X[xb + i]; }
  }
  let yi = p.y_off + t * p.cout + o;
  if (p.epi == 1u) { Y[yi] = mish(acc); }
  else if (p.epi == 2u) { Y[yi] = Y[yi] + mish(acc); }
  else { Y[yi] = acc; }
}
