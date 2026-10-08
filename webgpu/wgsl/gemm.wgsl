// Y[m][n] = epilogue( sum_k X[m][k] * W[n][k] + bias[n] ).  W — строки матрицы [N][K] в одном из форматов vox.
// {{WTYPE}}: 0 f32, 1 f16, 2 bf16, 3 q8, 4 q4 (блоки по 32, байт j = w[j] | w[j+16] << 4, значение (u - 8) * масштаб).
// epi: 0 нет, 1 gelu (tanh), 2 gelu (erf), 3 Y += G[n] * v, 4 silu, 5 Y += v.
const WTYPE: u32 = {{WTYPE}}u;

struct P {
  M: u32, N: u32, K: u32, epi: u32,
  x_off: u32, y_off: u32, g_off: u32, b_off: u32,
  ldx: u32, ldy: u32, has_bias: u32, pad0: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> S: array<f32>;
@group(0) @binding(4) var<storage, read> B: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
@group(0) @binding(6) var<storage, read> G: array<f32>;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;

fn wload(row: u32, col: u32) -> f32 {
  let K = p.K;
  if (WTYPE == 0u) {
    return bitcast<f32>(W[row * K + col]);
  } else if (WTYPE == 1u) {
    let i = row * K + col;
    return unpack2x16float(W[i >> 1u])[i & 1u];
  } else if (WTYPE == 2u) {
    let i = row * K + col;
    let u = (W[i >> 1u] >> (16u * (i & 1u))) & 0xffffu;
    return bitcast<f32>(u << 16u);
  } else if (WTYPE == 3u) {
    let i = row * K + col;
    let b = (W[i >> 2u] >> (8u * (i & 3u))) & 0xffu;
    let v = (i32(b << 24u)) >> 24u;
    return f32(v) * S[row * (K / 32u) + col / 32u];
  } else {
    let blk = col / 32u;
    let j = col % 32u;
    let bi = row * (K / 2u) + blk * 16u + (j % 16u);
    let byte = (W[bi >> 2u] >> (8u * (bi & 3u))) & 0xffu;
    let u = (byte >> (4u * (j / 16u))) & 0xfu;
    return (f32(u) - 8.0) * S[row * (K / 32u) + blk];
  }
}

fn erf_approx(z: f32) -> f32 {
  let a = abs(z);
  let t = 1.0 / (1.0 + 0.5 * a);
  let r = t * exp(-a * a - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
          t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  let e = 1.0 - r;
  return select(-e, e, z >= 0.0);
}

fn tanh_safe(x: f32) -> f32 {
  let c = clamp(x, -15.0, 15.0);
  let e = exp(2.0 * c);
  return (e - 1.0) / (e + 1.0);
}

fn finish(v_in: f32, m: u32, n: u32) {
  var v = v_in;
  if (p.has_bias != 0u) { v = v + B[p.b_off + n]; }
  let yi = p.y_off + m * p.ldy + n;
  switch (p.epi) {
    case 1u: { v = 0.5 * v * (1.0 + tanh_safe(0.7978845608028654 * (v + 0.044715 * v * v * v))); Y[yi] = v; }
    case 2u: { v = 0.5 * v * (1.0 + erf_approx(v * 0.7071067811865476)); Y[yi] = v; }
    case 3u: { Y[yi] = Y[yi] + G[p.g_off + n] * v; }
    case 4u: { Y[yi] = v / (1.0 + exp(-v)); }
    case 5u: { Y[yi] = Y[yi] + v; }
    default: { Y[yi] = v; }
  }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.y * 16u + lid.x;
  let m0 = wg.y * BM;
  let n0 = wg.x * BN;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
  for (var k0 = 0u; k0 < p.K; k0 += BK) {
    for (var i = 0u; i < 4u; i++) {
      let e = tid + i * 256u;
      let r = e / BK;
      let c = e % BK;
      let gm = m0 + r;
      let gk = k0 + c;
      var a = 0.0;
      if (gm < p.M && gk < p.K) { a = X[p.x_off + gm * p.ldx + gk]; }
      As[e] = a;
      let gn = n0 + r;
      var b = 0.0;
      if (gn < p.N && gk < p.K) { b = wload(gn, gk); }
      Bs[e] = b;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < BK; kk++) {
      var a: array<f32, 4>;
      var b: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) { a[i] = As[(lid.y * 4u + i) * BK + kk]; b[i] = Bs[(lid.x * 4u + i) * BK + kk]; }
      for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { acc[i * 4u + j] += a[i] * b[j]; } }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    for (var j = 0u; j < 4u; j++) {
      let m = m0 + lid.y * 4u + i;
      let n = n0 + lid.x * 4u + j;
      if (m < p.M && n < p.N) { finish(acc[i * 4u + j], m, n); }
    }
  }
}
