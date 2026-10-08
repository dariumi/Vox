// Сгенерировано tools/gen_wgsl.py из webgpu/wgsl/*.wgsl — не править вручную.
export const KERNELS = {
  attention: `// Внимание без маски: O[q] = softmax(Q[q] K^T * scale) V, поток по ключам (онлайн-softmax). Группа — QB запросов одной головы
// одной последовательности; потоков HD (по одному на измерение выхода). {{HD}} — размер головы.
const HD: u32 = {{HD}}u;
const QB: u32 = 4u;
const KB: u32 = 32u;

struct P { N: u32, heads: u32, seq: u32, pad0: u32, scale: f32, pad1: f32, pad2: f32, pad3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;

var<workgroup> Qs: array<f32, 4 * {{HD}}>;
var<workgroup> Ts: array<f32, 32 * {{HD}}>;
var<workgroup> Sc: array<f32, 128>;
var<workgroup> Ms: array<f32, 4>;
var<workgroup> Ls: array<f32, 4>;
var<workgroup> Al: array<f32, 4>;

@compute @workgroup_size({{HD}})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let h = wg.y % p.heads;
  let sq = wg.y / p.heads;           // номер последовательности
  let q0 = wg.x * QB;                // первый запрос группы (внутри последовательности)
  let inner = p.heads * HD;
  let rowbase = sq * p.N;
  for (var q = 0u; q < QB; q++) {
    var v = 0.0;
    if (q0 + q < p.N) { v = Q[(rowbase + q0 + q) * inner + h * HD + t]; }
    Qs[q * HD + t] = v;
  }
  if (t < QB) { Ms[t] = -3.0e38; Ls[t] = 0.0; }
  var acc: array<f32, 4>;
  for (var q = 0u; q < QB; q++) { acc[q] = 0.0; }
  workgroupBarrier();
  for (var k0 = 0u; k0 < p.N; k0 += KB) {
    // ключи плитки -> Ts
    for (var e = t; e < KB * HD; e += HD) {
      let j = e / HD;
      let d = e % HD;
      var v = 0.0;
      if (k0 + j < p.N) { v = K[(rowbase + k0 + j) * inner + h * HD + d]; }
      Ts[e] = v;
    }
    workgroupBarrier();
    for (var idx = t; idx < QB * KB; idx += HD) {
      let q = idx / KB;
      let j = idx % KB;
      var s = -3.0e38;
      if (k0 + j < p.N) {
        var a = 0.0;
        for (var d = 0u; d < HD; d++) { a += Qs[q * HD + d] * Ts[j * HD + d]; }
        s = a * p.scale;
      }
      Sc[idx] = s;
    }
    workgroupBarrier();
    if (t < QB) {
      var mx = Ms[t];
      let m_old = mx;
      for (var j = 0u; j < KB; j++) { mx = max(mx, Sc[t * KB + j]); }
      let alpha = exp(m_old - mx);
      var l = Ls[t] * alpha;
      for (var j = 0u; j < KB; j++) {
        var pj = 0.0;
        if (k0 + j < p.N) { pj = exp(Sc[t * KB + j] - mx); }
        Sc[t * KB + j] = pj;
        l += pj;
      }
      Ms[t] = mx; Ls[t] = l; Al[t] = alpha;
    }
    workgroupBarrier();
    // значения плитки -> Ts
    for (var e = t; e < KB * HD; e += HD) {
      let j = e / HD;
      let d = e % HD;
      var v = 0.0;
      if (k0 + j < p.N) { v = V[(rowbase + k0 + j) * inner + h * HD + d]; }
      Ts[e] = v;
    }
    workgroupBarrier();
    for (var q = 0u; q < QB; q++) {
      var a = acc[q] * Al[q];
      for (var j = 0u; j < KB; j++) { a += Sc[q * KB + j] * Ts[j * HD + t]; }
      acc[q] = a;
    }
    workgroupBarrier();
  }
  for (var q = 0u; q < QB; q++) {
    if (q0 + q < p.N) { O[(rowbase + q0 + q) * inner + h * HD + t] = acc[q] / Ls[q]; }
  }
}
`,
  conv1d: `// Свёртка по времени, "same", нечётное K, группы (depthwise при cig = 1). X: [T][cin], W: [cout][cin/groups][K], Y: [T][cout].
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
`,
  ew: `// Поэлементные операции (точки входа):
//   concat_in : out[r] = [x[seq(r)] (mel) | cond[r] (mel) | temb[r] (td)]       X0=x(N*mel), X1=cond(R*mel), X2=temb(R*td), Y=out
//   euler     : y += dt * (pc + (pc - pu) * cfg)                                  X0=pred(2*N*mel), Y=y
//   euler1    : y += dt * pc  (без CFG)                                        X0=pred(N*mel)
//   zero_rows : Y[from*C .. to*C) = 0
//   silu_vec  : Y[i] = silu(X0[i])
//   copy      : Y[y_off + i] = X0[x_off + i]
//   gate      : Y[i] = X0[i] * X1[i]
struct P { n: u32, a: u32, b: u32, c: u32, d: u32, x_off: u32, y_off: u32, pad1: u32, f0: f32, f1: f32, f2: f32, f3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X0: array<f32>;
@group(0) @binding(2) var<storage, read> X1: array<f32>;
@group(0) @binding(3) var<storage, read> X2: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y2: array<f32>;

fn gid1(g: vec3<u32>) -> u32 { return g.x + g.y * 65535u * 64u; }

// a = rows (R), b = N, c = mel, d = td; n = R * (2*mel + td)
@compute @workgroup_size(64)
fn concat_in(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  let mel = p.c;
  let td = p.d;
  let wdt = 2u * mel + td;
  let r = i / wdt;
  let c = i % wdt;
  let seq = r / p.b;
  let pos = r % p.b;
  var v = 0.0;
  if (c < mel) { v = X0[pos * mel + c]; }
  else if (c < 2u * mel) { v = X1[r * mel + (c - mel)]; }
  else { v = X2[r * td + (c - 2u * mel)]; }
  Y[i] = v;
}

// n = N*mel, f0 = dt, f1 = cfg
@compute @workgroup_size(64)
fn euler(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  let pc = X0[i];
  let pu = X0[p.n + i];
  let yn = Y[i] + p.f0 * (pc + (pc - pu) * p.f1);
  Y[i] = yn;
}

@compute @workgroup_size(64)
fn euler1(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  let yn = Y[i] + p.f0 * X0[i];
  Y[i] = yn;
}

// n = (to - from) * C элементов, смещение from * C в y_off
@compute @workgroup_size(64)
fn zero_rows(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  Y[p.y_off + i] = 0.0;
}

@compute @workgroup_size(64)
fn silu_vec(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  let v = X0[p.x_off + i];
  Y[p.y_off + i] = v / (1.0 + exp(-v));
}

@compute @workgroup_size(64)
fn copy(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  Y[p.y_off + i] = X0[p.x_off + i];
}
`,
  gemm: `// Y[m][n] = epilogue( sum_k X[m][k] * W[n][k] + bias[n] ).  W — строки матрицы [N][K] в одном из форматов vox.
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
`,
  grn: `// GRN (ConvNeXt-V2): Gx[c] = ||x[:, c]||_2 по времени; Nx[c] = Gx[c] / (mean(Gx) + 1e-6); y = gamma * (x * Nx) + beta + x.
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
`,
  ln: `// Построчный LayerNorm. mode 0: аффинный (W[c] * x̂ + Bv[c]); mode 1: без параметров с модуляцией x̂ * (1 + V[sc_off + c]) + V[sh_off + c].
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
`,
  rope: `// RoPE в виде пар (2j, 2j+1) по таблице cs[pos][j] = (cos, sin); на месте для q и k. Строка r = последовательность * N + позиция.
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
`,
};
