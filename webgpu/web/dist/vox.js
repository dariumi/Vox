/*! vox.js 0.10.0 — универсальный TTS-движок vox в браузере (WebGPU). Лицензия — как у проекта: https://github.com/dariumi/Vox */
(function (global) {
"use strict";
// Сгенерировано tools/gen_wgsl.py из webgpu/wgsl/*.wgsl — не править вручную.
const KERNELS = {
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

// ============================================================================
// vox.js — универсальный TTS-движок vox в браузере (WebGPU). Семейство F5-TTS (DiT + Vocos).
// Один файл, без зависимостей:  <script src="vox.js"></script>  →  window.Vox
// Эта часть: утилиты, формат .tmod, форматы весов, Philox-шум, расписание шагов.
// ============================================================================

const VOX_VERSION = "0.10.0";

class VoxError extends Error {
  constructor(message, code) { super(message); this.name = "VoxError"; this.code = code || "error"; }
}
const fail = (msg, code) => { throw new VoxError(msg, code); };

// ---------- числа ----------
const f32r = Math.fround;
const _cvt = new DataView(new ArrayBuffer(8));
function f16ToF32(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  if (e === 0) return s * m * 5.960464477539063e-8;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * Math.pow(2, e - 15);
}
function bf16ToF32(h) { _cvt.setUint32(0, (h << 16) >>> 0); return _cvt.getFloat32(0); }

function bessel_i0(x) {
  let sum = 1, term = 1;
  const q = x * x / 4;
  for (let k = 1; k < 200; k++) { term *= q / (k * k); sum += term; if (term < sum * 1e-17) break; }
  return sum;
}

// Окно Кайзера + sinc, как resample_sinc в ядре vox (одинаковый результат с C++ в пределах float).
function resampleSinc(x, srFrom, srTo) {
  const n = x.length;
  if (srFrom === srTo || n === 0) return x;
  const ratio = srTo / srFrom;
  const cutoff = Math.min(1, ratio) * 0.97;
  const zc = 32, half = zc / cutoff, beta = 9.0, i0b = bessel_i0(beta);
  const nout = Math.ceil(n * ratio);
  const y = new Float32Array(nout);
  for (let m = 0; m < nout; m++) {
    const t = m / ratio;
    const lo = Math.ceil(t - half), hi = Math.floor(t + half);
    let acc = 0, wsum = 0;
    for (let k = lo; k <= hi; k++) {
      const d = t - k, u = d / half;
      if (u <= -1 || u >= 1) continue;
      const win = bessel_i0(beta * Math.sqrt(1 - u * u)) / i0b;
      const arg = Math.PI * d * cutoff;
      const s = Math.abs(arg) < 1e-12 ? 1 : Math.sin(arg) / arg;
      const h = cutoff * s * win;
      wsum += h;
      if (k >= 0 && k < n) acc += h * x[k];
    }
    y[m] = wsum !== 0 ? acc / wsum : 0;
  }
  return y;
}

// ---------- Philox4x32-10 (тот же генератор, что в ядре: шум не зависит от устройства) ----------
function mulhilo(a, b) {
  const aL = a & 0xffff, aH = a >>> 16, bL = b & 0xffff, bH = b >>> 16;
  const ll = aL * bL, lh = aL * bH, hl = aH * bL, hh = aH * bH;
  const mid = (ll >>> 16) + (lh & 0xffff) + (hl & 0xffff);
  return [(hh + (lh >>> 16) + (hl >>> 16) + (mid >>> 16)) >>> 0, (((mid & 0xffff) << 16) | (ll & 0xffff)) >>> 0];
}
function philoxBlock(c0, c1, c2, c3, k0, k1) {
  for (let i = 0; i < 10; i++) {
    const [h0, l0] = mulhilo(0xD2511F53, c0);
    const [h1, l1] = mulhilo(0xCD9E8D57, c2);
    const n0 = (h1 ^ c1 ^ k0) >>> 0, n2 = (h0 ^ c3 ^ k1) >>> 0;
    c0 = n0; c1 = l1; c2 = n2; c3 = l0;
    if (i < 9) { k0 = (k0 + 0x9E3779B9) >>> 0; k1 = (k1 + 0xBB67AE85) >>> 0; }
  }
  return c0;
}
// Нормальный шум N(0,1) для куска `chunk`: Box–Muller над счётчиковым Philox (как F5Model::noise).
function f5Noise(seed, chunk, n) {
  const k0 = seed >>> 0, k1 = Math.floor(seed / 4294967296) >>> 0;
  const stream = (0xF5000000 + chunk) >>> 0;
  const out = new Float32Array(n);
  const inv = 1 / 16777216;
  for (let p = 0; p < ((n + 1) >> 1); p++) {
    const u1 = (philoxBlock(p >>> 0, Math.floor(p / 4294967296) >>> 0, stream, 0, k0, k1) >>> 8) * inv;
    const u2 = (philoxBlock(p >>> 0, Math.floor(p / 4294967296) >>> 0, stream, 1, k0, k1) >>> 8) * inv;
    const r = Math.sqrt(-2 * Math.log(1 - u1)), a = 2 * Math.PI * u2;
    out[2 * p] = r * Math.cos(a);
    if (2 * p + 1 < n) out[2 * p + 1] = r * Math.sin(a);
  }
  return out;
}

// Сетка времени CFM: linspace или EPSS, затем sway.
const EPSS = {
  5: [0, 2, 4, 8, 16, 32], 6: [0, 2, 4, 6, 8, 16, 32], 7: [0, 2, 4, 6, 8, 16, 24, 32],
  10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32], 12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
  16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
};
function f5Timesteps(nfe, sway, epss) {
  const n = Math.max(1, nfe | 0);
  let t;
  if (epss && EPSS[n]) t = EPSS[n].map((v) => v / 32);
  else { t = []; for (let i = 0; i <= n; i++) t.push(i / n); }
  return t.map((v) => f32r(v + sway * (Math.cos(Math.PI / 2 * v) - 1 + v)));
}

// ---------- .tmod ----------
class Tmod {
  constructor(bytes) {
    if (bytes instanceof ArrayBuffer) bytes = new Uint8Array(bytes);
    if (bytes.length < 64) fail("файл слишком мал для .tmod", "format");
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
    if (magic !== "VOXM") fail("это не файл .tmod (нет сигнатуры VOXM)", "format");
    if (dv.getUint32(4, true) !== 1) fail("версия формата .tmod не поддерживается этой сборкой vox.js", "format");
    const u64 = (o) => Number(dv.getBigUint64(o, true));
    const mo = u64(16), ms = u64(24), dof = u64(32), dsz = u64(40);
    if (mo + ms > bytes.length || dof + dsz > bytes.length) fail("файл .tmod обрезан (не докачался?)", "format");
    this.manifest = JSON.parse(new TextDecoder().decode(bytes.subarray(mo, mo + ms)));
    this.bytes = bytes;
    this.dataOffset = dof;
    this.table = this.manifest.blobs || {};
  }
  has(name) { return Object.prototype.hasOwnProperty.call(this.table, name); }
  info(name) {
    const b = this.table[name];
    if (!b) fail("в .tmod нет тензора: " + name, "format");
    return b;
  }
  // сырые байты тензора (вид, без копии)
  raw(name) {
    const b = this.info(name);
    return this.bytes.subarray(this.dataOffset + b.offset, this.dataOffset + b.offset + b.nbytes);
  }
  f32(name) {
    const b = this.info(name);
    if (b.dtype !== "F32") fail("тензор должен быть F32: " + name, "format");
    const r = this.raw(name);
    if ((r.byteOffset & 3) === 0) return new Float32Array(r.buffer, r.byteOffset, r.byteLength >> 2);
    return new Float32Array(r.slice().buffer);
  }
  shape(name) { return this.info(name).shape; }
}

// Матрица весов [rows][cols] в одном из форматов vox (для CPU-стороны: таблица токенов, MLP времени).
function matInfo(tm, name) {
  const b = tm.info(name);
  const rows = b.shape[0];
  let cols = b.shape.length > 1 ? b.shape[1] : 1;
  let wtype;
  switch (b.dtype) {
    case "F32": wtype = 0; break;
    case "F16": wtype = 1; break;
    case "BF16": wtype = 2; break;
    case "I8": wtype = 3; break;
    case "U8": wtype = 4; cols *= 2; break;
    default: fail("неподдерживаемый формат весов у " + name + ": " + b.dtype, "format");
  }
  return { name, rows, cols, wtype, scale: wtype >= 3 ? tm.f32(name + ".scale") : null, raw: null };
}
function matRow(tm, m, r, out) {
  const raw = m.raw || tm.raw(m.name);
  const C = m.cols;
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  switch (m.wtype) {
    case 0: for (let c = 0; c < C; c++) out[c] = dv.getFloat32((r * C + c) * 4, true); break;
    case 1: for (let c = 0; c < C; c++) out[c] = f16ToF32(dv.getUint16((r * C + c) * 2, true)); break;
    case 2: for (let c = 0; c < C; c++) out[c] = bf16ToF32(dv.getUint16((r * C + c) * 2, true)); break;
    case 3: for (let c = 0; c < C; c++) out[c] = dv.getInt8(r * C + c) * m.scale[r * (C >> 5) + (c >> 5)]; break;
    default:
      for (let c = 0; c < C; c++) {
        const blk = c >> 5, j = c & 31;
        const byte = raw[r * (C >> 1) + blk * 16 + (j & 15)];
        const u = (byte >> (j >> 4 ? 4 : 0)) & 15;
        out[c] = (u - 8) * m.scale[r * (C >> 5) + blk];
      }
  }
  return out;
}
function matFloat(tm, m) {
  const out = new Float32Array(m.rows * m.cols), tmp = new Float32Array(m.cols);
  for (let r = 0; r < m.rows; r++) { matRow(tm, m, r, tmp); out.set(tmp, r * m.cols); }
  return out;
}

// ---------- загрузка файла с прогрессом ----------
async function fetchBytes(src, onProgress) {
  if (src instanceof Uint8Array) return src;
  if (src instanceof ArrayBuffer) return new Uint8Array(src);
  if (typeof Blob !== "undefined" && src instanceof Blob) {
    const buf = new Uint8Array(await src.arrayBuffer());
    if (onProgress) onProgress(1, buf.length, buf.length);
    return buf;
  }
  const resp = typeof Response !== "undefined" && src instanceof Response ? src : await fetch(src);
  if (!resp.ok) fail("не удалось загрузить модель: HTTP " + resp.status + " " + resp.statusText, "network");
  const total = Number(resp.headers.get("content-length")) || 0;
  if (!resp.body || !resp.body.getReader) {
    const buf = new Uint8Array(await resp.arrayBuffer());
    if (onProgress) onProgress(1, buf.length, buf.length);
    return buf;
  }
  const reader = resp.body.getReader();
  let got = 0, whole = total ? new Uint8Array(total) : null;
  const parts = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (whole && got + value.length <= whole.length) whole.set(value, got); else { if (whole) { parts.push(whole.subarray(0, got)); whole = null; } parts.push(value); }
    got += value.length;
    if (onProgress) onProgress(total ? got / total : 0, got, total);
  }
  if (whole && got === whole.length) return whole;
  if (whole) parts.push(whole.subarray(0, got));
  const out = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

// ============================================================================
// Текст: токенизатор F5 (латиница посимвольно, китайский через таблицу пиньиня из .tmod), нарезка на куски как в f5_tts.
// ============================================================================
const utf8 = new TextEncoder();
const blen = (s) => utf8.encode(s).length;

class F5Tokenizer {
  constructor(tok) {
    this.pinyin = (tok.type || "pinyin") === "pinyin";
    this.vocab = new Map();
    tok.vocab.forEach((s, i) => { if (!this.vocab.has(s)) this.vocab.set(s, i); });
    this.pmap = new Map();
    if (tok.pinyin) for (const k of Object.keys(tok.pinyin)) this.pmap.set(k.codePointAt(0), tok.pinyin[k]);
  }
  symbols(text) {
    if (!this.pinyin) return Array.from(text);
    const cps = [];
    for (const ch of text) {
      let c = ch.codePointAt(0);
      if (c === 0x3B) c = 0x2C;                        // ; → ,
      else if (c === 0x201C || c === 0x201D) c = 0x22; // “ ” → "
      else if (c === 0x2018 || c === 0x2019) c = 0x27; // ‘ ’ → '
      cps.push(c);
    }
    const isAl = (c) => (c >= 48 && c <= 57) || (c >= 97 && c <= 122) || (c >= 65 && c <= 90);
    const isDg = (c) => c >= 48 && c <= 57;
    const out = [], n = cps.length;
    let i = 0;
    while (i < n) {
      const c = cps[i];
      if (isAl(c)) {   // сегмент [A-Za-z0-9]+(\.\d+)?%? — одно слово для jieba
        let j = i;
        while (j < n && isAl(cps[j])) j++;
        if (j + 1 < n && cps[j] === 46 && isDg(cps[j + 1])) { j++; while (j < n && isDg(cps[j])) j++; }
        if (j < n && cps[j] === 37) j++;
        if (j - i > 1 && out.length) {
          const last = out[out.length - 1];
          if (!(last === " " || last === ":" || last === "'" || last === '"')) out.push(" ");
        }
        for (let k = i; k < j; k++) out.push(String.fromCharCode(cps[k]));
        i = j;
        continue;
      }
      if (c >= 0x3100 && c <= 0x9FFF) {
        out.push(" ");
        const p = this.pmap.get(c);
        out.push(p === undefined ? String.fromCodePoint(c) : p);
      } else out.push(String.fromCodePoint(c));
      i++;
    }
    return out;
  }
  encode(text) { return this.symbols(text).map((s) => (this.vocab.has(s) ? this.vocab.get(s) : 0)); }
}

const FULLWIDTH_STOPS = new Set(["；", "：", "，", "。", "！", "？"]);
function splitSentences(text) {
  const out = [];
  let cur = "";
  const cs = Array.from(text);
  const isWs = (c) => c === " " || c === "\t" || c === "\n" || c === "\r" || c === "\f" || c === "\v";
  for (let i = 0; i < cs.length; i++) {
    const c = cs[i];
    cur += c;
    let boundary = false;
    if (c.length === 1 && c.charCodeAt(0) < 0x80) {
      if (";:,.!?".includes(c)) {
        let j = i + 1;
        while (j < cs.length && isWs(cs[j])) j++;
        if (j > i + 1) { boundary = true; i = j - 1; }
      }
    } else if (FULLWIDTH_STOPS.has(c)) boundary = true;
    if (boundary) { out.push(cur); cur = ""; }
  }
  if (cur) out.push(cur);
  return out;
}
const pyStrip = (s) => s.replace(/^[ \t\n\r\f\v]+|[ \t\n\r\f\v]+$/g, "");
function chunkText(text, maxChars) {
  const chunks = [];
  let cur = "", curLen = 0;
  for (const sent of splitSentences(text)) {
    if (!sent) continue;
    const sl = blen(sent);
    const lastAscii = sent.charCodeAt(sent.length - 1) < 0x80;
    const add = lastAscii ? sent + " " : sent;
    if (curLen + sl <= maxChars) { cur += add; curLen += blen(add); }
    else {
      if (cur) chunks.push(pyStrip(cur));
      cur = add; curLen = blen(add);
    }
  }
  if (cur) chunks.push(pyStrip(cur));
  return chunks;
}

// ============================================================================
// DSP: БПФ, mel-спектрограмма образца, обратное STFT для Vocos (в double, как в ядре)
// ============================================================================
class FFT {
  constructor(n) {
    this.n = n;
    this.rev = new Uint32Array(n);
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      this.rev[i] = j;
    }
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { this.cos[i] = Math.cos(2 * Math.PI * i / n); this.sin[i] = Math.sin(2 * Math.PI * i / n); }
  }
  // на месте; sign = -1 прямое, +1 обратное (без нормировки)
  run(re, im, sign) {
    const n = this.n, rev = this.rev;
    for (let i = 1; i < n; i++) { const j = rev[i]; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1, step = n / len;
      for (let i = 0; i < n; i += len) {
        for (let k = 0; k < half; k++) {
          const wr = this.cos[k * step], wi = sign * this.sin[k * step];
          const a = i + k, b = a + half;
          const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
  }
}
const hannPeriodic = (n) => { const w = new Float64Array(n); for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n); return w; };

// log-mel как в F5/Vocos: reflect-дополнение, периодическое окно Ханна, |STFT|, банк фильтров, log(max(x, 1e-5)).
function melLogSpectrogram(cfg, fb, pcm) {
  const n = pcm.length, nfft = cfg.n_fft, hop = cfg.hop, nb = (nfft >> 1) + 1, pad = nfft >> 1, nm = cfg.n_mels;
  if (n <= pad) fail("образец слишком короткий для mel-спектрограммы", "arg");
  const frames = 1 + Math.floor(n / hop);
  const xp = new Float32Array(n + 2 * pad);
  for (let i = -pad; i < n + pad; i++) {
    let j = i < 0 ? -i : i;
    if (j >= n) j = 2 * (n - 1) - j;
    xp[i + pad] = pcm[j];
  }
  const win = new Float64Array(nfft);
  const h = hannPeriodic(cfg.win), off = (nfft - cfg.win) >> 1;
  for (let i = 0; i < cfg.win; i++) win[off + i] = h[i];
  const fft = new FFT(nfft);
  const re = new Float64Array(nfft), im = new Float64Array(nfft), mag = new Float64Array(nb);
  const out = new Float32Array(frames * nm);
  for (let t = 0; t < frames; t++) {
    const base = t * hop;
    for (let i = 0; i < nfft; i++) { re[i] = xp[base + i] * win[i]; im[i] = 0; }
    fft.run(re, im, -1);
    for (let k = 0; k < nb; k++) mag[k] = f32r(Math.hypot(re[k], im[k]));
    for (let m = 0; m < nm; m++) {
      let acc = 0;
      const o = m * nb;
      for (let k = 0; k < nb; k++) acc += fb[o + k] * mag[k];
      out[t * nm + m] = Math.log(Math.max(acc, 1e-5));
    }
  }
  return { mel: out, frames };
}

// Выход головы Vocos: [T][2*nb] = (log|X|, фаза) → звук ("same" padding, как ISTFT в Vocos).
function istftSame(head, T, nfft, hop) {
  const nb = (nfft >> 1) + 1, pad = (nfft - hop) >> 1, total = (T - 1) * hop + nfft;
  const win = hannPeriodic(nfft), fft = new FFT(nfft);
  const y = new Float64Array(total), env = new Float64Array(total);
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  for (let t = 0; t < T; t++) {
    const row = t * 2 * nb;
    for (let k = 0; k < nb; k++) {
      const m = Math.min(Math.exp(head[row + k]), 1e2), p = head[row + nb + k];
      let vr = m * Math.cos(p), vi = m * Math.sin(p);
      if (k === 0 || k === nb - 1) vi = 0;
      re[k] = vr; im[k] = vi;
      if (k > 0 && k < nb - 1) { re[nfft - k] = vr; im[nfft - k] = -vi; }
    }
    fft.run(re, im, +1);
    const o = t * hop;
    for (let i = 0; i < nfft; i++) { y[o + i] += f32r(re[i] / nfft) * win[i]; env[o + i] += win[i] * win[i]; }
  }
  const out = new Float32Array(total - 2 * pad);
  for (let i = 0; i < out.length; i++) out[i] = y[i + pad] / env[i + pad];
  return out;
}

// ============================================================================
// Тонкая обёртка над WebGPU: буферы, конвейеры из WGSL (явные layout по тексту шейдера), кэш bind group,
// параметры диспетчеров в одном uniform-буфере с динамическими смещениями.
// ============================================================================
const PARAM_SLOT = 256;                // minUniformBufferOffsetAlignment
const PARAM_BYTES = 8 * 1024 * 1024;   // 32768 диспетчеров на проход

const f = (x) => ({ f: x });           // пометка float в списке параметров

class Gpu {
  constructor(device, adapterInfo) {
    this.device = device;
    this.info = adapterInfo || {};
    this.pipes = new Map();
    this.bgs = new Map();
    this.nextId = 1;
    this.bufId = new WeakMap();
    this.lost = null;
    device.lost.then((l) => { this.lost = l; });
    const U = GPUBufferUsage;
    this.paramBuf = device.createBuffer({ size: PARAM_BYTES, usage: U.UNIFORM | U.COPY_DST, label: "vox.params" });
    this.pdata = new ArrayBuffer(PARAM_BYTES);
    this.pu = new Uint32Array(this.pdata);
    this.pf = new Float32Array(this.pdata);
    this.poff = 0;
    this.dummyRO = this.buffer(256, "dummyRO");
    this.dummyRW = [this.buffer(256, "dummyRW0"), this.buffer(256, "dummyRW1"), this.buffer(256, "dummyRW2")];
    this.enc = null;
    this.pass = null;
    this.dispatches = 0;
    this.bytes = 0;
  }

  static async create(opts = {}) {
    if (typeof navigator === "undefined" || !navigator.gpu) fail("WebGPU недоступен в этом браузере (navigator.gpu не найден). Нужен Chrome/Edge 113+, Firefox 141+ (Windows) или Safari 26.", "no-webgpu");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: opts.powerPreference || "high-performance", forceFallbackAdapter: !!opts.forceFallbackAdapter });
    if (!adapter) fail("WebGPU есть, но подходящий адаптер не найден (драйвер или список блокировок).", "no-adapter");
    const lim = adapter.limits;
    const want = { maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize,
      maxComputeWorkgroupStorageSize: Math.min(lim.maxComputeWorkgroupStorageSize, 32768) };
    let device;
    try { device = await adapter.requestDevice({ requiredLimits: want }); }
    catch (e) { device = await adapter.requestDevice(); }
    let ai = {};
    try { ai = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {}); } catch (e) { /* нет */ }
    const g = new Gpu(device, { vendor: ai.vendor || "", architecture: ai.architecture || "", device: ai.device || "", description: ai.description || "" });
    g.limits = { maxBuffer: device.limits.maxBufferSize, maxBinding: device.limits.maxStorageBufferBindingSize };
    return g;
  }

  idOf(b) { let i = this.bufId.get(b); if (!i) { i = this.nextId++; this.bufId.set(b, i); } return i; }

  buffer(bytes, label, usage) {
    const U = GPUBufferUsage;
    const size = Math.max(16, (bytes + 3) & ~3);
    if (size > this.device.limits.maxBufferSize) fail("буфер " + (label || "") + " (" + (size >> 20) + " МиБ) больше лимита устройства (" + (this.device.limits.maxBufferSize >> 20) + " МиБ)", "limits");
    this.bytes += size;
    return this.device.createBuffer({ size, usage: usage || (U.STORAGE | U.COPY_SRC | U.COPY_DST), label });
  }
  // буфер с содержимым (байты выравниваются до 4)
  upload(u8, label) {
    const b = this.buffer(u8.byteLength, label);
    const n4 = (u8.byteLength + 3) & ~3;
    if (n4 === u8.byteLength) this.device.queue.writeBuffer(b, 0, u8.buffer, u8.byteOffset, u8.byteLength);
    else { const t = new Uint8Array(n4); t.set(u8); this.device.queue.writeBuffer(b, 0, t); }
    return b;
  }
  write(buf, f32, offsetFloats = 0) { this.device.queue.writeBuffer(buf, offsetFloats * 4, f32.buffer, f32.byteOffset, f32.byteLength); }
  free(b) { if (b && b !== this.dummyRO) { try { b.destroy(); } catch (e) { /* ок */ } } }

  // конвейер: kernel — имя из KERNELS, defs — подстановки {{...}}
  pipeline(kernel, entry, defs) {
    const key = kernel + "/" + entry + "/" + JSON.stringify(defs || {});
    let p = this.pipes.get(key);
    if (p) return p;
    let src = KERNELS[kernel];
    if (!src) fail("нет шейдера " + kernel, "internal");
    for (const k of Object.keys(defs || {})) src = src.split("{{" + k + "}}").join(String(defs[k]));
    const entries = [];
    const re = /@group\(0\)\s*@binding\((\d+)\)\s*var<(uniform|storage,\s*read_write|storage,\s*read|storage)>/g;
    let m;
    while ((m = re.exec(src))) {
      const binding = Number(m[1]), kind = m[2].replace(/\s+/g, "");
      const e = { binding, visibility: GPUShaderStage.COMPUTE };
      if (kind === "uniform") e.buffer = { type: "uniform", hasDynamicOffset: true, minBindingSize: 0 };
      else if (kind === "storage,read_write") e.buffer = { type: "storage" };
      else e.buffer = { type: "read-only-storage" };   // "storage,read" и "storage" (по умолчанию read)
      entries.push(e);
    }
    const module = this.device.createShaderModule({ code: src, label: kernel });
    const bgl = this.device.createBindGroupLayout({ entries });
    const pipeline = this.device.createComputePipeline({
      label: key, layout: this.device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module, entryPoint: entry },
    });
    p = { pipeline, bgl, entries, key, id: this.nextId++, module };
    this.pipes.set(key, p);
    return p;
  }
  async checkShader(kernel, defs) {
    let src = KERNELS[kernel];
    for (const k of Object.keys(defs || {})) src = src.split("{{" + k + "}}").join(String(defs[k]));
    const mod = this.device.createShaderModule({ code: src });
    const info = await mod.getCompilationInfo();
    return info.messages.filter((x) => x.type === "error").map((x) => kernel + ":" + x.lineNum + " " + x.message);
  }

  // ---- запись команд ----
  begin() {
    this.enc = this.device.createCommandEncoder();
    this.pass = this.enc.beginComputePass();
    this.poff = 0;
    this.dispatches = 0;
  }
  // bufs — буферы для bindings 1..n в порядке объявления (null → заглушка)
  run(pipe, bufs, params, gx, gy = 1) {
    if (this.poff + PARAM_SLOT > PARAM_BYTES) fail("слишком много диспетчеров в одном проходе", "internal");
    const base = this.poff >> 2;
    for (let i = 0; i < params.length; i++) {
      const v = params[i];
      if (typeof v === "object") this.pf[base + i] = v.f; else this.pu[base + i] = v >>> 0;
    }
    let key = String(pipe.id);
    const entries = [{ binding: 0, resource: { buffer: this.paramBuf, offset: 0, size: 64 } }];
    let rw = 0;
    for (let i = 1; i < pipe.entries.length; i++) {
      const e = pipe.entries[i];
      let b = bufs[i - 1];
      if (!b) b = e.buffer.type === "storage" ? this.dummyRW[rw++] : this.dummyRO;
      key += "," + this.idOf(b);
      entries.push({ binding: e.binding, resource: { buffer: b } });
    }
    let bg = this.bgs.get(key);
    if (!bg) { bg = this.device.createBindGroup({ layout: pipe.bgl, entries }); this.bgs.set(key, bg); }
    this.pass.setPipeline(pipe.pipeline);
    this.pass.setBindGroup(0, bg, [this.poff]);
    this.pass.dispatchWorkgroups(gx, gy, 1);
    this.poff += PARAM_SLOT;
    this.dispatches++;
  }
  // одномерный диспетчер: count потоков по 64 (нумерация gid.x + gid.y * 65535 * 64, как в шейдерах)
  run1d(pipe, bufs, params, count) {
    const groups = Math.max(1, Math.ceil(count / 64));
    this.run(pipe, bufs, params, Math.min(groups, 65535), Math.ceil(groups / 65535));
  }
  submit() {
    this.pass.end();
    this.device.queue.writeBuffer(this.paramBuf, 0, this.pdata, 0, this.poff);
    this.device.queue.submit([this.enc.finish()]);
    this.enc = null; this.pass = null;
  }
  async idle() { await this.device.queue.onSubmittedWorkDone(); }
  async read(buf, floats, offsetFloats = 0) {
    const U = GPUBufferUsage;
    const st = this.device.createBuffer({ size: (floats * 4 + 3) & ~3, usage: U.MAP_READ | U.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, offsetFloats * 4, st, 0, (floats * 4 + 3) & ~3);
    this.device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(st.getMappedRange().slice(0, floats * 4));
    st.unmap(); st.destroy();
    return out;
  }
}

// Именованные рабочие буферы: растут по мере надобности, bind group'ы кэшируются по id буфера.
class Pool {
  constructor(gpu) { this.gpu = gpu; this.m = new Map(); }
  get(name, floats) {
    const need = Math.max(4, floats) * 4;
    const cur = this.m.get(name);
    if (cur && cur.size >= need) return cur.buf;
    if (cur) this.gpu.free(cur.buf);
    const cap = Math.ceil(need * 1.25 / 256) * 256;
    const buf = this.gpu.buffer(Math.min(cap, this.gpu.device.limits.maxStorageBufferBindingSize), "ws." + name);
    if (buf.size < need) fail("рабочий буфер " + name + " не помещается в лимит устройства; сократите длину фразы", "limits");
    this.m.set(name, { buf, size: buf.size });
    return buf;
  }
  clear() { for (const v of this.m.values()) this.gpu.free(v.buf); this.m.clear(); }
}

// ============================================================================
// F5-TTS на WebGPU: DiT (с CFG), текстовый энкодер ConvNeXt-V2, Vocos. Все веса и активации живут в GPU;
// на CPU остаются только токены, mel образца, вектор времени, таблица RoPE и iSTFT.
// ============================================================================
class F5Gpu {
  constructor(gpu, tm) {
    this.gpu = gpu;
    this.tm = tm;
    this.pool = new Pool(gpu);
    const c = tm.manifest.f5;
    if (!c) fail("в этом .tmod нет раздела f5: это не модель F5-TTS", "format");
    if (c.qk_norm != null) fail("F5-TTS: qk_norm не поддерживается", "unsupported");
    if (c.long_skip_connection) fail("F5-TTS: long_skip_connection не поддерживается", "unsupported");
    if (c.text_embedding_average_upsampling) fail("F5-TTS: average upsampling не поддерживается", "unsupported");
    this.cfg = {
      dim: c.dim, depth: c.depth, heads: c.heads, dim_head: c.dim_head || 64, ff_mult: c.ff_mult || 2,
      text_dim: c.text_dim, text_rows: c.text_rows, mel_dim: c.mel_dim || 100, conv_layers: c.conv_layers ?? 4, conv_mult: c.conv_mult || 2,
      mask_padding: c.text_mask_padding !== false,
      mel: Object.assign({ sample_rate: 24000, n_fft: 1024, hop: 256, win: 1024, n_mels: 100 }, c.mel),
      vocos: Object.assign({ n_fft: 1024, hop: 256 }, c.vocos),
    };
    const k = this.cfg;
    k.sample_rate = k.mel.sample_rate;
    if (k.mel.n_mels !== k.mel_dim) fail("F5-TTS: n_mels != mel_dim", "format");
    if (k.dim_head > 128 || (k.dim_head & 1)) fail("F5-TTS: размер головы " + k.dim_head + " не поддерживается WebGPU-ядром", "unsupported");
    this.tok = new F5Tokenizer(c.tokenizer);
    this.melFb = tm.f32("mel.fb").slice();
    this.generation = tm.manifest.generation || {};
    this.timing = { dit_ms: 0, vocos_ms: 0, text_ms: 0 };
    this._load();
  }

  // ---------- веса ----------
  lin(name, bias = true) {
    const m = matInfo(this.tm, name);
    if (m.wtype >= 3 && (m.cols & 31)) fail("квантованная матрица " + name + ": число столбцов не кратно 32", "format");
    const g = this.gpu;
    const o = { rows: m.rows, cols: m.cols, wtype: m.wtype, W: g.upload(this.tm.raw(name), name), S: null, B: null, hasBias: 0 };
    if (m.wtype >= 3) o.S = g.upload(this.tm.raw(name + ".scale"), name + ".scale");
    const bn = name.slice(0, -6) + "bias";
    if (bias && this.tm.has(bn)) { o.B = g.upload(this.tm.raw(bn), bn); o.hasBias = 1; }
    this.wbytes += this.tm.info(name).nbytes;
    return o;
  }
  vec(name) { this.wbytes += this.tm.info(name).nbytes; return this.gpu.upload(this.tm.raw(name), name); }
  conv(name, cin, cout, K, groups) {
    const shp = this.tm.shape(name + ".weight");
    if (shp.reduce((a, b) => a * b, 1) !== cout * (cin / groups) * K) fail("неверная форма свёртки: " + name, "format");
    return { W: this.vec(name + ".weight"), B: this.tm.has(name + ".bias") ? this.vec(name + ".bias") : null, cin, cout, K, groups };
  }
  _load() {
    const k = this.cfg, tm = this.tm;
    this.wbytes = 0;
    const d = "dit.";
    // CPU-копии: MLP времени и таблица токенов
    const t1 = matInfo(tm, d + "time_embed.time_mlp.0.weight"), t2 = matInfo(tm, d + "time_embed.time_mlp.2.weight");
    this.time = { w1: matFloat(tm, t1), b1: tm.f32(d + "time_embed.time_mlp.0.bias").slice(), w2: matFloat(tm, t2), b2: tm.f32(d + "time_embed.time_mlp.2.bias").slice(), in1: t1.cols, D: t1.rows };
    if (t1.cols !== 256 || t2.cols !== t1.rows) fail("F5-TTS: неожиданная форма MLP времени", "format");
    this.tokEmb = matInfo(tm, d + "text_embed.text_embed.weight");
    this.tokEmb.raw = tm.raw(this.tokEmb.name).slice();
    if (this.tokEmb.scale) this.tokEmb.scale = this.tokEmb.scale.slice();
    this.in_proj = this.lin(d + "input_embed.proj.weight");
    this.pos1 = this.conv(d + "input_embed.conv_pos_embed.conv1d.0", k.dim, k.dim, 31, 16);
    this.pos2 = this.conv(d + "input_embed.conv_pos_embed.conv1d.2", k.dim, k.dim, 31, 16);
    this.tblocks = [];
    for (let i = 0; i < k.conv_layers; i++) {
      const p = d + "text_embed.text_blocks." + i + ".";
      this.tblocks.push({
        dw: this.conv(p + "dwconv", k.text_dim, k.text_dim, 7, k.text_dim),
        ln_w: this.vec(p + "norm.weight"), ln_b: this.vec(p + "norm.bias"),
        pw1: this.lin(p + "pwconv1.weight"), pw2: this.lin(p + "pwconv2.weight"),
        grn_g: this.vec(p + "grn.gamma"), grn_b: this.vec(p + "grn.beta"),
      });
    }
    this.blocks = [];
    for (let i = 0; i < k.depth; i++) {
      const p = d + "transformer_blocks." + i + ".";
      this.blocks.push({
        ada: this.lin(p + "attn_norm.linear.weight"), q: this.lin(p + "attn.to_q.weight"), k: this.lin(p + "attn.to_k.weight"),
        v: this.lin(p + "attn.to_v.weight"), o: this.lin(p + "attn.to_out.0.weight"),
        ff1: this.lin(p + "ff.ff.0.0.weight"), ff2: this.lin(p + "ff.ff.2.weight"),
      });
    }
    this.norm_out = this.lin(d + "norm_out.linear.weight");
    this.proj_out = this.lin(d + "proj_out.weight");
    // Vocos
    const v = "vocos.", vc = tm.manifest.f5.vocos;
    this.vk = { dim: vc.dim, inter: vc.intermediate, layers: vc.layers, n_fft: vc.n_fft || 1024, hop: vc.hop || 256 };
    this.v_embed = this.conv(v + "backbone.embed", k.mel_dim, vc.dim, 7, 1);
    this.v_ln = { w: this.vec(v + "backbone.norm.weight"), b: this.vec(v + "backbone.norm.bias") };
    this.v_fln = { w: this.vec(v + "backbone.final_layer_norm.weight"), b: this.vec(v + "backbone.final_layer_norm.bias") };
    this.v_blocks = [];
    for (let i = 0; i < vc.layers; i++) {
      const p = v + "backbone.convnext." + i + ".";
      this.v_blocks.push({
        dw: this.conv(p + "dwconv", vc.dim, vc.dim, 7, vc.dim), ln_w: this.vec(p + "norm.weight"), ln_b: this.vec(p + "norm.bias"),
        pw1: this.lin(p + "pwconv1.weight"), pw2: this.lin(p + "pwconv2.weight"), gamma: this.vec(p + "gamma"),
      });
    }
    this.v_head = this.lin(v + "head.out.weight");
  }
  dispose() { this.pool.clear(); }

  // ---------- примитивы ----------
  gemm(X, L, Y, M, o = {}) {
    const g = this.gpu, N = L.rows, K = L.cols;
    const pipe = g.pipeline("gemm", "main", { WTYPE: L.wtype });
    g.run(pipe, [X, L.W, L.S, L.B, Y, o.G || null],
      [M, N, K, o.epi || 0, o.xOff || 0, o.yOff || 0, o.gOff || 0, 0, o.ldx || K, o.ldy || N, L.hasBias, 0], Math.ceil(N / 64), Math.ceil(M / 64));
  }
  ln(X, Y, rows, C, o) {
    const g = this.gpu, pipe = g.pipeline("ln", "main");
    g.run(pipe, [X, o.W || null, o.Bv || null, o.V || null, Y],
      [rows, C, o.mode || 0, o.xOff || 0, o.yOff || 0, o.scOff || 0, o.shOff || 0, 0, f(o.eps ?? 1e-6), 0, 0, 0], Math.min(rows, 32768), Math.ceil(rows / 32768));
  }
  convk(X, Cv, Y, T, epi, xOff = 0, yOff = 0) {
    const g = this.gpu, pipe = g.pipeline("conv1d", "main");
    g.run1d(pipe, [X, Cv.W, Cv.B, Y], [T, Cv.cin, Cv.cout, Cv.K, Cv.groups, epi, xOff, yOff, Cv.B ? 1 : 0, 0, 0, 0], T * Cv.cout);
  }
  ew(entry, bufs, p) {
    const g = this.gpu, pipe = g.pipeline("ew", entry);
    g.run1d(pipe, bufs, [p.n, p.a || 0, p.b || 0, p.c || 0, p.d || 0, p.xOff || 0, p.yOff || 0, 0, f(p.f0 || 0), f(p.f1 || 0), 0, 0], p.n);
  }
  grn(X, Gx, Gm, Bt, T, C, xOff) {
    const g = this.gpu;
    g.run(g.pipeline("grn", "sumsq"), [X, Gx, Gm, Bt], [T, C, xOff, 0], Math.ceil(C / 32), 1);
    g.run(g.pipeline("grn", "stats"), [X, Gx, Gm, Bt], [T, C, xOff, 0], 1, 1);
    g.run1d(g.pipeline("grn", "apply"), [X, Gx, Gm, Bt], [T, C, xOff, 0], T * C);
  }

  // ---------- текстовый энкодер ----------
  // Строка таблицы + абсолютная позиция + обнуление заполнителя; дальше ConvNeXt-V2 на GPU. Пишет в temb[b*N*TD …].
  textEmbed(ids, N, drop, b, temb, ws) {
    const k = this.cfg, D = k.text_dim, g = this.gpu;
    const emb = new Float32Array(N * D), row = new Float32Array(D);
    for (let i = 0; i < N; i++) {
      const id = drop || i >= ids.length ? 0 : ids[i] + 1;
      if (id >= this.tokEmb.rows) fail("F5-TTS: идентификатор токена вне таблицы эмбеддингов", "format");
      matRow(this.tm, this.tokEmb, id, row);
      emb.set(row, i * D);
    }
    if (k.conv_layers > 0) {
      const half = D >> 1;
      const inv = new Float64Array(half);
      for (let j = 0; j < half; j++) inv[j] = 1 / Math.pow(10000, (2 * j) / D);
      for (let i = 0; i < N; i++)
        for (let j = 0; j < half; j++) { const a = f32r(i * inv[j]); emb[i * D + j] += Math.cos(a); emb[i * D + half + j] += Math.sin(a); }
    }
    const len = Math.min(ids.length, N);
    if (k.mask_padding) emb.fill(0, len * D);
    const off = b * N * D;
    g.write(temb, emb, off);
    if (k.conv_layers <= 0) return;
    const I = D * k.conv_mult;
    for (const L of this.tblocks) {
      this.convk(temb, L.dw, ws.y1, N, 0, off, 0);
      this.ln(ws.y1, ws.y2, N, D, { W: L.ln_w, Bv: L.ln_b, eps: 1e-6 });
      this.gemm(ws.y2, L.pw1, ws.hb, N, { epi: 2 });
      this.grn(ws.hb, ws.gx, L.grn_g, L.grn_b, N, I, 0);
      this.gemm(ws.hb, L.pw2, temb, N, { epi: 5, yOff: off });
      if (k.mask_padding && len < N) this.ew("zero_rows", [null, null, null, temb], { n: (N - len) * D, yOff: off + len * D });
    }
  }
}

// ============================================================================
// F5-TTS на WebGPU, часть 2: шаг DiT, цикл CFM (Эйлер + CFG), Vocos.
// ============================================================================
Object.assign(F5Gpu.prototype, {
  // sin/cos-вектор времени → Linear → SiLU → Linear → SiLU  (D чисел), на CPU: это 1.3 млн умножений на шаг
  timeVector(t) {
    const T = this.time, D = T.D, half = 128;
    const te = new Float32Array(256), th = new Float32Array(D), ts = new Float32Array(D);
    const e = Math.log(10000) / (half - 1);
    for (let i = 0; i < half; i++) { const a = f32r(1000 * t * Math.exp(i * -e)); te[i] = Math.sin(a); te[half + i] = Math.cos(a); }
    for (let o = 0; o < D; o++) {
      let acc = T.b1[o];
      const w = o * 256;
      for (let k = 0; k < 256; k++) acc += T.w1[w + k] * te[k];
      th[o] = acc / (1 + Math.exp(-acc));
    }
    for (let o = 0; o < D; o++) {
      let acc = T.b2[o];
      const w = o * D;
      for (let k = 0; k < D; k++) acc += T.w2[w + k] * th[k];
      ts[o] = acc / (1 + Math.exp(-acc));
    }
    return ts;
  },

  // Один проход DiT над B последовательностями длины N (B = 2: условная и безусловная ветки CFG). Пишет pred [B*N][mel].
  encodeDit(ws, B, N) {
    const k = this.cfg, D = k.dim, M = k.mel_dim, TD = k.text_dim, H = k.heads, HD = k.dim_head, INNER = H * HD, R = B * N;
    this.ew("concat_in", [ws.y, ws.cond, ws.temb, ws.cat], { n: R * (2 * M + TD), a: R, b: N, c: M, d: TD });
    this.gemm(ws.cat, this.in_proj, ws.h, R);
    for (let s = 0; s < B; s++) {            // свёрточное позиционное кодирование: две группированные свёртки + Mish, остаток
      this.convk(ws.h, this.pos1, ws.cv, N, 1, s * N * D, 0);
      this.convk(ws.cv, this.pos2, ws.h, N, 2, 0, s * N * D);
    }
    const g = this.gpu, rope = g.pipeline("rope", "main"), attn = g.pipeline("attention", "main", { HD });
    const scale = f32r(1 / Math.sqrt(HD));
    for (let i = 0; i < this.blocks.length; i++) {
      const L = this.blocks[i], a0 = i * 6 * D;
      this.gemm(ws.ts, L.ada, ws.ada, 1, { yOff: a0 });
      this.ln(ws.h, ws.nrm, R, D, { mode: 1, V: ws.ada, scOff: a0 + D, shOff: a0 });
      this.gemm(ws.nrm, L.q, ws.q, R);
      this.gemm(ws.nrm, L.k, ws.k, R);
      this.gemm(ws.nrm, L.v, ws.v, R);
      g.run1d(rope, [ws.q, ws.k, ws.cs], [R, N, H, HD, 0, 0, 0, 0], R * H * (HD >> 1));
      g.run(attn, [ws.q, ws.k, ws.v, ws.at], [N, H, B, 0, f(scale), 0, 0, 0], Math.ceil(N / 4), H * B);
      this.gemm(ws.at, L.o, ws.h, R, { epi: 3, G: ws.ada, gOff: a0 + 2 * D });
      this.ln(ws.h, ws.nrm, R, D, { mode: 1, V: ws.ada, scOff: a0 + 4 * D, shOff: a0 + 3 * D });
      this.gemm(ws.nrm, L.ff1, ws.ff, R, { epi: 1 });
      this.gemm(ws.ff, L.ff2, ws.h, R, { epi: 3, G: ws.ada, gOff: a0 + 5 * D });
    }
    this.gemm(ws.ts, this.norm_out, ws.fin, 1);
    this.ln(ws.h, ws.nrm, R, D, { mode: 1, V: ws.fin, scOff: 0, shOff: D });
    this.gemm(ws.nrm, this.proj_out, ws.pred, R);
  },

  workspace(N, B) {
    const k = this.cfg, P = this.pool, D = k.dim, M = k.mel_dim, TD = k.text_dim, INNER = k.heads * k.dim_head, R = B * N, I = TD * k.conv_mult;
    return {
      y: P.get("y", N * M), cond: P.get("cond", R * M), temb: P.get("temb", R * TD), cat: P.get("cat", R * (2 * M + TD)),
      h: P.get("h", R * D), cv: P.get("cv", N * D), nrm: P.get("nrm", R * D),
      q: P.get("q", R * INNER), k: P.get("k", R * INNER), v: P.get("v", R * INNER), at: P.get("at", R * INNER),
      ff: P.get("ff", R * D * k.ff_mult), ada: P.get("ada", k.depth * 6 * D), ts: P.get("ts", D), fin: P.get("fin", 2 * D),
      pred: P.get("pred", R * M), cs: P.get("cs", N * (k.dim_head >> 1) * 2),
      y1: P.get("ty1", N * TD), y2: P.get("ty2", N * TD), hb: P.get("thb", N * I), gx: P.get("tgx", I),
    };
  },

  // Цикл CFM. Возвращает mel [N][mel] (первые condFrames кадров заменены на mel образца, как в F5).
  async sample(condMel, condFrames, ids, N, smp, noise, opt = {}) {
    const k = this.cfg, M = k.mel_dim, g = this.gpu;
    if (N < condFrames) fail("F5-TTS: длительность короче образца", "arg");
    if (g.lost) fail("устройство WebGPU потеряно: " + (g.lost.message || g.lost.reason), "device-lost");
    const useCfg = smp.cfg >= 1e-5, B = useCfg ? 2 : 1, R = B * N;
    const ws = this.workspace(N, B);
    const t0 = performance.now();
    // условие: mel образца + нули; безусловная ветка — нули
    const cond = new Float32Array(R * M);
    cond.set(condMel.subarray(0, condFrames * M), 0);
    g.write(ws.cond, cond);
    g.write(ws.y, noise.subarray(0, N * M));
    // таблица RoPE (пары (2j, 2j+1), угол = позиция * theta^(-2j/HD))
    const HH = k.dim_head >> 1, cs = new Float32Array(N * HH * 2);
    for (let j = 0; j < HH; j++) {
      const inv = f32r(1 / Math.pow(10000, (2 * j) / k.dim_head));
      for (let i = 0; i < N; i++) { const a = f32r(f32r(i) * inv); cs[(i * HH + j) * 2] = Math.cos(a); cs[(i * HH + j) * 2 + 1] = Math.sin(a); }
    }
    g.write(ws.cs, cs);
    // текстовые эмбеддинги обеих веток
    g.begin();
    this.textEmbed(ids, N, false, 0, ws.temb, ws);
    if (useCfg) this.textEmbed(ids, N, true, 1, ws.temb, ws);
    g.submit();
    await g.idle();
    this.timing.text_ms += performance.now() - t0;
    const ts = f5Timesteps(smp.nfe, smp.sway, smp.epss);
    const t1 = performance.now();
    for (let i = 0; i + 1 < ts.length; i++) {
      if (opt.signal && opt.signal.aborted) fail("синтез отменён", "aborted");
      g.write(ws.ts, this.timeVector(ts[i]));
      g.begin();
      this.encodeDit(ws, B, N);
      const dt = f32r(ts[i + 1] - ts[i]);
      if (useCfg) this.ew("euler", [ws.pred, null, null, ws.y], { n: N * M, f0: dt, f1: smp.cfg });
      else this.ew("euler1", [ws.pred, null, null, ws.y], { n: N * M, f0: dt });
      g.submit();
      await g.idle();
      if (opt.onStep) opt.onStep(i + 1, ts.length - 1);
    }
    const out = await g.read(ws.y, N * M);
    out.set(condMel.subarray(0, condFrames * M), 0);
    this.timing.dit_ms += performance.now() - t1;
    return out;
  },

  // Vocos: mel [T][mel] → звук
  async vocode(mel, T) {
    const k = this.cfg, g = this.gpu, P = this.pool, V = this.vk, D = V.dim, I = V.inter, nb = (V.n_fft >> 1) + 1;
    const t0 = performance.now();
    const mb = P.get("vmel", T * k.mel_dim), xa = P.get("vxa", T * D), xb = P.get("vxb", T * D), y1 = P.get("vy1", T * D), y2 = P.get("vy2", T * D);
    const hb = P.get("vh", T * I), head = P.get("vhead", T * 2 * nb);
    g.write(mb, mel.subarray(0, T * k.mel_dim));
    g.begin();
    this.convk(mb, this.v_embed, xa, T, 0);
    this.ln(xa, xb, T, D, { W: this.v_ln.w, Bv: this.v_ln.b });
    for (const L of this.v_blocks) {
      this.convk(xb, L.dw, y1, T, 0);
      this.ln(y1, y2, T, D, { W: L.ln_w, Bv: L.ln_b });
      this.gemm(y2, L.pw1, hb, T, { epi: 2 });
      this.gemm(hb, L.pw2, xb, T, { epi: 3, G: L.gamma });
    }
    this.ln(xb, y2, T, D, { W: this.v_fln.w, Bv: this.v_fln.b });
    this.gemm(y2, this.v_head, head, T);
    g.submit();
    const h = await g.read(head, T * 2 * nb);
    const pcm = istftSame(h, T, V.n_fft, V.hop);
    this.timing.vocos_ms += performance.now() - t0;
    return pcm;
  },
});

// ============================================================================
// Публичный API: Vox.load(), model.cloneVoice(), model.synthesize()/stream(), утилиты WAV/воспроизведения.
// ============================================================================
const DEFAULTS = { nfe_step: 32, cfg_strength: 2.0, sway_sampling_coef: -1.0, use_epss: true, speed: 1.0, cross_fade_duration: 0.15, target_rms: 0.1, seed: 1 };

// Голос = mel-спектрограмма образца + его текст + громкость образца. Хранится в памяти страницы; export()/Voice.from() — в JSON.
class Voice {
  constructor(d) { Object.assign(this, d); }
  get seconds() { return this.samples / this.sampleRate; }
  export() {
    const b = new Uint8Array(this.mel.buffer, this.mel.byteOffset, this.mel.byteLength);
    let s = "";
    for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000));
    return { vox_voice: 1, name: this.name, refText: this.refText, rms: this.rms, samples: this.samples, sampleRate: this.sampleRate, frames: this.frames, melDim: this.melDim, mel: btoa(s) };
  }
  static from(o) {
    if (typeof o === "string") o = JSON.parse(o);
    if (!o || o.vox_voice !== 1) fail("это не голос vox.js (ожидался JSON из voice.export())", "arg");
    const bin = atob(o.mel), u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Voice({ name: o.name, refText: o.refText, rms: o.rms, samples: o.samples, sampleRate: o.sampleRate, frames: o.frames, melDim: o.melDim, mel: new Float32Array(u8.buffer) });
  }
}

async function decodeAudio(src) {
  if (src && src.getChannelData) {   // AudioBuffer
    const ch = src.numberOfChannels, n = src.length, out = new Float32Array(n);
    for (let c = 0; c < ch; c++) { const d = src.getChannelData(c); for (let i = 0; i < n; i++) out[i] += d[i] / ch; }
    return { pcm: out, sampleRate: src.sampleRate };
  }
  let buf;
  if (typeof src === "string") { const r = await fetch(src); if (!r.ok) fail("не удалось загрузить аудио: HTTP " + r.status, "network"); buf = await r.arrayBuffer(); }
  else if (typeof Blob !== "undefined" && src instanceof Blob) buf = await src.arrayBuffer();
  else if (src instanceof ArrayBuffer) buf = src.slice(0);
  else if (ArrayBuffer.isView(src)) buf = src.buffer.slice(src.byteOffset, src.byteOffset + src.byteLength);
  else fail("неизвестный источник звука", "arg");
  const Ctx = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!Ctx) fail("в этом окружении нет AudioContext: передайте Float32Array и sampleRate", "arg");
  const ctx = new Ctx();
  try { return await decodeAudio(await ctx.decodeAudioData(buf)); }
  catch (e) { if (e instanceof VoxError) throw e; fail("браузер не смог декодировать аудио (поддерживаются wav, mp3, ogg, m4a, webm): " + (e && e.message || e), "decode"); }
  finally { if (ctx.close) ctx.close(); }
}

function wavBlob(pcm, sampleRate, opts = {}) {
  const float = !!opts.float, bps = float ? 4 : 2, n = pcm.length;
  const buf = new ArrayBuffer(44 + n * bps), dv = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); dv.setUint32(4, 36 + n * bps, true); str(8, "WAVE"); str(12, "fmt ");
  dv.setUint32(16, 16, true); dv.setUint16(20, float ? 3 : 1, true); dv.setUint16(22, 1, true); dv.setUint32(24, sampleRate, true);
  dv.setUint32(28, sampleRate * bps, true); dv.setUint16(32, bps, true); dv.setUint16(34, bps * 8, true); str(36, "data"); dv.setUint32(40, n * bps, true);
  for (let i = 0; i < n; i++) {
    if (float) dv.setFloat32(44 + i * 4, pcm[i], true);
    else dv.setInt16(44 + i * 2, Math.max(-32768, Math.min(32767, Math.round(pcm[i] * 32767))), true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

function playPcm(pcm, sampleRate, ctx) {
  const Ctx = globalThis.AudioContext || globalThis.webkitAudioContext;
  const ac = ctx || new Ctx();
  const ab = ac.createBuffer(1, pcm.length, sampleRate);
  ab.copyToChannel(pcm, 0);
  const src = ac.createBufferSource();
  src.buffer = ab; src.connect(ac.destination);
  const ended = new Promise((res) => { src.onended = res; });
  src.start();
  return { stop: () => { try { src.stop(); } catch (e) { /* уже остановлено */ } }, ended, context: ac };
}

// Проигрыватель потока: куски ставятся встык по расписанию, без щелчков.
class StreamPlayer {
  constructor(sampleRate, ctx) {
    const Ctx = globalThis.AudioContext || globalThis.webkitAudioContext;
    this.ac = ctx || new Ctx();
    this.sr = sampleRate;
    this.t = 0;
    this.nodes = [];
  }
  push(pcm) {
    if (!pcm.length) return;
    const ab = this.ac.createBuffer(1, pcm.length, this.sr);
    ab.copyToChannel(pcm, 0);
    const s = this.ac.createBufferSource();
    s.buffer = ab; s.connect(this.ac.destination);
    const at = Math.max(this.t, this.ac.currentTime + 0.05);
    s.start(at);
    this.t = at + pcm.length / this.sr;
    this.nodes.push(s);
  }
  get ended() { return new Promise((res) => { const w = () => (this.ac.currentTime >= this.t ? res() : setTimeout(w, 50)); w(); }); }
  stop() { for (const s of this.nodes) { try { s.stop(); } catch (e) { /* ок */ } } this.nodes = []; }
}

class VoxModel {
  constructor(gpu, tm, f5) {
    this.gpu = gpu; this.f5 = f5;
    this.manifest = tm.manifest;
    this.sampleRate = f5.cfg.sample_rate;
    this.generation = Object.assign({}, DEFAULTS, tm.manifest.generation || {});
    const m = tm.manifest.model || {};
    this.info = {
      name: m.name || "", family: m.family || "f5-tts", variant: (tm.manifest.f5 && tm.manifest.f5.variant) || "",
      sampleRate: this.sampleRate, weightBytes: f5.wbytes, device: gpu.info, precision: tm.manifest.precision || {},
    };
  }

  static async load(src, opts = {}) {
    const t0 = performance.now();
    const bytes = await fetchBytes(src, opts.onProgress);
    const tm = new Tmod(bytes);
    const fam = (tm.manifest.model || {}).family;
    if (fam && fam !== "f5-tts") fail("vox.js пока умеет только семейство f5-tts, а в файле: " + fam, "unsupported");
    if (opts.onStatus) opts.onStatus("инициализация WebGPU…");
    const gpu = opts.gpu || await Gpu.create(opts);
    if (opts.onStatus) opts.onStatus("загрузка весов в GPU…");
    const f5 = new F5Gpu(gpu, tm);
    tm.bytes = null;   // веса уже в видеопамяти
    const m = new VoxModel(gpu, tm, f5);
    m.info.loadMs = performance.now() - t0;
    return m;
  }

  // ---- голос ----
  // audio: Float32Array (+ opts.sampleRate), AudioBuffer, Blob/File, ArrayBuffer, URL. refText — что сказано в образце.
  async cloneVoice(audio, refText, opts = {}) {
    if (typeof refText !== "string" || !refText.trim()) fail("нужен текст образца (refText): что произносится в записи", "arg");
    let pcm, sr;
    if (audio instanceof Float32Array) { pcm = audio; sr = opts.sampleRate; if (!sr) fail("для Float32Array укажите opts.sampleRate", "arg"); }
    else ({ pcm, sampleRate: sr } = await decodeAudio(audio));
    const k = this.f5.cfg, target = this.generation.target_rms;
    let a = resampleSinc(pcm, sr, k.sample_rate);
    const maxn = 12 * k.sample_rate;       // как в preprocess_ref_audio_text: не больше 12 с образца
    if (a.length > maxn) a = a.slice(0, maxn); else if (a === pcm) a = a.slice();
    if (a.length < k.mel.n_fft) fail("образец слишком короткий: нужно хотя бы пара секунд речи", "arg");
    let ss = 0;
    for (let i = 0; i < a.length; i++) ss += a[i] * a[i];
    const rms = f32r(Math.sqrt(ss / a.length));
    if (rms < target && rms > 0) { const g = f32r(target / rms); for (let i = 0; i < a.length; i++) a[i] *= g; }
    const { mel, frames } = melLogSpectrogram(k.mel, this.f5.melFb, a);
    return new Voice({ name: opts.name || "", refText: refText.trim(), rms, samples: a.length, sampleRate: k.sample_rate, frames, melDim: k.mel_dim, mel });
  }

  // ---- синтез ----
  _opts(o) {
    const g = this.generation, num = (key, d) => (o[key] != null ? Number(o[key]) : g[key] != null ? Number(g[key]) : d);
    if (o.fix_duration != null) fail("fix_duration пока не поддерживается (используйте speed)", "unsupported");
    const s = {
      nfe: Math.max(1, Math.trunc(num("nfe_step", 32))), cfg: f32r(num("cfg_strength", 2)), sway: f32r(num("sway_sampling_coef", -1)),
      epss: (o.use_epss != null ? !!o.use_epss : g.use_epss != null ? !!g.use_epss : true),
      speed: f32r(num("speed", 1)), crossFade: f32r(num("cross_fade_duration", 0.15)), targetRms: f32r(num("target_rms", 0.1)),
      seed: Math.trunc(num("seed", 1)), maxChars: o.max_chars != null ? Math.trunc(o.max_chars) : 0,
    };
    if (!(s.speed > 0)) fail("speed должен быть положительным", "arg");
    return s;
  }

  async *stream(text, opts = {}) {
    const voice = opts.voice;
    if (!(voice instanceof Voice)) fail("передайте голос: opts.voice = await model.cloneVoice(...)", "arg");
    if (voice.melDim !== this.f5.cfg.mel_dim) fail("голос создан для другой модели", "arg");
    if (typeof text !== "string" || !text.trim()) fail("пустой текст", "arg");
    const s = this._opts(opts), k = this.f5.cfg;
    let rt = voice.refText;
    if (!rt.endsWith(". ") && !rt.endsWith("。")) rt += rt.endsWith(".") ? " " : ". ";
    const refAudioLen = Math.floor(voice.samples / k.mel.hop);
    const refSec = f32r(voice.samples / k.sample_rate);
    let maxChars = s.maxChars;
    if (maxChars <= 0) maxChars = Math.max(1, Math.trunc(blen(rt) / refSec * (22 - refSec) * s.speed));
    const chunks = chunkText(text, maxChars);
    const cfMax = Math.floor(f32r(Math.max(0, s.crossFade) * k.sample_rate));
    let tail = new Float32Array(0);
    const t0 = performance.now();
    this.f5.timing = { dit_ms: 0, vocos_ms: 0, text_ms: 0 };   // статистика — за этот запуск
    let produced = 0;
    for (let idx = 0; idx < chunks.length; idx++) {
      if (opts.signal && opts.signal.aborted) fail("синтез отменён", "aborted");
      const gen = chunks[idx];
      let ref = rt;
      if (ref.charCodeAt(ref.length - 1) < 0x80) ref += " ";
      const ids = this.f5.tok.encode(ref + gen);
      const localSpeed = blen(gen) < 10 ? f32r(0.3) : s.speed;
      let dur = refAudioLen + Math.trunc(refAudioLen / blen(ref) * blen(gen) / localSpeed);
      dur = Math.max(dur, Math.max(ids.length, voice.frames) + 1);
      dur = Math.min(dur, 65536);
      let noise;
      if (opts.noise) noise = typeof opts.noise === "function" ? opts.noise(idx, dur * k.mel_dim) : opts.noise[idx];
      if (!noise) noise = f5Noise(s.seed, idx, dur * k.mel_dim);
      if (noise.length < dur * k.mel_dim) fail("переданный шум короче нужного для куска " + idx, "arg");
      const mel = await this.f5.sample(voice.mel, voice.frames, ids, dur, { nfe: s.nfe, cfg: s.cfg, sway: s.sway, epss: s.epss }, noise,
        { signal: opts.signal, onStep: opts.onStep ? (a, b) => opts.onStep({ chunk: idx, chunks: chunks.length, step: a, steps: b }) : null });
      const from = refAudioLen;
      const wave = await this.f5.vocode(mel.subarray(from * k.mel_dim), dur - from);
      if (voice.rms < s.targetRms && voice.rms > 0) { const g = f32r(voice.rms / s.targetRms); for (let i = 0; i < wave.length; i++) wave[i] *= g; }
      let piece;
      if (cfMax === 0) piece = wave;
      else {
        let buf;
        if (tail.length === 0 && idx === 0) buf = wave;
        else {
          const cfs = Math.min(cfMax, tail.length, wave.length);
          if (cfs === 0) { buf = new Float32Array(tail.length + wave.length); buf.set(tail); buf.set(wave, tail.length); }
          else {
            buf = new Float32Array(tail.length - cfs + wave.length);
            buf.set(tail.subarray(0, tail.length - cfs));
            for (let i = 0; i < cfs; i++) {
              const fo = cfs === 1 ? 1 : 1 - i / (cfs - 1), fi = cfs === 1 ? 0 : i / (cfs - 1);
              buf[tail.length - cfs + i] = tail[tail.length - cfs + i] * fo + wave[i] * fi;
            }
            buf.set(wave.subarray(cfs), tail.length);
          }
        }
        const keep = Math.min(cfMax, buf.length);
        piece = buf.subarray(0, buf.length - keep);
        tail = buf.slice(buf.length - keep);
      }
      produced += piece.length;
      if (piece.length) yield piece;
    }
    if (tail.length) { produced += tail.length; yield tail; }
    this.lastRun = { audioSeconds: produced / k.sample_rate, wallMs: performance.now() - t0, chunks: chunks.length, timing: Object.assign({}, this.f5.timing) };
  }

  // Весь звук целиком. opts: voice, seed, nfe_step, cfg_strength, sway_sampling_coef, speed, cross_fade_duration, max_chars,
  // onChunk(Float32Array), onStep({chunk,chunks,step,steps}), signal (AbortSignal).
  async synthesize(text, opts = {}) {
    const parts = [];
    let n = 0;
    for await (const p of this.stream(text, opts)) { parts.push(p); n += p.length; if (opts.onChunk) opts.onChunk(p); }
    const pcm = new Float32Array(n);
    let o = 0;
    for (const p of parts) { pcm.set(p, o); o += p.length; }
    return { pcm, sampleRate: this.sampleRate, seconds: n / this.sampleRate, stats: this.lastRun };
  }

  // log-mel спектрограмма готового звука (для отрисовки): { mel: Float32Array [frames][bands], frames, bands }
  spectrogram(pcm) {
    const k = this.f5.cfg;
    const r = melLogSpectrogram(k.mel, this.f5.melFb, pcm);
    return { mel: r.mel, frames: r.frames, bands: k.mel_dim };
  }

  dispose() { this.f5.dispose(); try { this.gpu.device.destroy(); } catch (e) { /* ок */ } }
}

const Vox = {
  version: VOX_VERSION,
  supported: () => typeof navigator !== "undefined" && !!navigator.gpu,
  load: (src, opts) => VoxModel.load(src, opts),
  Voice, VoxError, decodeAudio, wav: wavBlob, play: playPcm, StreamPlayer,
  splitText: chunkText,
  _internal: { Gpu, Tmod, F5Gpu, F5Tokenizer, f5Noise, f5Timesteps, melLogSpectrogram, istftSame, resampleSinc, blen },
};

if (typeof module === "object" && module.exports) module.exports = Vox;
global.Vox = Vox;
})(typeof globalThis !== "undefined" ? globalThis : window);
