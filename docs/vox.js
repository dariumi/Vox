/*! vox.js 0.19.0 — универсальный TTS-движок vox в браузере (WebGPU). Лицензия — как у проекта: https://github.com/dariumi/Vox */
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
  attn_kv: `// Причинное внимание с GQA над KV-кэшем. Запрос i (0..m) стоит на позиции pos0 + i и видит ключи [first, pos0 + i],
// first = max(0, pos - window + 1) при window > 0. K, V: [cap][nkv * HD]. Q, O: [m][nh * HD].
// Группа = один запрос одной головы; потоков HD (по одному на измерение выхода). {{HD}} — размер головы.
const HD: u32 = {{HD}}u;
const KB: u32 = 32u;

struct P { m: u32, nh: u32, nkv: u32, pos0: u32, window: u32, q_off: u32, o_off: u32, pad0: u32, scale: f32, pad1: f32, pad2: f32, pad3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;

var<workgroup> Qs: array<f32, {{HD}}>;
var<workgroup> Ts: array<f32, 32 * {{HD}}>;
var<workgroup> Sc: array<f32, 32>;
var<workgroup> St: array<f32, 4>;   // m, l, alpha

@compute @workgroup_size({{HD}})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let i = wg.x;
  let h = wg.y;
  let kvh = h / (p.nh / p.nkv);
  let kd = p.nkv * HD;
  let pos = p.pos0 + i;
  var first = 0u;
  if (p.window > 0u && pos + 1u > p.window) { first = pos + 1u - p.window; }
  Qs[t] = Q[p.q_off + i * p.nh * HD + h * HD + t];
  if (t == 0u) { St[0] = -3.0e38; St[1] = 0.0; }
  var acc = 0.0;
  workgroupBarrier();
  for (var k0 = first; k0 <= pos; k0 += KB) {
    for (var e = t; e < KB * HD; e += HD) {
      let j = e / HD;
      let d = e % HD;
      var v = 0.0;
      if (k0 + j <= pos) { v = K[(k0 + j) * kd + kvh * HD + d]; }
      Ts[e] = v;
    }
    workgroupBarrier();
    for (var j = t; j < KB; j += HD) {
      var s = -3.0e38;
      if (k0 + j <= pos) {
        var a = 0.0;
        for (var d = 0u; d < HD; d++) { a += Qs[d] * Ts[j * HD + d]; }
        s = a * p.scale;
      }
      Sc[j] = s;
    }
    workgroupBarrier();
    if (t == 0u) {
      var mx = St[0];
      let m_old = mx;
      for (var j = 0u; j < KB; j++) { mx = max(mx, Sc[j]); }
      let alpha = exp(m_old - mx);
      var l = St[1] * alpha;
      for (var j = 0u; j < KB; j++) {
        var pj = 0.0;
        if (k0 + j <= pos) { pj = exp(Sc[j] - mx); }
        Sc[j] = pj;
        l += pj;
      }
      St[0] = mx; St[1] = l; St[2] = alpha;
    }
    workgroupBarrier();
    for (var e = t; e < KB * HD; e += HD) {
      let j = e / HD;
      let d = e % HD;
      var v = 0.0;
      if (k0 + j <= pos) { v = V[(k0 + j) * kd + kvh * HD + d]; }
      Ts[e] = v;
    }
    workgroupBarrier();
    var a = acc * St[2];
    for (var j = 0u; j < KB; j++) { a += Sc[j] * Ts[j * HD + t]; }
    acc = a;
    workgroupBarrier();
  }
  O[p.o_off + i * p.nh * HD + h * HD + t] = acc / St[1];
}
`,
  cconv: `// Потоковые (причинные) свёртки декодера кодека. Вход = история H (ctx строк) + новый кусок X (T строк), cin каналов.
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
//   swiglu    : Y[i] = silu(X0[i]) * X1[i]
//   add       : Y[y_off + i] += X0[x_off + i]
//   snake     : Y[y_off + i] = x + X2[c] * sin(x * X1[c])^2, x = X0[x_off + i], c = i % a   (X1 = exp(alpha), X2 = 1/(exp(beta)+1e-9))
//   clampcopy : Y[y_off + i] = clamp(X0[x_off + i], -1, 1)
struct P { n: u32, a: u32, b: u32, c: u32, d: u32, x_off: u32, y_off: u32, pad1: u32, f0: f32, f1: f32, f2: f32, f3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X0: array<f32>;
@group(0) @binding(2) var<storage, read> X1: array<f32>;
@group(0) @binding(3) var<storage, read> X2: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y2: array<f32>;

fn gid1(g: vec3<u32>) -> u32 { return g.x + g.y * 65535u * 64u; }

// sin с настоящей редукцией аргумента и полиномом (cephes sinf): встроенный sin() в WGSL обязан быть точным лишь до 2^-11,
// на практике (SwiftShader, Metal) ошибка доходит до 1e-3 — для Snake-активаций декодера этого слишком много.
fn sin_acc(x: f32) -> f32 {
  let k = round(x * 0.6366197723675814);
  let r = ((x - k * 1.5703125) - k * 4.837512969970703e-4) - k * 7.549789948768648e-8;
  let r2 = r * r;
  let sn = r + r * r2 * (-1.6666654611e-1 + r2 * (8.3321608736e-3 + r2 * -1.9515295891e-4));
  let cs = 1.0 - 0.5 * r2 + r2 * r2 * (4.166664568298827e-2 + r2 * (-1.388731625493765e-3 + r2 * 2.443315711809948e-5));
  let q = u32(i32(k)) & 3u;
  if (q == 0u) { return sn; }
  if (q == 1u) { return cs; }
  if (q == 2u) { return -sn; }
  return -cs;
}

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

@compute @workgroup_size(64)
fn swiglu(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  let v = X0[p.x_off + i];
  Y[p.y_off + i] = (v / (1.0 + exp(-v))) * X1[i];
}

@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  Y[p.y_off + i] = Y[p.y_off + i] + X0[p.x_off + i];
}

@compute @workgroup_size(64)
fn snake(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  let c = i % p.a;
  let v = X0[p.x_off + i];
  let s = sin_acc(v * X1[c]);
  Y[p.y_off + i] = v + X2[c] * s * s;
}

@compute @workgroup_size(64)
fn clampcopy(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = gid1(g);
  if (i >= p.n) { return; }
  Y[p.y_off + i] = clamp(X0[p.x_off + i], -1.0, 1.0);
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
  rms: `// RMSNorm по строкам: y[c] = w[c] * x[c] / sqrt(mean(x²) + eps).
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
`,
  rope: `// RoPE в виде пар (2j, 2j+1) по таблице cs[pos][j] = (cos, sin); на месте для q и k. Строка r = последовательность * N + позиция.
struct P { rows: u32, N: u32, heads: u32, HD: u32, q_off: u32, k_off: u32, hlim: u32, pad1: u32 }   // hlim > 0: rotate only heads < hlim (F5TTS_Base v0: pe_attn_head)
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
  if (p.hlim > 0u && h >= p.hlim) { return; }
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
  rope2: `// RoPE «rotate_half» (как в Qwen/Llama): пары (j, j + HD/2) в каждой голове; на месте.
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
`,
  trans: `// Транспонированная свёртка (stride s, ядро K <= 2 s) в потоковом виде. TMP = gemm(x, W[K*co][cin]) → [T][K*co];
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
`,
};

// ============================================================================
// vox.js — универсальный TTS-движок vox в браузере (WebGPU). Семейство F5-TTS (DiT + Vocos).
// Один файл, без зависимостей:  <script src="vox.js"></script>  →  window.Vox
// Эта часть: утилиты, формат .tmod, форматы весов, Philox-шум, расписание шагов.
// ============================================================================

const VOX_VERSION = "0.19.0";
// Папка, откуда загружен vox.js: рядом лежат pinyin.json и прочие ресурсы. Меняется через Vox.config({ baseUrl }).
let VOX_BASE_URL = "";
try { if (typeof document !== "undefined" && document.currentScript && document.currentScript.src) VOX_BASE_URL = document.currentScript.src.replace(/[^/]*$/, ""); } catch (e) { /* ок */ }

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

// Выход головы Vocos: [T][2*nb] = (log|X|, фаза) → звук. center=false: padding "same" (обрезка (n_fft-hop)/2 с краёв);
// center=true: как torch.istft(center=True) у charactr/vocos-mel-24khz (обрезка n_fft/2, длина (T-1)*hop).
function istftSame(head, T, nfft, hop, center = false) {
  const nb = (nfft >> 1) + 1, pad = center ? (nfft >> 1) : (nfft - hop) >> 1, total = (T - 1) * hop + nfft;
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
  const out = new Float32Array(Math.max(0, total - 2 * pad));
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
  constructor(device, adapterInfo, opts_load) {
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
    this.ops = [];       // записанные диспетчеры текущего begin()…submit()
    this.q = [];         // отправленные, но ещё не исполненные пачки (исполняются порциями в drain())
    this.setLoad(opts_load);
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
    const g = new Gpu(device, { vendor: ai.vendor || "", architecture: ai.architecture || "", device: ai.device || "", description: ai.description || "" }, opts.load);
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
  write(buf, f32, offsetFloats = 0) { this._flushSync(); this.device.queue.writeBuffer(buf, offsetFloats * 4, f32.buffer, f32.byteOffset, f32.byteLength); }
  free(b) { if (b && b !== this.dummyRO) { try { b.destroy(); } catch (e) { /* ок */ } } }

  // конвейер: kernel — имя из KERNELS, defs — подстановки {{...}}
  pipeline(kernel, entry, defs) {
    const key = kernel + "/" + entry + "/" + JSON.stringify(defs || {});
    let p = this.pipes.get(key);
    if (p) return p;
    const made = this._pipeDesc(kernel, entry, defs);
    const pipeline = this.device.createComputePipeline(made.desc);
    p = { pipeline, bgl: made.bgl, entries: made.entries, key, id: this.nextId++, module: made.module };
    this.pipes.set(key, p);
    return p;
  }
  _pipeDesc(kernel, entry, defs) {
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
    const key = kernel + "/" + entry + "/" + JSON.stringify(defs || {});
    const desc = { label: key, layout: this.device.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module, entryPoint: entry } };
    return { desc, bgl, entries, module };
  }

  async checkShader(kernel, defs) {
    let src = KERNELS[kernel];
    for (const k of Object.keys(defs || {})) src = src.split("{{" + k + "}}").join(String(defs[k]));
    const mod = this.device.createShaderModule({ code: src });
    const info = await mod.getCompilationInfo();
    return info.messages.filter((x) => x.type === "error").map((x) => kernel + ":" + x.lineNum + " " + x.message);
  }

  // ---- запись команд ----
  // Диспетчеры записываются в список, а кодируются и отправляются порциями в drain() — после submit() вызывающий делает await g.read()/g.idle().
  // Так видеокарта не получает за раз многосекундную пачку (ОС и браузер не замирают, нет срабатывания сторожевого таймера драйвера).
  begin() {
    this._flushSync();
    this.ops = [];
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
    this.ops.push({ pipeline: pipe.pipeline, bg, off: this.poff, gx, gy });
    this.poff += PARAM_SLOT;
    this.dispatches++;
  }
  // одномерный диспетчер: count потоков по 64 (нумерация gid.x + gid.y * 65535 * 64, как в шейдерах)
  run1d(pipe, bufs, params, count) {
    const groups = Math.max(1, Math.ceil(count / 64));
    this.run(pipe, bufs, params, Math.min(groups, 65535), Math.ceil(groups / 65535));
  }
  submit() {
    this.device.queue.writeBuffer(this.paramBuf, 0, this.pdata, 0, this.poff);
    if (this.ops.length) this.q.push(this.ops);
    this.ops = [];
  }
  _encode(ops) {
    const enc = this.device.createCommandEncoder(), pass = enc.beginComputePass();
    for (const o of ops) { pass.setPipeline(o.pipeline); pass.setBindGroup(0, o.bg, [o.off]); pass.dispatchWorkgroups(o.gx, o.gy, 1); }
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
  // всё накопленное — немедленно одной пачкой (для записей в буферы и копий, которым важен порядок)
  _flushSync() {
    while (this.q.length) this._encode(this.q.shift());
  }
  // Нагрузка: "full" — без пауз, "balanced" (по умолчанию) — GPU занят не более ~70% времени, "gentle" — ~40% (ноутбуки, слабые встроенные видеокарты)
  setLoad(load) {
    const L = typeof load === "object" && load ? load : { full: { duty: 1, ms: 60 }, balanced: { duty: 0.7, ms: 30 }, gentle: { duty: 0.4, ms: 16 } }[load || "balanced"];
    if (!L) fail("load: full | balanced | gentle", "arg");
    this.pace = { duty: Math.min(1, Math.max(0.1, L.duty)), ms: Math.max(4, L.ms) };
    this.sliceOps = this.sliceOps || 48;
    this.pstat = this.pstat || { slices: 0, busyMs: 0, sleptMs: 0, maxSliceMs: 0 };
  }
  // Исполнить накопленное порциями ~pace.ms: размер порции подстраивается по измеренному времени, между порциями — пауза под нагрузку.
  async drain() {
    const P = this.pace;
    while (this.q.length) {
      const batch = this.q.shift();
      for (let i = 0; i < batch.length;) {
        const n = Math.min(this.sliceOps, batch.length - i);
        const t0 = performance.now();
        this._encode(batch.slice(i, i + n));
        i += n;
        if (i >= batch.length && !this.q.length && n === batch.length) return;   // вся пачка помещается в одну порцию: ждать будет read()/idle()
        await this.device.queue.onSubmittedWorkDone();
        const t = performance.now() - t0;
        this.pstat.slices++; this.pstat.busyMs += t; if (t > this.pstat.maxSliceMs) this.pstat.maxSliceMs = t;
        if (t > P.ms * 1.3) this.sliceOps = Math.max(1, Math.floor(this.sliceOps * P.ms / t));
        else if (t < P.ms * 0.6) this.sliceOps = Math.min(4096, Math.ceil(this.sliceOps * 1.5));
        const rest = (i < batch.length || this.q.length) ? Math.round(t * (1 - P.duty) / P.duty) : 0;
        if (rest >= 2 && t >= 4) { this.pstat.sleptMs += rest; await new Promise((r) => setTimeout(r, rest)); }
        else await new Promise((r) => setTimeout(r, 0));   // отдать поток странице (ввод, отрисовка)
      }
    }
  }
  async idle() { await this.drain(); await this.device.queue.onSubmittedWorkDone(); }
  async read(buf, floats, offsetFloats = 0) {
    await this.drain();
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
  // Асинхронная сборка конвейеров заранее (createComputePipeline блокирует поток страницы на время компиляции шейдера)
  async precompile(list, onProgress) {
    let done = 0;
    const one = async ([kernel, entry, defs]) => {
      const key = kernel + "/" + entry + "/" + JSON.stringify(defs || {});
      if (this.pipes.has(key)) return;
      const made = this._pipeDesc(kernel, entry, defs);
      let pipeline;
      try { pipeline = await this.device.createComputePipelineAsync(made.desc); } catch (e) { return; }   // не вышло — соберётся синхронно при первом использовании
      if (!this.pipes.has(key)) this.pipes.set(key, { pipeline, bgl: made.bgl, entries: made.entries, key, id: this.nextId++, module: made.module });
    };
    for (let i = 0; i < list.length; i += 4) {
      await Promise.all(list.slice(i, i + 4).map(one));
      done = Math.min(list.length, i + 4);
      if (onProgress) onProgress(done / list.length);
      await new Promise((r) => setTimeout(r, 0));
    }
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
// Запасной путь без WebGPU: «процессорное устройство» с тем же интерфейсом, что у Gpu (буферы, конвейеры, диспетчеры).
// Каждый шейдер из webgpu/wgsl/ реализован на JS (те же буферы и параметры), поэтому F5-TTS и Qwen3-TTS работают без изменений.
// Команды копятся при submit() и исполняются в idle()/read() небольшими порциями с уступкой потока интерфейсу:
// вкладка не замирает, но считает на одном ядре — это на порядки медленнее видеокарты (годится для Qwen3-TTS 0.6B и коротких фраз).
// ============================================================================
class CBuf {
  constructor(u8, label) { this.u8 = u8; this.label = label || ""; this.size = u8.byteLength; this._f = null; this._u16 = null; }
  get f() { return this._f || (this._f = new Float32Array(this.u8.buffer, this.u8.byteOffset, this.u8.byteLength >> 2)); }
  get u16() { return this._u16 || (this._u16 = new Uint16Array(this.u8.buffer, this.u8.byteOffset, this.u8.byteLength >> 1)); }
}
const _f16tab = (() => { let t = null; return () => { if (!t) { t = new Float32Array(65536); for (let i = 0; i < 65536; i++) t[i] = f16ToF32(i); } return t; }; })();
const _empty = new CBuf(new Uint8Array(16), "empty");

// ---- реализации шейдеров: (defs) → (b, u, f) → { total, cost, fn(lo, hi) }; b — буферы по bindings 1.., u/f — параметры (u32/f32) ----
const _id1 = (b) => b;
const CPU_KERNELS = {
  "gemm/main": (defs) => (b, u, f) => {
    const WT = Number(defs.WTYPE), [X, W, S, B, Y, G] = b;
    const M = u[0], N = u[1], K = u[2], epi = u[3], xo = u[4], yo = u[5], go = u[6], bo = u[7], ldx = u[8], ldy = u[9], hb = u[10];
    const Xf = X.f, Yf = Y.f, Gf = G.f, Bf = B.f, Sf = S.f;
    const tmp = new Float32Array(K), tmpU = new Uint32Array(tmp.buffer);
    const f16 = WT === 1 ? _f16tab() : null;
    const Wf = WT === 0 ? W.f : null, W16 = WT === 1 || WT === 2 ? W.u16 : null, W8 = WT >= 3 ? W.u8 : null;
    const nb = K >> 5;
    return {
      total: N, cost: Math.max(1, M * K * 2),
      fn(lo, hi) {
        for (let n = lo; n < hi; n++) {
          let w;
          if (WT === 0) w = Wf.subarray(n * K, n * K + K);
          else {
            w = tmp;
            if (WT === 1) { const o = n * K; for (let c = 0; c < K; c++) tmp[c] = f16[W16[o + c]]; }
            else if (WT === 2) { const o = n * K; for (let c = 0; c < K; c++) tmpU[c] = W16[o + c] << 16; }
            else if (WT === 3) { const o = n * K, so = n * nb; for (let c = 0; c < K; c++) tmp[c] = ((W8[o + c] << 24) >> 24) * Sf[so + (c >> 5)]; }
            else { const o = n * (K >> 1), so = n * nb; for (let c = 0; c < K; c++) { const blk = c >> 5, j = c & 31; const byte = W8[o + blk * 16 + (j & 15)]; tmp[c] = (((byte >> (j >= 16 ? 4 : 0)) & 15) - 8) * Sf[so + blk]; } }
          }
          for (let m = 0; m < M; m++) {
            const xb = xo + m * ldx;
            let a0 = 0, a1 = 0, a2 = 0, a3 = 0, c = 0;
            for (; c + 3 < K; c += 4) { a0 += Xf[xb + c] * w[c]; a1 += Xf[xb + c + 1] * w[c + 1]; a2 += Xf[xb + c + 2] * w[c + 2]; a3 += Xf[xb + c + 3] * w[c + 3]; }
            for (; c < K; c++) a0 += Xf[xb + c] * w[c];
            let v = (a0 + a1) + (a2 + a3);
            if (hb) v += Bf[bo + n];
            const yi = yo + m * ldy + n;
            switch (epi) {
              case 1: v = 0.5 * v * (1 + _tanhSafe(0.7978845608028654 * (v + 0.044715 * v * v * v))); Yf[yi] = v; break;
              case 2: v = 0.5 * v * (1 + _erfApprox(v * 0.7071067811865476)); Yf[yi] = v; break;
              case 3: Yf[yi] += Gf[go + n] * v; break;
              case 4: Yf[yi] = v / (1 + Math.exp(-v)); break;
              case 5: Yf[yi] += v; break;
              default: Yf[yi] = v;
            }
          }
        }
      },
    };
  },
  "attention/main": (defs) => (b, u, f) => {
    const [Q, K, V, O] = b, HD = Number(defs.HD), N = u[0], heads = u[1], seq = u[2], scale = f[4];
    const Qf = Q.f, Kf = K.f, Vf = V.f, Of = O.f, inner = heads * HD;
    const sc = new Float32Array(N), acc = new Float32Array(HD);
    return {
      total: seq * heads * N, cost: Math.max(1, N * HD * 4),
      fn(lo, hi) {
        for (let i = lo; i < hi; i++) {
          const q = i % N, h = Math.floor(i / N) % heads, sq = Math.floor(i / (N * heads));
          const rb = sq * N, qo = (rb + q) * inner + h * HD;
          let mx = -3e38;
          for (let j = 0; j < N; j++) { const ko = (rb + j) * inner + h * HD; let a = 0; for (let d = 0; d < HD; d++) a += Qf[qo + d] * Kf[ko + d]; a *= scale; sc[j] = a; if (a > mx) mx = a; }
          let l = 0;
          for (let j = 0; j < N; j++) { const e = Math.exp(sc[j] - mx); sc[j] = e; l += e; }
          acc.fill(0);
          for (let j = 0; j < N; j++) { const w = sc[j], vo = (rb + j) * inner + h * HD; for (let d = 0; d < HD; d++) acc[d] += w * Vf[vo + d]; }
          for (let d = 0; d < HD; d++) Of[qo + d] = acc[d] / l;
        }
      },
    };
  },
  "attn_kv/main": (defs) => (b, u, f) => {
    const [Q, K, V, O] = b, HD = Number(defs.HD), m = u[0], nh = u[1], nkv = u[2], pos0 = u[3], window = u[4], qoff = u[5], ooff = u[6], scale = f[8];
    const Qf = Q.f, Kf = K.f, Vf = V.f, Of = O.f, kd = nkv * HD, grp = nh / nkv;
    const sc = new Float32Array(pos0 + m + 1), acc = new Float32Array(HD);
    return {
      total: m * nh, cost: Math.max(1, (pos0 + m) * HD * 4),
      fn(lo, hi) {
        for (let t = lo; t < hi; t++) {
          const i = Math.floor(t / nh), h = t % nh, kvh = Math.floor(h / grp), pos = pos0 + i;
          const first = window > 0 && pos + 1 > window ? pos + 1 - window : 0;
          const qo = qoff + i * nh * HD + h * HD;
          let mx = -3e38;
          for (let j = first; j <= pos; j++) { const ko = j * kd + kvh * HD; let a = 0; for (let d = 0; d < HD; d++) a += Qf[qo + d] * Kf[ko + d]; a *= scale; sc[j] = a; if (a > mx) mx = a; }
          let l = 0;
          for (let j = first; j <= pos; j++) { const e = Math.exp(sc[j] - mx); sc[j] = e; l += e; }
          acc.fill(0);
          for (let j = first; j <= pos; j++) { const w = sc[j], vo = j * kd + kvh * HD; for (let d = 0; d < HD; d++) acc[d] += w * Vf[vo + d]; }
          const oo = ooff + i * nh * HD + h * HD;
          for (let d = 0; d < HD; d++) Of[oo + d] = acc[d] / l;
        }
      },
    };
  },
  "cconv/im2col": () => (b, u) => {
    const [X, H, , , Y] = b, T = u[0], cin = u[1], K = u[2], dil = u[3], ctx = u[4], xo = u[5], yo = u[6];
    const Xf = X.f, Hf = H.f, Yf = Y.f, kc = K * cin;
    return {
      total: T * K, cost: cin,
      fn(lo, hi) {
        for (let i = lo; i < hi; i++) {
          const t = Math.floor(i / K), k = i - t * K, s = t + k * dil, d = yo + t * kc + k * cin;
          if (s < ctx) { const o = s * cin; for (let c = 0; c < cin; c++) Yf[d + c] = Hf[o + c]; }
          else { const o = xo + (s - ctx) * cin; for (let c = 0; c < cin; c++) Yf[d + c] = Xf[o + c]; }
        }
      },
    };
  },
  "cconv/hist": () => (b, u) => {
    const [X, H, , , Y] = b, T = u[0], cin = u[1], ctx = u[4], xo = u[5];
    const Xf = X.f, Hf = H.f, Yf = Y.f;
    const tmp = new Float32Array(ctx * cin);   // история и новая история могут быть одним буфером: сначала считаем, потом пишем
    return {
      total: ctx, cost: cin * 2,
      fn(lo, hi) {
        for (let j = lo; j < hi; j++) {
          const s = T + j, o = j * cin;
          if (s < ctx) { const so = s * cin; for (let c = 0; c < cin; c++) tmp[o + c] = Hf[so + c]; }
          else { const so = xo + (s - ctx) * cin; for (let c = 0; c < cin; c++) tmp[o + c] = Xf[so + c]; }
        }
        if (hi === ctx) Yf.set(tmp, 0);
      },
    };
  },
  "cconv/dw": () => (b, u) => {
    const [X, H, W, B, Y] = b, T = u[0], cin = u[1], K = u[2], ctx = u[4], xo = u[5], yo = u[6], hb = u[7];
    const Xf = X.f, Hf = H.f, Wf = W.f, Bf = B.f, Yf = Y.f;
    return {
      total: T, cost: cin * K * 2,
      fn(lo, hi) {
        for (let t = lo; t < hi; t++) for (let c = 0; c < cin; c++) {
          let a = hb ? Bf[c] : 0;
          for (let k = 0; k < K; k++) { const s = t + k; a += Wf[c * K + k] * (s < ctx ? Hf[s * cin + c] : Xf[xo + (s - ctx) * cin + c]); }
          Yf[yo + t * cin + c] = a;
        }
      },
    };
  },
  "conv1d/main": () => (b, u) => {
    const [X, W, B, Y] = b, T = u[0], cin = u[1], cout = u[2], K = u[3], groups = u[4], epi = u[5], xo = u[6], yo = u[7], hb = u[8];
    const Xf = X.f, Wf = W.f, Bf = B.f, Yf = Y.f, cig = cin / groups, cog = cout / groups, pad = K >> 1;
    return {
      total: T * cout, cost: K * cig * 2,
      fn(lo, hi) {
        for (let i = lo; i < hi; i++) {
          const t = Math.floor(i / cout), o = i - t * cout, g = Math.floor(o / cog);
          let a = hb ? Bf[o] : 0;
          for (let k = 0; k < K; k++) {
            const tt = t + k - pad;
            if (tt < 0 || tt >= T) continue;
            const xb = xo + tt * cin + g * cig, wb = o * cig * K + k;
            for (let c = 0; c < cig; c++) a += Wf[wb + c * K] * Xf[xb + c];
          }
          const yi = yo + t * cout + o;
          if (epi === 1) Yf[yi] = _mish(a); else if (epi === 2) Yf[yi] += _mish(a); else Yf[yi] = a;
        }
      },
    };
  },
  "grn/sumsq": () => (b, u) => {
    const [X, Gx] = b, T = u[0], C = u[1], xo = u[2], Xf = X.f, Gf = Gx.f;
    return { total: C, cost: T * 2, fn(lo, hi) { for (let c = lo; c < hi; c++) { let s = 0; for (let t = 0; t < T; t++) { const v = Xf[xo + t * C + c]; s += v * v; } Gf[c] = s; } } };
  },
  "grn/stats": () => (b, u) => {
    const [, Gx] = b, C = u[1], Gf = Gx.f;
    return { total: 1, cost: C * 4, fn() { let s = 0; for (let c = 0; c < C; c++) { const g = Math.sqrt(Gf[c]); Gf[c] = g; s += g; } const mean = s / C; for (let c = 0; c < C; c++) Gf[c] = Gf[c] / (mean + 1e-6); } };
  },
  "grn/apply": () => (b, u) => {
    const [X, Gx, Gm, Bt] = b, T = u[0], C = u[1], xo = u[2], Xf = X.f, Gf = Gx.f, Mf = Gm.f, Bf = Bt.f;
    return { total: T, cost: C * 2, fn(lo, hi) { for (let t = lo; t < hi; t++) for (let c = 0; c < C; c++) { const i = xo + t * C + c, v = Xf[i]; Xf[i] = Mf[c] * (v * Gf[c]) + Bf[c] + v; } } };
  },
  "ln/main": () => (b, u, f) => {
    const [X, Wt, Bv, V, Y] = b, R = u[0], C = u[1], mode = u[2], xo = u[3], yo = u[4], sc = u[5], sh = u[6], eps = f[8];
    const Xf = X.f, Wf = Wt.f, Bf = Bv.f, Vf = V.f, Yf = Y.f;
    return {
      total: R, cost: C * 8,
      fn(lo, hi) {
        for (let r = lo; r < hi; r++) {
          const bs = xo + r * C;
          let s = 0; for (let c = 0; c < C; c++) s += Xf[bs + c];
          const mean = s / C;
          let v = 0; for (let c = 0; c < C; c++) { const d = Xf[bs + c] - mean; v += d * d; }
          const inv = 1 / Math.sqrt(v / C + eps), ob = yo + r * C;
          if (mode === 0) for (let c = 0; c < C; c++) Yf[ob + c] = (Xf[bs + c] - mean) * inv * Wf[c] + Bf[c];
          else for (let c = 0; c < C; c++) Yf[ob + c] = (Xf[bs + c] - mean) * inv * (1 + Vf[sc + c]) + Vf[sh + c];
        }
      },
    };
  },
  "rms/main": () => (b, u, f) => {
    const [X, Wt, Y] = b, R = u[0], C = u[1], xo = u[2], yo = u[3], eps = f[4], Xf = X.f, Wf = Wt.f, Yf = Y.f;
    return { total: R, cost: C * 4, fn(lo, hi) { for (let r = lo; r < hi; r++) { const bs = xo + r * C; let s = 0; for (let c = 0; c < C; c++) { const v = Xf[bs + c]; s += v * v; } const inv = 1 / Math.sqrt(s / C + eps), ob = yo + r * C; for (let c = 0; c < C; c++) Yf[ob + c] = Wf[c] * (Xf[bs + c] * inv); } } };
  },
  "rms/ip": () => (b, u, f) => {
    const [, Wt, Y] = b, R = u[0], C = u[1], yo = u[3], eps = f[4], Wf = Wt.f, Yf = Y.f;
    return { total: R, cost: C * 4, fn(lo, hi) { for (let r = lo; r < hi; r++) { const bs = yo + r * C; let s = 0; for (let c = 0; c < C; c++) { const v = Yf[bs + c]; s += v * v; } const inv = 1 / Math.sqrt(s / C + eps); for (let c = 0; c < C; c++) Yf[bs + c] = Wf[c] * (Yf[bs + c] * inv); } } };
  },
  "rope/main": () => (b, u) => {
    const [Q, K, CS] = b, rows = u[0], N = u[1], heads = u[2], HD = u[3], qo = u[4], ko = u[5], hlim = u[6], half = HD >> 1, Qf = Q.f, Kf = K.f, Cf = CS.f, inner = heads * HD;
    return {
      total: rows * heads, cost: half * 12,
      fn(lo, hi) {
        for (let i = lo; i < hi; i++) {
          const r = Math.floor(i / heads), h = i - r * heads, pos = r % N;
          if (hlim > 0 && h >= hlim) continue;
          for (let j = 0; j < half; j++) {
            const c = Cf[(pos * half + j) * 2], s = Cf[(pos * half + j) * 2 + 1];
            const qi = qo + r * inner + h * HD + 2 * j, ki = ko + r * inner + h * HD + 2 * j;
            const q0 = Qf[qi], q1 = Qf[qi + 1]; Qf[qi] = q0 * c - q1 * s; Qf[qi + 1] = q1 * c + q0 * s;
            const k0 = Kf[ki], k1 = Kf[ki + 1]; Kf[ki] = k0 * c - k1 * s; Kf[ki + 1] = k1 * c + k0 * s;
          }
        }
      },
    };
  },
  "rope2/main": () => (b, u) => {
    const [X, CS] = b, rows = u[0], heads = u[1], HD = u[2], off = u[3], half = HD >> 1, Xf = X.f, Cf = CS.f;
    return {
      total: rows * heads, cost: half * 8,
      fn(lo, hi) {
        for (let i = lo; i < hi; i++) {
          const r = Math.floor(i / heads), h = i - r * heads, bb = off + r * heads * HD + h * HD;
          for (let j = 0; j < half; j++) {
            const c = Cf[(r * half + j) * 2], s = Cf[(r * half + j) * 2 + 1], a = Xf[bb + j], d = Xf[bb + j + half];
            Xf[bb + j] = a * c - d * s; Xf[bb + j + half] = d * c + a * s;
          }
        }
      },
    };
  },
  "trans/main": () => (b, u) => {
    const [TMP, TAIL, Bias, Y, NT] = b, T = u[0], K = u[1], s = u[2], co = u[3], tr = u[4], yo = u[5], Tf = TMP.f, Af = TAIL.f, Bf = Bias.f, Yf = Y.f, Nf = NT.f;
    const n = T * s + tr;
    return {
      total: n, cost: co * 6,
      fn(lo, hi) {
        for (let pos = lo; pos < hi; pos++) {
          const t = Math.floor(pos / s), k = pos - t * s;
          for (let o = 0; o < co; o++) {
            let v = 0;
            if (pos < tr) v = Af[pos * co + o];
            if (t < T && k < K) v += Tf[(t * K + k) * co + o];
            if (t >= 1 && k + s < K) v += Tf[((t - 1) * K + (k + s)) * co + o];
            if (pos < T * s) Yf[yo + pos * co + o] = v + Bf[o]; else Nf[(pos - T * s) * co + o] = v;
          }
        }
      },
    };
  },
};
// ew.wgsl: одна функция на точку входа
const _EW = {
  concat_in: (b, u) => { const [X0, X1, X2, Y] = b, n = u[0], bb = u[2], mel = u[3], td = u[4], wdt = 2 * mel + td; return { total: n, cost: 2, fn(lo, hi) { for (let i = lo; i < hi; i++) { const r = Math.floor(i / wdt), c = i - r * wdt, pos = r % bb; Y.f[i] = c < mel ? X0.f[pos * mel + c] : c < 2 * mel ? X1.f[r * mel + (c - mel)] : X2.f[r * td + (c - 2 * mel)]; } } }; },
  euler: (b, u, f) => { const [X0, , , Y] = b, n = u[0], dt = f[8], cfg = f[9]; return { total: n, cost: 6, fn(lo, hi) { for (let i = lo; i < hi; i++) { const pc = X0.f[i], pu = X0.f[n + i]; Y.f[i] = Y.f[i] + dt * (pc + (pc - pu) * cfg); } } }; },
  euler1: (b, u, f) => { const [X0, , , Y] = b, n = u[0], dt = f[8]; return { total: n, cost: 3, fn(lo, hi) { for (let i = lo; i < hi; i++) Y.f[i] = Y.f[i] + dt * X0.f[i]; } }; },
  zero_rows: (b, u) => { const Y = b[3], n = u[0], yo = u[6]; return { total: n, cost: 1, fn(lo, hi) { Y.f.fill(0, yo + lo, yo + hi); } }; },
  silu_vec: (b, u) => { const [X0, , , Y] = b, n = u[0], xo = u[5], yo = u[6]; return { total: n, cost: 8, fn(lo, hi) { for (let i = lo; i < hi; i++) { const v = X0.f[xo + i]; Y.f[yo + i] = v / (1 + Math.exp(-v)); } } }; },
  copy: (b, u) => { const [X0, , , Y] = b, n = u[0], xo = u[5], yo = u[6]; return { total: n, cost: 1, fn(lo, hi) { Y.f.set(X0.f.subarray(xo + lo, xo + hi), yo + lo); } }; },
  swiglu: (b, u) => { const [X0, X1, , Y] = b, n = u[0], xo = u[5], yo = u[6]; return { total: n, cost: 10, fn(lo, hi) { for (let i = lo; i < hi; i++) { const v = X0.f[xo + i]; Y.f[yo + i] = (v / (1 + Math.exp(-v))) * X1.f[i]; } } }; },
  add: (b, u) => { const [X0, , , Y] = b, n = u[0], xo = u[5], yo = u[6]; return { total: n, cost: 2, fn(lo, hi) { for (let i = lo; i < hi; i++) Y.f[yo + i] = Y.f[yo + i] + X0.f[xo + i]; } }; },
  snake: (b, u) => { const [X0, X1, X2, Y] = b, n = u[0], a = u[1], xo = u[5], yo = u[6]; return { total: n, cost: 20, fn(lo, hi) { for (let i = lo; i < hi; i++) { const c = i % a, v = X0.f[xo + i], s = Math.sin(v * X1.f[c]); Y.f[yo + i] = v + X2.f[c] * s * s; } } }; },
  clampcopy: (b, u) => { const [X0, , , Y] = b, n = u[0], xo = u[5], yo = u[6]; return { total: n, cost: 2, fn(lo, hi) { for (let i = lo; i < hi; i++) { const v = X0.f[xo + i]; Y.f[yo + i] = v < -1 ? -1 : v > 1 ? 1 : v; } } }; },
};
for (const k of Object.keys(_EW)) { const impl = _EW[k]; CPU_KERNELS["ew/" + k] = () => impl; }

function _tanhSafe(x) { const c = Math.min(15, Math.max(-15, x)), e = Math.exp(2 * c); return (e - 1) / (e + 1); }
function _erfApprox(z) {
  const a = Math.abs(z), t = 1 / (1 + 0.5 * a);
  const r = t * Math.exp(-a * a - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return z >= 0 ? 1 - r : r - 1;
}
function _mish(x) { let sp = x; if (x <= 20) sp = Math.log(1 + Math.exp(x)); const c = Math.min(15, Math.max(-15, sp)), e = Math.exp(2 * c); return x * ((e - 1) / (e + 1)); }

// Устройство на процессоре. Интерфейс совпадает с Gpu.
class CpuGpu {
  constructor(opts = {}) {
    this.info = { vendor: "cpu", architecture: "javascript", device: "", description: "процессор (без WebGPU)" };
    this.isCpu = true;
    this.lost = null;
    this.limits = { maxBuffer: 2147483647, maxBinding: 2147483647 };
    this.pending = [];
    this.rec = null;
    this.bytes = 0; this.dispatches = 0;
    this.pipes = new Map(); this.nextId = 1;
    this.y = new Yielder(opts.sliceMs || 24, () => { if (opts.signal && opts.signal.aborted) fail("синтез отменён", "aborted"); });
    const self = this;
    this.enc = null;
    this.device = {
      limits: { maxBufferSize: 2147483647, maxStorageBufferBindingSize: 2147483647 },
      lost: new Promise(() => {}),
      destroy() {},
      createCommandEncoder() { const ops = []; return { copyBufferToBuffer(s, so, d, doff, n) { ops.push({ k: "copy", s, so, d, doff, n }); }, finish() { return ops; } }; },
      queue: { submit(list) { for (const ops of list) for (const o of ops) self.pending.push(o); }, onSubmittedWorkDone: () => self.idle() },
    };
    this.dummyRO = _empty;
  }
  static async create(opts = {}) { const g = new CpuGpu(opts); if (opts.load) g.setLoad(opts.load); return g; }
  _flushSync() {}
  async precompile() {}
  // нагрузка на процессор: порция работы между уступками потока (мс) — для слабых машин «gentle»
  setLoad(load) { const ms = typeof load === "object" && load ? load.ms : { full: 50, balanced: 24, gentle: 10 }[load || "balanced"]; if (!ms) fail("load: full | balanced | gentle", "arg"); this.y.slice = ms; }

  buffer(bytes, label) {
    const size = Math.max(16, (bytes + 3) & ~3);
    if (size > 2147483647) fail("буфер " + (label || "") + " (" + (size >> 20) + " МиБ) больше лимита процессорного режима", "limits");
    this.bytes += size;
    return new CBuf(new Uint8Array(size), label);
  }
  // буфер с содержимым: без копии, если выровнен (веса остаются теми же байтами, что скачали)
  upload(u8, label) {
    this.bytes += u8.byteLength;
    if ((u8.byteOffset & 3) === 0) return new CBuf(u8, label);
    return new CBuf(u8.slice(), label);
  }
  write(buf, f32, offsetFloats = 0) {
    const data = f32.slice();   // запись упорядочена относительно уже поставленных в очередь команд
    if (this.pending.length || this.rec) this.pending.push({ k: "write", buf, data, off: offsetFloats });
    else buf.f.set(data, offsetFloats);
  }
  free() {}
  idOf() { return 0; }
  pipeline(kernel, entry, defs) {
    const key = kernel + "/" + entry + "/" + JSON.stringify(defs || {});
    let p = this.pipes.get(key);
    if (p) return p;
    const mk = CPU_KERNELS[kernel + "/" + entry];
    if (!mk) fail("нет процессорной реализации шейдера " + kernel + "/" + entry, "internal");
    p = { key, id: this.nextId++, impl: mk(defs || {}) };
    this.pipes.set(key, p);
    return p;
  }
  begin() { this.rec = []; this.enc = { copyBufferToBuffer: (s, so, d, doff, n) => this.rec.push({ k: "copy", s, so, d, doff, n }) }; this.dispatches = 0; }
  run(pipe, bufs, params) {
    const pb = new ArrayBuffer(64), u = new Uint32Array(pb), f = new Float32Array(pb);
    for (let i = 0; i < params.length; i++) { const v = params[i]; if (typeof v === "object") f[i] = v.f; else u[i] = v >>> 0; }
    this.rec.push({ k: "run", pipe, bufs: bufs.map((x) => x || _empty), u, f });
    this.dispatches++;
  }
  run1d(pipe, bufs, params) { this.run(pipe, bufs, params); }
  submit() { for (const o of this.rec) this.pending.push(o); this.rec = null; this.enc = null; }
  async idle() {
    while (this.pending.length) {
      const o = this.pending.shift();
      if (o.k === "write") o.buf.f.set(o.data, o.off);
      else if (o.k === "copy") o.d.u8.set(o.s.u8.subarray(o.so, o.so + o.n), o.doff);
      else {
        const job = o.pipe.impl(o.bufs, o.u, o.f), step = Math.max(1, Math.floor(4e6 / Math.max(1, job.cost)));
        for (let lo = 0; lo < job.total; lo += step) { job.fn(lo, Math.min(job.total, lo + step)); await this.y.tick(); }
      }
    }
  }
  async read(buf, floats, offsetFloats = 0) {
    await this.idle();
    return buf.f.slice(offsetFloats, offsetFloats + floats);
  }
}

// Выбор устройства: backend "auto" (WebGPU, а если его нет или он не поднялся — процессор), "webgpu" или "cpu".
async function createDevice(opts = {}) {
  const want = opts.backend || "auto";
  if (want === "cpu") return CpuGpu.create(opts);
  try { return await Gpu.create(opts); }
  catch (e) {
    if (want === "webgpu" || (e && e.code && e.code !== "no-webgpu" && e.code !== "no-adapter")) throw e;
    if (opts.onStatus) opts.onStatus("WebGPU недоступен — считаю на процессоре (медленно)");
    return CpuGpu.create(opts);
  }
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
      pe_attn_head: Number.isInteger(c.pe_attn_head) && c.pe_attn_head > 0 ? c.pe_attn_head : 0,
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
    this.vk = { dim: vc.dim, inter: vc.intermediate, layers: vc.layers, n_fft: vc.n_fft || 1024, hop: vc.hop || 256, center: vc.padding === "center" };
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
// Токенизатор Qwen2/Qwen3 (byte-level BPE) — порт src/tokenizer.cpp: тот же претокенизатор, те же ранги слияний,
// те же особые токены и та же «безопасная граница» для потокового ввода.
// ============================================================================
const _reLetter = /\p{L}/u, _reNumber = /\p{N}/u, _reSpace = /\p{White_Space}/u;
const _isLetter = (c) => _reLetter.test(c), _isNumber = (c) => _reNumber.test(c), _isSpace = (c) => _reSpace.test(c);
const _isNl = (c) => c === "\r" || c === "\n";

// Претокены: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
function qwenPretokenize(s) {
  const c = Array.from(s), n = c.length, out = [];
  const lower = (ch) => (ch >= "A" && ch <= "Z" ? String.fromCharCode(ch.charCodeAt(0) + 32) : ch);
  const ci = (i, ch) => i < n && lower(c[i]) === ch;
  let i = 0;
  while (i < n) {
    let end = 0;
    if (c[i] === "'" && i + 1 < n) {
      if (ci(i + 1, "s") || ci(i + 1, "t") || ci(i + 1, "m") || ci(i + 1, "d")) end = i + 2;
      else if ((ci(i + 1, "r") && ci(i + 2, "e")) || (ci(i + 1, "v") && ci(i + 2, "e")) || (ci(i + 1, "l") && ci(i + 2, "l"))) end = i + 3;
    }
    if (!end) {
      let j = i;
      if (!_isNl(c[j]) && !_isLetter(c[j]) && !_isNumber(c[j]) && j + 1 < n && _isLetter(c[j + 1])) j = i + 1;
      if (_isLetter(c[j])) { while (j < n && _isLetter(c[j])) j++; end = j; }
    }
    if (!end && _isNumber(c[i])) end = i + 1;
    if (!end) {
      let j = i;
      if (c[j] === " " && j + 1 < n && !_isSpace(c[j + 1]) && !_isLetter(c[j + 1]) && !_isNumber(c[j + 1])) j = i + 1;
      if (!_isSpace(c[j]) && !_isLetter(c[j]) && !_isNumber(c[j])) {
        while (j < n && !_isSpace(c[j]) && !_isLetter(c[j]) && !_isNumber(c[j])) j++;
        while (j < n && _isNl(c[j])) j++;
        end = j;
      }
    }
    if (!end && _isSpace(c[i])) {
      let run = i;
      while (run < n && _isSpace(c[run])) run++;
      let afterNl = 0;
      for (let k = run; k > i; k--) if (_isNl(c[k - 1])) { afterNl = k; break; }
      if (afterNl) end = afterNl;
      else if (run === n) end = run;
      else if (run - i >= 2) end = run - 1;
      else end = run;
    }
    if (!end) end = i + 1;
    out.push(c.slice(i, end).join(""));
    i = end;
  }
  return out;
}

class QwenTokenizer {
  // vocabText: строки "id\tтокен"; mergesText: строки "a b" по рангу; specials: токены, которые ищутся в тексте буквально.
  constructor(vocabText, mergesText, specials) {
    const bs = [];
    for (let b = 33; b <= 126; b++) bs.push(b);
    for (let b = 161; b <= 172; b++) bs.push(b);
    for (let b = 174; b <= 255; b++) bs.push(b);
    const cs = bs.slice();
    let extra = 0;
    for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + extra++); }
    this.byteUni = new Array(256);
    for (let k = 0; k < bs.length; k++) this.byteUni[bs[k]] = String.fromCodePoint(cs[k]);
    this.tokToId = new Map();
    let maxId = -1;
    for (const line of vocabText.split("\n")) {
      if (!line) continue;
      const tab = line.indexOf("\t");
      if (tab < 0) fail("токенизатор: плохая строка словаря", "format");
      const id = parseInt(line.slice(0, tab), 10);
      this.tokToId.set(line.slice(tab + 1), id);
      if (id > maxId) maxId = id;
    }
    this.vocabSize = maxId + 1;
    this.merges = new Map();
    let rank = 0;
    for (const line of mergesText.split("\n")) {
      if (!line || line[0] === "#") continue;
      const sp = line.indexOf(" ");
      if (sp < 0) continue;
      const a = line.slice(0, sp), b = line.slice(sp + 1);
      const ia = this.tokToId.get(a), ib = this.tokToId.get(b), im = this.tokToId.get(a + b);
      if (ia !== undefined && ib !== undefined && im !== undefined) this.merges.set(ia * 4294967296 + ib, [rank, im]);
      rank++;
    }
    this.specials = [];
    for (const s of specials || []) { const id = this.tokToId.get(s); if (id !== undefined) this.specials.push([s, id]); }
    this.specials.sort((x, y) => y[0].length - x[0].length);
    this.cache = new Map();
  }
  _bpe(word, out) {
    const hit = this.cache.get(word);
    if (hit) { for (const t of hit) out.push(t); return; }
    const sym = [];
    for (const b of new TextEncoder().encode(word)) {
      const id = this.tokToId.get(this.byteUni[b]);
      if (id === undefined) fail("в словаре токенизатора нет однобайтового токена", "format");
      sym.push(id);
    }
    while (sym.length > 1) {
      let best = Infinity, bi = -1, bid = 0;
      for (let k = 0; k + 1 < sym.length; k++) {
        const m = this.merges.get(sym[k] * 4294967296 + sym[k + 1]);
        if (m && m[0] < best) { best = m[0]; bi = k; bid = m[1]; }
      }
      if (bi < 0) break;
      sym[bi] = bid;
      sym.splice(bi + 1, 1);
    }
    if (this.cache.size > 50000) this.cache.clear();
    this.cache.set(word, sym.slice());
    for (const t of sym) out.push(t);
  }
  _plain(s, out) { for (const p of qwenPretokenize(s)) this._bpe(p, out); }
  encode(s) {
    const out = [];
    let seg = 0, i = 0;
    while (i < s.length) {
      let matched = false;
      if (s[i] === "<") {
        for (const [tok, id] of this.specials) {
          if (s.startsWith(tok, i)) {
            if (i > seg) this._plain(s.slice(seg, i), out);
            out.push(id);
            i += tok.length; seg = i; matched = true;
            break;
          }
        }
      }
      if (!matched) i++;
    }
    if (seg < s.length) this._plain(s.slice(seg), out);
    return out;
  }
  // Граница в строке JS (индекс UTF-16), до которой пока можно токенизировать, не меняя результат относительно целого текста;
  // 0 — пока нельзя. Порт Tokenizer::safe_cut.
  safeCut(pending) {
    const u = Array.from(pending), offs = [];
    let o = 0;
    for (const ch of u) { offs.push(o); o += ch.length; }
    for (let i = u.length; i-- > 1;) {
      if (_isSpace(u[i]) && !_isSpace(u[i - 1]) && (_isLetter(u[i - 1]) || _isNumber(u[i - 1]))) return offs[i];
    }
    return 0;
  }
}

// ============================================================================
// Qwen3-TTS на WebGPU: talker (трансформер с GQA и KV-кэшем), code predictor, потоковый декодер кодека (RVQ → pre-conv →
// трансформер с окном → апсэмплинг ConvNeXt → Snake-декодер). Веса — в видеопамяти; на CPU остаются таблицы эмбеддингов,
// сэмплинг и сборка входов. Математика та же, что в ядре (src/backend_qwen3.cpp, src/qwen3.cpp).
// ============================================================================
const QCHUNK_MAX_FRAMES = 64;   // за один вызов декодера кодека

// Историческое окно «причинной» свёртки / транспонированной свёртки: пара буферов (текущий, следующий), меняются местами
class QHist {
  constructor(gpu, floats, label) { this.gpu = gpu; this.n = floats; this.a = gpu.buffer(Math.max(16, floats * 4), label + ".a"); this.b = gpu.buffer(Math.max(16, floats * 4), label + ".b"); }
  get cur() { return this.a; }
  get next() { return this.b; }
  swap() { const t = this.a; this.a = this.b; this.b = t; }
  copyFrom(o, enc) { enc.copyBufferToBuffer(o.a, 0, this.a, 0, Math.max(16, this.n * 4)); }
  free() { this.gpu.free(this.a); this.gpu.free(this.b); }
}

class QwenGpu {
  constructor(gpu, tm) {
    this.gpu = gpu;
    this.tm = tm;
    this.pool = new Pool(gpu);
    const q = tm.manifest.qwen3;
    if (!q) fail("в этом .tmod нет раздела qwen3: это не модель Qwen3-TTS", "format");
    const t = q.talker, p = q.predictor, c = q.codec, i = q.ids;
    const lower = (o) => { const r = {}; for (const k of Object.keys(o || {})) r[k.toLowerCase()] = o[k]; return r; };
    this.tc = {
      hidden: t.hidden_size, inter: t.intermediate_size, layers: t.num_hidden_layers, heads: t.num_attention_heads, kv_heads: t.num_key_value_heads,
      head_dim: t.head_dim || t.hidden_size / t.num_attention_heads, vocab: t.vocab_size, text_hidden: t.text_hidden_size, text_vocab: t.text_vocab_size,
      groups: t.num_code_groups, eps: f32r(t.rms_norm_eps ?? 1e-6), theta: t.rope_theta ?? 1e6,
      eos: t.codec_eos_token_id, think: t.codec_think_id, nothink: t.codec_nothink_id, think_bos: t.codec_think_bos_id, think_eos: t.codec_think_eos_id,
      pad: t.codec_pad_id, bos: t.codec_bos_id, lang: lower(t.codec_language_id), spk_id: {}, spk_dialect: lower(t.spk_is_dialect),
    };
    const spk = lower(t.spk_id);
    for (const k of Object.keys(spk)) this.tc.spk_id[k] = Array.isArray(spk[k]) ? spk[k][0] : spk[k];
    this.pc = {
      hidden: p.hidden_size, inter: p.intermediate_size, layers: p.num_hidden_layers, heads: p.num_attention_heads, kv_heads: p.num_key_value_heads,
      head_dim: p.head_dim || p.hidden_size / p.num_attention_heads, vocab: p.vocab_size, eps: f32r(p.rms_norm_eps ?? 1e-6), theta: p.rope_theta ?? 1e6,
    };
    this.cc = {
      codebook_size: c.codebook_size, hidden: c.hidden_size, latent: c.latent_dim, heads: c.num_attention_heads, kv_heads: c.num_key_value_heads,
      head_dim: c.head_dim || c.hidden_size / c.num_attention_heads, window: c.sliding_window, inter: c.intermediate_size, layers: c.num_hidden_layers,
      quantizers: c.num_quantizers, decoder_dim: c.decoder_dim, codebook_dim: c.codebook_dim, eps: f32r(c.rms_norm_eps ?? 1e-5), theta: c.rope_theta ?? 10000,
      up_rates: c.upsample_rates.slice(), up_ratios: c.upsampling_ratios.slice(),
    };
    this.ids = { im_start: i.im_start, im_end: i.im_end, tts_pad: i.tts_pad, tts_bos: i.tts_bos, tts_eos: i.tts_eos };
    this.cc.samples_per_frame = [...this.cc.up_rates, ...this.cc.up_ratios].reduce((a, b) => a * b, 1);
    if (this.cc.quantizers !== this.tc.groups) fail("число квантизаторов кодека не равно числу групп talker", "format");
    this.wbytes = 0;
  }

  static async create(gpu, tm) {
    const q = new QwenGpu(gpu, tm);
    // шейдеры компилируются в фоне, пока качаются и грузятся веса
    const pre = gpu.precompile ? gpu.precompile(q.pipelineList()).catch(() => {}) : null;
    await q._load();
    if (pre) await pre;
    return q;
  }
  pipelineList() {
    const mf = (this.tm.manifest.precision || {}), wt = mf.quant === "q8" ? [0, 1, 2, 3] : mf.quant === "q4" ? [0, 1, 2, 4] : [0, 1, 2];
    const hds = Array.from(new Set([this.tc.head_dim, this.pc.head_dim, this.cc.head_dim]));
    const L = [];
    for (const w of wt) L.push(["gemm", "main", { WTYPE: w }]);
    for (const hd of hds) L.push(["attn_kv", "main", { HD: hd }]);
    for (const e of ["main", "ip"]) L.push(["rms", e]);
    L.push(["rope2", "main"], ["cconv", "im2col"], ["cconv", "hist"], ["cconv", "dw"], ["trans", "main"], ["ln", "main"]);
    for (const e of ["swiglu", "add", "copy", "snake", "clampcopy", "silu_vec", "zero_rows"]) L.push(["ew", e]);
    return L;
  }

  // ---------- загрузка весов ----------
  async _raw(name) { return this.tm.rawAsync ? await this.tm.rawAsync(name) : this.tm.raw(name); }
  async _mi(name) { if (this.tm.prepare) await this.tm.prepare(name); return matInfo(this.tm, name); }
  async lin(name, bias = true) {
    const m = await this._mi(name);
    if (m.wtype >= 3 && (m.cols & 31)) fail("квантованная матрица " + name + ": число столбцов не кратно 32", "format");
    const g = this.gpu;
    const o = { rows: m.rows, cols: m.cols, wtype: m.wtype, W: g.upload(await this._raw(name), name), S: null, B: null, hasBias: 0 };
    if (m.wtype >= 3) o.S = g.upload(await this._raw(name + ".scale"), name + ".scale");
    const bn = name.slice(0, -6) + "bias";
    if (bias && this.tm.has(bn)) { o.B = g.upload(await this._raw(bn), bn); o.hasBias = 1; }
    this.wbytes += this.tm.info(name).nbytes;
    if (this.tm.release) this.tm.release(name);
    return o;
  }
  async vec(name) { this.wbytes += this.tm.info(name).nbytes; const b = this.gpu.upload(await this._raw(name), name); if (this.tm.release) this.tm.release(name); return b; }
  async vecF32(name) { const r = await this._raw(name); return new Float32Array(r.slice().buffer, 0, r.byteLength >> 2); }
  // таблица на CPU (эмбеддинги): строки читаются по требованию
  async cpuMat(name) { const m = await this._mi(name); m.raw = await this._raw(name); return m; }
  row(m, r, out) {
    if (r < 0 || r >= m.rows) fail("идентификатор " + r + " вне таблицы " + m.name, "arg");
    return matRow(this.tm, m, r, out);
  }

  _layerNames(p, c) {
    const n = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight"];
    if (c.qkNorm) n.push("self_attn.q_norm.weight", "self_attn.k_norm.weight");
    if (c.layerScale) n.push("self_attn_layer_scale.scale", "mlp_layer_scale.scale");
    return n.map((x) => p + x);
  }
  async xfmr(prefix, c, tag) {
    const T = { tag, cfg: c, layers: [], invFreq: new Float32Array(c.head_dim / 2) };
    for (let j = 0; j < c.head_dim / 2; j++) T.invFreq[j] = f32r(1 / Math.pow(c.theta, (2 * j) / c.head_dim));
    for (let i = 0; i < c.layers; i++) {
      const p = prefix + "layers." + i + ".", L = {};
      if (this.tm.warm) for (const j of [i, i + 1]) if (j < c.layers) this.tm.warm(this._layerNames(prefix + "layers." + j + ".", c));
      L.q = await this.lin(p + "self_attn.q_proj.weight"); L.k = await this.lin(p + "self_attn.k_proj.weight");
      L.v = await this.lin(p + "self_attn.v_proj.weight"); L.o = await this.lin(p + "self_attn.o_proj.weight");
      L.gate = await this.lin(p + "mlp.gate_proj.weight"); L.up = await this.lin(p + "mlp.up_proj.weight"); L.down = await this.lin(p + "mlp.down_proj.weight");
      L.ln1 = await this.vec(p + "input_layernorm.weight"); L.ln2 = await this.vec(p + "post_attention_layernorm.weight");
      if (c.qkNorm) { L.qn = await this.vec(p + "self_attn.q_norm.weight"); L.kn = await this.vec(p + "self_attn.k_norm.weight"); }
      if (c.layerScale) { L.as = await this.vec(p + "self_attn_layer_scale.scale"); L.ms = await this.vec(p + "mlp_layer_scale.scale"); }
      T.layers.push(L);
    }
    T.fn = await this.vec(prefix + "norm.weight");
    return T;
  }

  async _load() {
    const tc = this.tc, pc = this.pc, cc = this.cc;
    // таблицы и проекции talker
    this.codecEmb = await this.cpuMat("talker.model.codec_embedding.weight");
    this.textEmb = await this.cpuMat("talker.model.text_embedding.weight");
    this.codecHead = await this.lin("talker.codec_head.weight", false);
    this.tp1 = await this.lin("talker.text_projection.linear_fc1.weight");
    this.tp2 = await this.lin("talker.text_projection.linear_fc2.weight");
    this.talker = await this.xfmr("talker.model.", { layers: tc.layers, H: tc.hidden, nh: tc.heads, nkv: tc.kv_heads, head_dim: tc.head_dim, I: tc.inter, eps: tc.eps, theta: tc.theta, qkNorm: true, layerScale: false, window: 0 }, "t.");
    // code predictor
    this.pred = await this.xfmr("talker.code_predictor.model.", { layers: pc.layers, H: pc.hidden, nh: pc.heads, nkv: pc.kv_heads, head_dim: pc.head_dim, I: pc.inter, eps: pc.eps, theta: pc.theta, qkNorm: true, layerScale: false, window: 0 }, "p.");
    this.predEmb = []; this.predHead = [];
    for (let g = 0; g < tc.groups - 1; g++) {
      this.predEmb.push(await this.cpuMat("talker.code_predictor.model.codec_embedding." + g + ".weight"));
      this.predHead.push(await this.lin("talker.code_predictor.lm_head." + g + ".weight", false));
    }
    this.hasSmall = this.tm.has("talker.code_predictor.small_to_mtp_projection.weight");
    if (this.hasSmall) this.smallProj = await this.lin("talker.code_predictor.small_to_mtp_projection.weight");
    else if (pc.hidden !== tc.hidden) fail("скрытые размеры talker и predictor различаются, а small_to_mtp_projection нет", "format");
    await this._loadCodec();
  }

  async _snake(p) {
    const a = await this.vecF32(p + ".alpha"), b = await this.vecF32(p + ".beta");
    const A = new Float32Array(a.length), IB = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) { A[i] = f32r(Math.exp(a[i])); IB[i] = f32r(1 / f32r(f32r(Math.exp(b[i])) + f32r(1e-9))); }
    const g = this.gpu;
    return { A: g.upload(new Uint8Array(A.buffer), p + ".exp_alpha"), IB: g.upload(new Uint8Array(IB.buffer), p + ".inv_beta"), C: a.length };
  }
  async _conv(p, K, cin, cout, dil = 1) {
    const w = await this.lin(p + ".weight");
    if (w.rows !== cout || w.cols !== K * cin) fail("форма свёртки не совпала: " + p, "format");
    return { w, K, cin, cout, dil, ctx: (K - 1) * dil };
  }

  async _loadCodec() {
    const cc = this.cc, dim = cc.codebook_dim / 2;
    if (this.tm.warm) this.tm.warm(Array.from(this.tm.table.keys()).filter((n) => n.startsWith("codec.")));
    this.qFirst = [await this.cpuMat("codec.quantizer.rvq_first.vq.layers.0._codebook.embedding")];
    this.qRest = [];
    for (let i = 0; i < cc.quantizers - 1; i++) this.qRest.push(await this.cpuMat("codec.quantizer.rvq_rest.vq.layers." + i + "._codebook.embedding"));
    const fo = await this.cpuMat("codec.quantizer.rvq_first.output_proj.weight"), ro = await this.cpuMat("codec.quantizer.rvq_rest.output_proj.weight");
    if (this.qFirst[0].cols !== dim || fo.cols !== dim || fo.rows !== cc.codebook_dim) fail("форма квантизатора кодека не совпала", "format");
    this.qFirstOut = matFloat(this.tm, fo); this.qRestOut = matFloat(this.tm, ro);
    this.preConv = await this._conv("codec.pre_conv.conv", 3, cc.codebook_dim, cc.latent);
    this.ptIn = await this.lin("codec.pre_transformer.input_proj.weight");
    this.ptOut = await this.lin("codec.pre_transformer.output_proj.weight");
    this.pt = await this.xfmr("codec.pre_transformer.", { layers: cc.layers, H: cc.hidden, nh: cc.heads, nkv: cc.kv_heads, head_dim: cc.head_dim, I: cc.inter, eps: cc.eps, theta: cc.theta, qkNorm: false, layerScale: true, window: cc.window }, "c.");
    this.up = [];
    for (let s = 0; s < cc.up_ratios.length; s++) {
      const p = "codec.upsample." + s + ".", r = cc.up_ratios[s], L = cc.latent;
      const u = { s: r, tr: { w: await this.lin(p + "0.conv.weight", false), K: r, stride: r, cin: L, cout: L, tail: 0 }, trB: await this.vec(p + "0.conv.bias") };
      u.dwW = await this.vec(p + "1.dwconv.conv.weight"); u.dwB = await this.vec(p + "1.dwconv.conv.bias");
      u.lnW = await this.vec(p + "1.norm.weight"); u.lnB = await this.vec(p + "1.norm.bias"); u.gamma = await this.vec(p + "1.gamma");
      u.pw1 = await this.lin(p + "1.pwconv1.weight"); u.pw2 = await this.lin(p + "1.pwconv2.weight");
      if (u.tr.w.rows !== r * L || u.tr.w.cols !== L) fail("форма транспонированной свёртки апсэмплинга не совпала", "format");
      this.up.push(u);
    }
    this.conv0 = await this._conv("codec.decoder.0.conv", 7, cc.latent, cc.decoder_dim);
    this.blocks = [];
    for (let b = 0; b < cc.up_rates.length; b++) {
      const inD = cc.decoder_dim >> b, outD = cc.decoder_dim >> (b + 1), r = cc.up_rates[b], p = "codec.decoder." + (b + 1) + ".block.";
      const d = { act: await this._snake(p + "0"), tr: { w: await this.lin(p + "1.conv.weight", false), K: 2 * r, stride: r, cin: inD, cout: outD, tail: r }, trB: await this.vec(p + "1.conv.bias"), a1: [], c1: [], a2: [], c2: [] };
      if (d.tr.w.rows !== 2 * r * outD || d.tr.w.cols !== inD) fail("форма транспонированной свёртки декодера не совпала", "format");
      const dil = [1, 3, 9];
      for (let u = 0; u < 3; u++) {
        const q = p + (u + 2) + ".";
        d.a1.push(await this._snake(q + "act1")); d.c1.push(await this._conv(q + "conv1.conv", 7, outD, outD, dil[u]));
        d.a2.push(await this._snake(q + "act2")); d.c2.push(await this._conv(q + "conv2.conv", 1, outD, outD));
      }
      this.blocks.push(d);
    }
    const nb = cc.up_rates.length;
    this.finalAct = await this._snake("codec.decoder." + (nb + 1));
    this.finalConv = await this._conv("codec.decoder." + (nb + 2) + ".conv", 7, cc.decoder_dim >> nb, 1);
  }

  // ---------- примитивы ----------
  gemm(X, L, Y, M, o = {}) {
    const g = this.gpu, N = L.rows, K = L.cols;
    const pipe = g.pipeline("gemm", "main", { WTYPE: L.wtype });
    g.run(pipe, [X, L.W, L.S, L.B, Y, o.G || null],
      [M, N, K, o.epi || 0, o.xOff || 0, o.yOff || 0, o.gOff || 0, 0, o.ldx || K, o.ldy || N, L.hasBias, 0], Math.ceil(N / 64), Math.ceil(M / 64));
  }
  ew(entry, bufs, p) {
    const g = this.gpu, pipe = g.pipeline("ew", entry);
    g.run1d(pipe, bufs, [p.n, p.a || 0, p.b || 0, p.c || 0, p.d || 0, p.xOff || 0, p.yOff || 0, 0, f(p.f0 || 0), f(p.f1 || 0), 0, 0], p.n);
  }
  rms(X, W, Y, rows, C, eps, xOff = 0, yOff = 0) {
    const g = this.gpu;
    g.run(g.pipeline("rms", "main"), [X, W, Y], [rows, C, xOff, yOff, f(eps), 0, 0, 0], Math.min(rows, 32768), Math.ceil(rows / 32768));
  }
  rmsIp(Y, W, rows, C, eps, yOff = 0) {
    const g = this.gpu;
    g.run(g.pipeline("rms", "ip"), [null, W, Y], [rows, C, 0, yOff, f(eps), 0, 0, 0], Math.min(rows, 32768), Math.ceil(rows / 32768));
  }
  rope(X, CS, rows, heads, HD, off) {
    const g = this.gpu;
    g.run1d(g.pipeline("rope2", "main"), [X, CS], [rows, heads, HD, off, 0, 0, 0, 0], rows * heads * (HD >> 1));
  }

  newKv(T, cap) {
    const c = T.cfg, kd = c.nkv * c.head_dim, g = this.gpu;
    const kv = { k: [], v: [], len: 0, cap };
    for (let i = 0; i < c.layers; i++) { kv.k.push(g.buffer(cap * kd * 4, T.tag + "k" + i)); kv.v.push(g.buffer(cap * kd * 4, T.tag + "v" + i)); }
    return kv;
  }
  freeKv(kv) { if (!kv) return; for (const b of kv.k) this.gpu.free(b); for (const b of kv.v) this.gpu.free(b); kv.k = []; kv.v = []; }

  // Прогон m строк через слои трансформера. h — GPU-буфер [m][H] (вход и остаток), на месте. Должен быть внутри begin()…submit().
  xfmrForward(T, kv, h, m) {
    const g = this.gpu, c = T.cfg, pool = this.pool, tag = T.tag;
    const qd = c.nh * c.head_dim, kd = c.nkv * c.head_dim, hd = c.head_dim, half = hd >> 1, pos0 = kv.len;
    if (pos0 + m > kv.cap) fail("контекст длиннее KV-кэша (" + (pos0 + m) + " > " + kv.cap + ")", "limits");
    const xn = pool.get(tag + "xn", m * c.H), q = pool.get(tag + "q", m * qd), ao = pool.get(tag + "ao", m * qd);
    const gg = pool.get(tag + "g", m * c.I), uu = pool.get(tag + "u", m * c.I), act = pool.get(tag + "act", m * c.I);
    const cs = new Float32Array(m * half * 2);
    for (let i = 0; i < m; i++) for (let j = 0; j < half; j++) {
      const a = f32r(T.invFreq[j] * f32r(pos0 + i));
      cs[(i * half + j) * 2] = f32r(Math.cos(a)); cs[(i * half + j) * 2 + 1] = f32r(Math.sin(a));
    }
    const csb = pool.get(tag + "cs", m * half * 2);
    g.write(csb, cs);
    const scale = f32r(1 / f32r(Math.sqrt(hd)));
    const attn = g.pipeline("attn_kv", "main", { HD: hd });
    for (let li = 0; li < T.layers.length; li++) {
      const L = T.layers[li], K = kv.k[li], V = kv.v[li];
      this.rms(h, L.ln1, xn, m, c.H, c.eps);
      this.gemm(xn, L.q, q, m);
      this.gemm(xn, L.k, K, m, { yOff: pos0 * kd });
      this.gemm(xn, L.v, V, m, { yOff: pos0 * kd });
      if (c.qkNorm) { this.rmsIp(q, L.qn, m * c.nh, hd, c.eps); this.rmsIp(K, L.kn, m * c.nkv, hd, c.eps, pos0 * kd); }
      this.rope(q, csb, m, c.nh, hd, 0);
      this.rope(K, csb, m, c.nkv, hd, pos0 * kd);
      g.run(attn, [q, K, V, ao], [m, c.nh, c.nkv, pos0, c.window || 0, 0, 0, 0, f(scale), 0, 0, 0], m, c.nh);
      this.gemm(ao, L.o, h, m, c.layerScale ? { epi: 3, G: L.as } : { epi: 5 });
      this.rms(h, L.ln2, xn, m, c.H, c.eps);
      this.gemm(xn, L.gate, gg, m);
      this.gemm(xn, L.up, uu, m);
      this.ew("swiglu", [gg, uu, null, act], { n: m * c.I });
      this.gemm(act, L.down, h, m, c.layerScale ? { epi: 3, G: L.ms } : { epi: 5 });
    }
    kv.len += m;
  }

  // ================= talker / predictor: исполнение =================
  // Строки-эмбеддинги → текстовая проекция (fc1 + SiLU + fc2) на GPU. ids: массив токенов → Float32Array [n][hidden].
  async textProject(ids) {
    const tc = this.tc, g = this.gpu, n = ids.length, th = tc.text_hidden, D = tc.hidden;
    const e = new Float32Array(n * th), row = new Float32Array(th);
    for (let i = 0; i < n; i++) { this.row(this.textEmb, ids[i], row); e.set(row, i * th); }
    const X = this.pool.get("tp.x", n * th), H = this.pool.get("tp.h", n * th), Y = this.pool.get("tp.y", n * D);
    g.write(X, e);
    g.begin();
    this.gemm(X, this.tp1, H, n, { epi: 4 });
    this.gemm(H, this.tp2, Y, n);
    g.submit();
    return await g.read(Y, n * D);
  }
  codecRow(id, out) { return this.row(this.codecEmb, id, out); }

  newTalkerState(cap) {
    const D = this.tc.hidden, g = this.gpu;
    const st = { kv: this.newKv(this.talker, cap), hid: g.buffer(D * 4, "t.hid"), pkv: this.newKv(this.pred, this.tc.groups + 2), free: () => { this.freeKv(st.kv); this.freeKv(st.pkv); g.free(st.hid); } };
    return st;
  }

  // Один проход talker. x: Float32Array [m][D] (m строк подряд — префилл или одна строка). Возвращает логиты codec_head последней
  // позиции; скрытое состояние после финальной нормы остаётся в st.hid (для predictor).
  async talkerStep(st, x, m) {
    const g = this.gpu, tc = this.tc, D = tc.hidden;
    const h = this.pool.get("t.h", m * D), lg = this.pool.get("t.logits", tc.vocab);
    g.write(h, x);
    g.begin();
    this.xfmrForward(this.talker, st.kv, h, m);
    this.rms(h, this.talker.fn, st.hid, 1, D, tc.eps, (m - 1) * D, 0);
    this.gemm(st.hid, this.codecHead, lg, 1);
    g.submit();
    return await g.read(lg, tc.vocab);
  }

  // Проход predictor для кодбука cb (1..groups-1). cb=1: вход = [скрытое talker; эмбеддинг кода 0]; дальше — эмбеддинг предыдущего кода.
  async predictorLogits(st, cb, code) {
    const g = this.gpu, tc = this.tc, pc = this.pc, D = tc.hidden, PH = pc.hidden;
    const m = cb === 1 ? 2 : 1;
    const a = this.pool.get("p.a", m * D), h = this.pool.get("p.h", m * PH), ph = this.pool.get("p.ph", PH), lg = this.pool.get("p.logits", pc.vocab);
    const row = new Float32Array(D);
    if (cb === 1) {
      this.codecRow(code, row);
      g.write(a, row, D);
      st.pkv.len = 0;
    } else {
      this.row(this.predEmb[cb - 2], code, row);
      g.write(a, row, 0);
    }
    g.begin();
    if (cb === 1) this.ew("copy", [st.hid, null, null, a], { n: D });
    if (this.hasSmall) this.gemm(a, this.smallProj, h, m);
    else this.ew("copy", [a, null, null, h], { n: m * D });
    this.xfmrForward(this.pred, st.pkv, h, m);
    this.rms(h, this.pred.fn, ph, 1, PH, pc.eps, (m - 1) * PH, 0);
    this.gemm(ph, this.predHead[cb - 1], lg, 1);
    g.submit();
    return await g.read(lg, pc.vocab);
  }

  // ================= декодер кодека =================
  newCodecState(cap) {
    const cc = this.cc, g = this.gpu, L = cc.latent;
    const st = { frames: 0, preH: new QHist(g, 2 * cc.codebook_dim, "pre"), kv: this.newKv(this.pt, cap), upH: [], c0H: new QHist(g, 6 * L, "c0"), blk: [], finH: null };
    for (const u of this.up) st.upH.push(new QHist(g, 6 * L, "updw"));
    for (const d of this.blocks) {
      const co = d.tr.cout;
      st.blk.push({ tail: new QHist(g, d.tr.tail * co, "tail"), ru: [new QHist(g, 6 * co, "ru0"), new QHist(g, 18 * co, "ru1"), new QHist(g, 54 * co, "ru2")] });
    }
    st.finH = new QHist(g, 6 * (cc.decoder_dim >> this.blocks.length), "fin");
    return st;
  }
  freeCodecState(st) {
    if (!st) return;
    st.preH.free(); this.freeKv(st.kv); st.upH.forEach((h) => h.free()); st.c0H.free(); st.finH.free();
    for (const b of st.blk) { b.tail.free(); b.ru.forEach((h) => h.free()); }
  }
  // Глубокая копия состояния (для кэша прогрева голоса)
  cloneCodecState(src, cap) {
    const g = this.gpu, dst = this.newCodecState(cap);
    if (g._flushSync) g._flushSync();
    const enc = g.device.createCommandEncoder();
    dst.preH.copyFrom(src.preH, enc); dst.c0H.copyFrom(src.c0H, enc); dst.finH.copyFrom(src.finH, enc);
    src.upH.forEach((h, i) => dst.upH[i].copyFrom(h, enc));
    src.blk.forEach((b, i) => { dst.blk[i].tail.copyFrom(b.tail, enc); b.ru.forEach((h, j) => dst.blk[i].ru[j].copyFrom(h, enc)); });
    const kd = this.pt.cfg.nkv * this.pt.cfg.head_dim, bytes = Math.max(16, src.kv.len * kd * 4);
    for (let i = 0; i < src.kv.k.length; i++) {
      if (src.kv.len) { enc.copyBufferToBuffer(src.kv.k[i], 0, dst.kv.k[i], 0, bytes); enc.copyBufferToBuffer(src.kv.v[i], 0, dst.kv.v[i], 0, bytes); }
    }
    g.device.queue.submit([enc.finish()]);
    dst.kv.len = src.kv.len; dst.frames = src.frames;
    return dst;
  }

  // свёртка с историей: Y = conv(ext([hist; X])) ; K == 1 — просто gemm
  _conv1(Cv, hist, X, T, Y, xOff = 0) {
    const g = this.gpu;
    if (Cv.K === 1) { this.gemm(X, Cv.w, Y, T, { xOff }); return; }
    const col = this.pool.get("cc.col", this._colMax);
    g.run1d(g.pipeline("cconv", "im2col"), [X, hist.cur, null, null, col], [T, Cv.cin, Cv.K, Cv.dil, Cv.ctx, xOff, 0, 0], T * Cv.K * Cv.cin);
    this.gemm(col, Cv.w, Y, T);
    g.run1d(g.pipeline("cconv", "hist"), [X, hist.cur, null, null, hist.next], [T, Cv.cin, Cv.K, Cv.dil, Cv.ctx, xOff, 0, 0], Cv.ctx * Cv.cin);
    hist.swap();
  }
  _trans(Tr, trB, tail, X, T, Y) {
    const g = this.gpu;
    const tmp = this.pool.get("cc.tmp", this._tmpMax);
    this.gemm(X, Tr.w, tmp, T);
    const n = T * Tr.stride + Tr.tail;
    g.run1d(g.pipeline("trans", "main"), [tmp, tail ? tail.cur : null, trB, Y, tail ? tail.next : null], [T, Tr.K, Tr.stride, Tr.cout, Tr.tail, 0, 0, 0], n * Tr.cout);
    if (tail) tail.swap();
  }
  _snakeRun(Sn, X, Y, n) { this.ew("snake", [X, Sn.A, Sn.IB, Y], { n, a: Sn.C }); }

  // codes: Int16Array [T][Q]. Возвращает Float32Array PCM (T * samples_per_frame). Состояние st продвигается.
  async codecForward(st, codes, T) {
    const cc = this.cc, g = this.gpu, pool = this.pool, Q = cc.quantizers, dim = cc.codebook_dim / 2, cb = cc.codebook_dim, L = cc.latent;
    if (T > QCHUNK_MAX_FRAMES) fail("слишком много кадров за один вызов декодера", "internal");
    // 1) квантизатор на CPU: сумма эмбеддингов по кодбукам → output_proj
    const s1 = new Float32Array(T * dim), s2 = new Float32Array(T * dim), tmpr = new Float32Array(dim);
    for (let t = 0; t < T; t++) {
      for (let q = 0; q < Q; q++) {
        const c = codes[t * Q + q];
        if (c < 0 || c >= cc.codebook_size) fail("код кодека вне диапазона", "arg");
        if (q === 0) this.row(this.qFirst[0], c, s1.subarray(t * dim, (t + 1) * dim));
        else { this.row(this.qRest[q - 1], c, tmpr); const o = t * dim; for (let i = 0; i < dim; i++) s2[o + i] += tmpr[i]; }
      }
    }
    const z = new Float32Array(T * cb);
    for (let t = 0; t < T; t++) for (let o = 0; o < cb; o++) {
      let a = 0, b = 0;
      const wf = o * dim;
      for (let i = 0; i < dim; i++) { a += this.qFirstOut[wf + i] * s1[t * dim + i]; b += this.qRestOut[wf + i] * s2[t * dim + i]; }
      z[t * cb + o] = f32r(f32r(a) + f32r(b));
    }
    // размеры рабочих буферов (одинаковы на протяжении записи)
    let len = T, maxCol = T * this.preConv.K * cb, maxTmp = 4;
    for (const u of this.up) { maxTmp = Math.max(maxTmp, len * u.tr.K * L); len *= u.s; }
    maxCol = Math.max(maxCol, len * this.conv0.K * L);
    for (const d of this.blocks) {
      maxTmp = Math.max(maxTmp, len * d.tr.K * d.tr.cout);
      len *= d.tr.stride;
      maxCol = Math.max(maxCol, len * 7 * d.tr.cout);
    }
    maxCol = Math.max(maxCol, len * this.finalConv.K * this.finalConv.cin);
    this._colMax = maxCol; this._tmpMax = maxTmp;
    const total = len;
    const zb = pool.get("cc.z", T * cb);
    g.write(zb, z);
    g.begin();
    // 2) pre_conv → трансформер
    const lat = pool.get("cc.lat", T * L), hin = pool.get("cc.hin", T * cc.hidden), lat2 = pool.get("cc.lat2", T * L);
    this._conv1(this.preConv, st.preH, zb, T, lat);
    this.gemm(lat, this.ptIn, hin, T);
    this.xfmrForward(this.pt, st.kv, hin, T);
    const hn = pool.get("cc.hn", T * cc.hidden);
    this.rms(hin, this.pt.fn, hn, T, cc.hidden, cc.eps);
    this.gemm(hn, this.ptOut, lat2, T);
    // 3) апсэмплинг: транспонированная свёртка + ConvNeXt
    let cur = lat2, n = T, flip = 0;
    for (let s = 0; s < this.up.length; s++) {
      const u = this.up[s], nn = n * u.s;
      const y = pool.get("cc.y" + s, nn * L), dwo = pool.get("cc.dw" + s, nn * L), nrm = pool.get("cc.nrm" + s, nn * L), h1 = pool.get("cc.h1" + s, nn * 4 * L);
      this._trans(u.tr, u.trB, null, cur, n, y);
      g.run1d(g.pipeline("cconv", "dw"), [y, st.upH[s].cur, u.dwW, u.dwB, dwo], [nn, L, 7, 1, 6, 0, 0, 1], nn * L);
      g.run1d(g.pipeline("cconv", "hist"), [y, st.upH[s].cur, null, null, st.upH[s].next], [nn, L, 7, 1, 6, 0, 0, 0], 6 * L);
      st.upH[s].swap();
      g.run(g.pipeline("ln", "main"), [dwo, u.lnW, u.lnB, null, nrm], [nn, L, 0, 0, 0, 0, 0, 0, f(1e-6), 0, 0, 0], Math.min(nn, 32768), Math.ceil(nn / 32768));
      this.gemm(nrm, u.pw1, h1, nn, { epi: 2 });
      this.gemm(h1, u.pw2, y, nn, { epi: 3, G: u.gamma });
      cur = y; n = nn;
    }
    // 4) декодер
    let x = pool.get("cc.x0", n * cc.decoder_dim);
    this._conv1(this.conv0, st.c0H, cur, n, x);
    let c = cc.decoder_dim;
    for (let b = 0; b < this.blocks.length; b++) {
      const d = this.blocks[b], out = d.tr.cout, nn = n * d.tr.stride;
      const xs = pool.get("cc.xs" + b, n * c), y = pool.get("cc.by" + b, nn * out);
      this._snakeRun(d.act, x, xs, n * c);
      this._trans(d.tr, d.trB, st.blk[b].tail, xs, n, y);
      n = nn; c = out;
      const t1 = pool.get("cc.t1" + b, n * c), t2 = pool.get("cc.t2" + b, n * c);
      for (let u = 0; u < 3; u++) {
        this._snakeRun(d.a1[u], y, t1, n * c);
        this._conv1(d.c1[u], st.blk[b].ru[u], t1, n, t2);
        this._snakeRun(d.a2[u], t2, t1, n * c);
        this._conv1(d.c2[u], null, t1, n, t2);
        this.ew("add", [t2, null, null, y], { n: n * c });
      }
      x = y;
    }
    const xs = pool.get("cc.xsF", n * c), wav = pool.get("cc.wav", n), wc = pool.get("cc.wavc", n);
    this._snakeRun(this.finalAct, x, xs, n * c);
    this._conv1(this.finalConv, st.finH, xs, n, wav);
    this.ew("clampcopy", [wav, null, null, wc], { n });
    g.submit();
    st.frames += T;
    if (n !== total) fail("внутренняя ошибка: длина выхода декодера", "internal");
    return await g.read(wc, n);
  }

  dispose() {
    this.pool.clear();
  }
}

// ============================================================================
// Qwen3-TTS: создание голоса из записи прямо в браузере (порт src/qwen3_voice.cpp): mel → ECAPA-TDNN (вектор диктора) и
// энкодер Mimi (SEANet + трансформер + RVQ) → коды образца. Считается на CPU небольшими порциями с уступкой потока интерфейсу
// (страница не замирает). Активации — по времени: [T][C]; свёртки — im2col + умножение матриц.
// ============================================================================
const _macro = typeof MessageChannel !== "undefined" ? (() => {
  const ch = new MessageChannel(); const q = [];
  if (ch.port1.unref) ch.port1.unref();
  ch.port1.onmessage = () => { const f = q.shift(); if (f) f(); };
  return () => new Promise((r) => { q.push(r); ch.port2.postMessage(0); });
})() : () => new Promise((r) => setTimeout(r, 0));

// Кооперативная уступка потока: вызывайте tick() между порциями работы; реально уступает не чаще раза в sliceMs.
class Yielder {
  constructor(sliceMs = 24, onTick) { this.slice = sliceMs; this.t = (typeof performance !== "undefined" ? performance.now() : Date.now()); this.onTick = onTick; }
  async tick() {
    const now = performance.now();
    if (now - this.t >= this.slice) { if (this.onTick) this.onTick(); await _macro(); this.t = performance.now(); }
  }
}

// W: { data: Float32Array [rows][cols], rows, cols, bias? }. Y[t][r] = dot(W[r], X[t]) + bias[r]
async function vMatmul(w, X, T, Y, y, xOff = 0) {
  const R = w.rows, C = w.cols, W = w.data, B = w.bias;
  const per = Math.max(1, Math.floor(3e7 / Math.max(1, R * C)));
  for (let t0 = 0; t0 < T; t0 += per) {
    const t1 = Math.min(T, t0 + per);
    for (let t = t0; t < t1; t++) {
      const xo = xOff + t * C, yo = t * R;
      let r = 0;
      for (; r + 3 < R; r += 4) {
        let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        const w0 = r * C, w1 = w0 + C, w2 = w1 + C, w3 = w2 + C;
        for (let c = 0; c < C; c++) { const x = X[xo + c]; a0 += W[w0 + c] * x; a1 += W[w1 + c] * x; a2 += W[w2 + c] * x; a3 += W[w3 + c] * x; }
        Y[yo + r] = a0 + (B ? B[r] : 0); Y[yo + r + 1] = a1 + (B ? B[r + 1] : 0); Y[yo + r + 2] = a2 + (B ? B[r + 2] : 0); Y[yo + r + 3] = a3 + (B ? B[r + 3] : 0);
      }
      for (; r < R; r++) { let a = 0; const w0 = r * C; for (let c = 0; c < C; c++) a += W[w0 + c] * X[xo + c]; Y[yo + r] = a + (B ? B[r] : 0); }
    }
    await y.tick();
  }
  return Y;
}
const vMM = async (w, X, T, y) => vMatmul(w, X, T, new Float32Array(T * w.rows), y);

// Conv1d над x [T][cin] с явными отступами слева/справа; режим: "zero" | "reflect" | "replicate"
async function vConv1d(c, x, T, L, R, mode, y) {
  const keff = (c.K - 1) * c.dil + 1, Tp = T + L + R;
  const Tout = Tp >= keff ? Math.floor((Tp - keff) / c.stride) + 1 : 0;
  const out = new Float32Array(Math.max(Tout, 0) * c.cout);
  if (Tout <= 0) return { y: out, T: 0 };
  const cin = c.cin, KC = c.K * cin;
  if (c.K === 1 && c.stride === 1 && L === 0 && R === 0) return { y: await vMatmul(c.w, x, T, out, y), T: Tout };
  const src = (i) => {   // индекс строки входа или -1 (нулевая строка)
    const j = i - L;
    if (j >= 0 && j < T) return j;
    if (mode === "zero") return -1;
    if (mode === "replicate") return Math.min(Math.max(j, 0), T - 1);
    if (T === 1) return 0;
    const period = 2 * (T - 1);
    let m = j % period; if (m < 0) m += period; if (m >= T) m = period - m;
    return m;
  };
  const block = Math.max(1, Math.min(Tout, Math.floor(3e6 / Math.max(1, KC))));
  const col = new Float32Array(block * KC);
  const sub = { data: c.w.data, rows: c.cout, cols: KC, bias: c.w.bias };
  for (let t0 = 0; t0 < Tout; t0 += block) {
    const nb = Math.min(block, Tout - t0);
    for (let t = 0; t < nb; t++) for (let k = 0; k < c.K; k++) {
      const s = src((t0 + t) * c.stride + k * c.dil), d = t * KC + k * cin;
      if (s >= 0) col.set(x.subarray(s * cin, s * cin + cin), d); else col.fill(0, d, d + cin);
    }
    const tmp = new Float32Array(nb * c.cout);
    await vMatmul(sub, col, nb, tmp, y);
    out.set(tmp, t0 * c.cout);
  }
  return { y: out, T: Tout };
}
// PyTorch Conv1d(padding="same", padding_mode="reflect"): всего dil*(K-1), слева половина
async function vConvSame(c, x, T, y) {
  const total = c.dil * (c.K - 1), L = Math.floor(total / 2), R = total - L;
  return (await vConv1d(c, x, T, L, R, "reflect", y)).y;
}
// MimiConv1d (причинная): слева keff - stride, справа добавка до целого последнего окна
async function vConvMimi(c, x, T, mode, y) {
  const keff = (c.K - 1) * c.dil + 1, s = c.stride, pt = keff - s;
  const xlen = T - keff + pt;
  const nfr = xlen >= 0 ? Math.floor((xlen + s - 1) / s) : -Math.floor((-xlen) / s);
  const ideal = nfr * s + keff - pt, extra = Math.max(0, ideal - T);
  return vConv1d(c, x, T, pt, extra, mode, y);
}
const vRelu = (v) => { for (let i = 0; i < v.length; i++) if (v[i] < 0) v[i] = 0; return v; };
const vElu = (v) => { for (let i = 0; i < v.length; i++) if (v[i] <= 0) v[i] = Math.expm1(v[i]); return v; };

function vErf(x) {   // erf с точностью ~1e-7 (Abramowitz-Stegun 7.1.26 слишком грубая; берём разложение W. J. Cody через erfc)
  const ax = Math.abs(x);
  let r;
  if (ax < 0.5) {   // ряд Тейлора
    let term = ax, sum = ax; const x2 = ax * ax;
    for (let n = 1; n < 30; n++) { term *= -x2 / n; sum += term / (2 * n + 1); }
    r = 2 / Math.sqrt(Math.PI) * sum;
  } else {   // цепная дробь для erfc
    const t = 1 / (1 + 0.5 * ax);
    const y = t * Math.exp(-ax * ax - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    r = 1 - y;
  }
  return x < 0 ? -r : r;
}

class QwenVoiceEncoder {
  constructor(tm) { this.tm = tm; this.names = []; this.hasSpk = false; this.hasEnc = false; }
  static async create(tm, yielder) {
    const e = new QwenVoiceEncoder(tm);
    const q3 = tm.manifest.qwen3 || {};
    e.y = yielder || new Yielder();
    if (q3.speaker_encoder && tm.has("spk.mel_basis")) await e._loadSpk(q3.speaker_encoder);
    if (q3.encoder && tm.has("enc.encoder.layers.0.conv.weight")) await e._loadEnc(q3.encoder);
    return e;
  }
  async _f32(name) {
    this.names.push(name);
    if (this.tm.prepare) { await this.tm.prepare(name); return this.tm.f32(name); }
    return this.tm.f32(name);
  }
  async _mat(name) {
    const info = this.tm.info(name);
    if (info.dtype !== "F32") fail("веса энкодера голоса должны быть F32: " + name, "format");
    return { data: await this._f32(name), rows: info.shape[0], cols: info.shape.length > 1 ? info.shape[1] : 1 };
  }
  async _conv(p, K, dil = 1, stride = 1, bias = true) {
    const w = await this._mat(p + ".weight");
    if (bias && this.tm.has(p + ".bias")) w.bias = await this._f32(p + ".bias");
    if (w.cols % K) fail("форма свёртки не совпала: " + p, "format");
    return { w, K, dil, stride, cout: w.rows, cin: w.cols / K };
  }
  release() { if (this.tm.release) for (const n of this.names) this.tm.release(n); this.names = []; }

  async _loadSpk(c) {
    this.melBasis = await this._f32("spk.mel_basis");
    this.nMels = c.mel_dim != null ? c.mel_dim : 128;
    const ch = c.enc_channels, ks = c.enc_kernel_sizes, ds = c.enc_dilations, scale = c.enc_res2net_scale != null ? c.enc_res2net_scale : 8;
    this.blk0 = await this._conv("spk.blocks.0.conv", ks[0], ds[0]);
    this.blocks = [];
    for (let i = 1; i + 1 < ch.length; i++) {
      const p = "spk.blocks." + i + ".";
      const b = { scale, res: [] };
      b.tdnn1 = await this._conv(p + "tdnn1.conv", 1); b.tdnn2 = await this._conv(p + "tdnn2.conv", 1);
      b.se1 = await this._conv(p + "se_block.conv1", 1); b.se2 = await this._conv(p + "se_block.conv2", 1);
      for (let j = 0; j < scale - 1; j++) b.res.push(await this._conv(p + "res2net_block.blocks." + j + ".conv", ks[i], ds[i]));
      this.blocks.push(b);
    }
    this.mfa = await this._conv("spk.mfa.conv", ks[ks.length - 1], ds[ds.length - 1]);
    this.aspTdnn = await this._conv("spk.asp.tdnn.conv", 1);
    this.aspConv = await this._conv("spk.asp.conv", 1);
    this.fc = await this._conv("spk.fc", 1);
    this.hasSpk = true;
  }

  async _loadEnc(c) {
    const ratios = c.upsampling_ratios, nres = c.num_residual_layers != null ? c.num_residual_layers : 1, growth = c.dilation_growth_rate != null ? c.dilation_growth_rate : 2;
    const K = c.kernel_size != null ? c.kernel_size : 7, lastK = c.last_kernel_size != null ? c.last_kernel_size : 3, rk = c.residual_kernel_size != null ? c.residual_kernel_size : 3;
    if (c.use_causal_conv === false) fail("энкодер Mimi: поддерживаются только причинные свёртки", "unsupported");
    if (c.use_conv_shortcut) fail("энкодер Mimi: conv shortcut не поддерживается", "unsupported");
    if ((c.pad_mode || "constant") !== "constant") fail("энкодер Mimi: pad_mode должен быть constant", "unsupported");
    if ((c.hidden_act || "gelu") !== "gelu") fail("энкодер Mimi: hidden_act должен быть gelu", "unsupported");
    this.conv0 = await this._conv("enc.encoder.layers.0.conv", K);
    let idx = 1;
    this.stages = [];
    for (let ri = ratios.length - 1; ri >= 0; ri--) {
      const st = { res: [] };
      for (let j = 0; j < nres; j++) {
        const p = "enc.encoder.layers." + idx + ".block.";
        st.res.push({ c1: await this._conv(p + "1.conv", rk, Math.pow(growth, j)), c2: await this._conv(p + "3.conv", 1) });
        idx++;
      }
      idx++;   // ELU
      st.down = await this._conv("enc.encoder.layers." + idx + ".conv", 2 * ratios[ri], 1, ratios[ri]);
      idx++;
      this.stages.push(st);
    }
    idx++;   // ELU
    this.convLast = await this._conv("enc.encoder.layers." + idx + ".conv", lastK);
    this.H = c.hidden_size; this.heads = c.num_attention_heads; this.kvHeads = c.num_key_value_heads != null ? c.num_key_value_heads : this.heads;
    this.hd = c.head_dim || this.H / this.heads; this.normEps = c.norm_eps != null ? c.norm_eps : 1e-5;
    this.invFreq = new Float32Array(this.hd / 2);
    for (let i = 0; i < this.hd / 2; i++) this.invFreq[i] = f32r(1 / Math.pow(c.rope_theta != null ? c.rope_theta : 10000, (2 * i) / this.hd));
    this.xl = [];
    for (let i = 0; i < c.num_hidden_layers; i++) {
      const p = "enc.encoder_transformer.layers." + i + ".";
      this.xl.push({
        ln1w: await this._f32(p + "input_layernorm.weight"), ln1b: await this._f32(p + "input_layernorm.bias"),
        ln2w: await this._f32(p + "post_attention_layernorm.weight"), ln2b: await this._f32(p + "post_attention_layernorm.bias"),
        sAttn: await this._f32(p + "self_attn_layer_scale.scale"), sMlp: await this._f32(p + "mlp_layer_scale.scale"),
        q: await this._mat(p + "self_attn.q_proj.weight"), k: await this._mat(p + "self_attn.k_proj.weight"), v: await this._mat(p + "self_attn.v_proj.weight"), o: await this._mat(p + "self_attn.o_proj.weight"),
        fc1: await this._mat(p + "mlp.fc1.weight"), fc2: await this._mat(p + "mlp.fc2.weight"),
      });
    }
    let prod = 1; for (const r of ratios) prod *= r;
    const encodecFr = (c.sampling_rate != null ? c.sampling_rate : 24000) / prod;
    const dk = 2 * Math.trunc(encodecFr / (c.frame_rate != null ? c.frame_rate : 12.5));
    this.downsample = await this._conv("enc.downsample.conv", dk, 1, 2, false);
    const sq = "enc.quantizer.semantic_residual_vector_quantizer.", aq = "enc.quantizer.acoustic_residual_vector_quantizer.";
    this.semIn = this.tm.has(sq + "input_proj.weight") ? await this._mat(sq + "input_proj.weight") : null;
    this.acIn = this.tm.has(aq + "input_proj.weight") ? await this._mat(aq + "input_proj.weight") : null;
    const nsem = c.num_semantic_quantizers != null ? c.num_semantic_quantizers : 1, nq = c.num_quantizers;
    this.semCb = []; this.acCb = [];
    for (let i = 0; i < nsem; i++) this.semCb.push(await this._mat(sq + "layers." + i + ".codebook.embed"));
    for (let i = 0; i < nq - nsem; i++) this.acCb.push(await this._mat(aq + "layers." + i + ".codebook.embed"));
    this.validQ = c.valid_quantizers != null ? c.valid_quantizers : 16;
    this.downRate = c.downsample_rate != null ? c.downsample_rate : 1920;
    if (this.validQ < nsem || this.validQ > nq) fail("энкодер Mimi: неверное valid_quantizers", "format");
    this.hasEnc = true;
  }

  // ---------- mel (n_fft 1024, hop 256, центрирование отступами, reflect) ----------
  async mel(pcm) {
    const nfft = 1024, hop = 256, nb = nfft / 2 + 1, pad = (nfft - hop) >> 1, N = pcm.length, nm = this.nMels;
    if (N <= pad) fail("запись слишком короткая", "arg");
    const yp = new Float64Array(N + 2 * pad);
    for (let i = 0; i < N + 2 * pad; i++) { let j = i - pad; if (j < 0) j = -j; if (j >= N) j = 2 * (N - 1) - j; yp[i] = pcm[j]; }
    const frames = Math.floor((yp.length - nfft) / hop) + 1;
    const win = new Float64Array(nfft);
    for (let i = 0; i < nfft; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / nfft);
    const fft = new FFT(nfft), re = new Float64Array(nfft), im = new Float64Array(nfft), mag = new Float64Array(nb);
    const out = new Float32Array(frames * nm), B = this.melBasis;
    for (let t = 0; t < frames; t++) {
      for (let i = 0; i < nfft; i++) { re[i] = yp[t * hop + i] * win[i]; im[i] = 0; }
      fft.run(re, im, -1);
      for (let k = 0; k < nb; k++) mag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k] + 1e-9);
      for (let m = 0; m < nm; m++) { let s = 0; const o = m * nb; for (let k = 0; k < nb; k++) s += B[o + k] * mag[k]; out[t * nm + m] = Math.log(Math.max(s, 1e-5)); }
      if ((t & 15) === 0) await this.y.tick();
    }
    return { mel: out, frames };
  }

  async _se(b, x, T) {
    const y = this.y;
    let h = vRelu(await vConvSame(b.tdnn1, x, T, y));
    const C = b.tdnn1.cout, cs = C / b.scale, outp = new Float32Array(T * C);
    let part = new Float32Array(T * cs), prev = null;
    for (let i = 0; i < b.scale; i++) {
      for (let t = 0; t < T; t++) part.set(h.subarray(t * C + i * cs, t * C + i * cs + cs), t * cs);
      let o;
      if (i === 0) o = part.slice();
      else {
        if (i >= 2) for (let k = 0; k < part.length; k++) part[k] += prev[k];
        o = vRelu(await vConvSame(b.res[i - 1], part, T, y));
      }
      for (let t = 0; t < T; t++) outp.set(o.subarray(t * cs, t * cs + cs), t * C + i * cs);
      prev = o;
    }
    const h2 = vRelu(await vConvSame(b.tdnn2, outp, T, y));
    const mean = new Float32Array(C);
    for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) mean[c] += h2[t * C + c];
    for (let c = 0; c < C; c++) mean[c] /= T;
    const s1 = vRelu(await vMM(b.se1.w, mean, 1, y)), s2 = await vMM(b.se2.w, s1, 1, y);
    for (let c = 0; c < s2.length; c++) s2[c] = 1 / (1 + Math.exp(-s2[c]));
    for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) h2[t * C + c] = h2[t * C + c] * s2[c] + x[t * C + c];
    return h2;
  }

  // Вектор диктора по записи 24 кГц (Float32Array)
  async speakerEmbedding(pcm) {
    if (!this.hasSpk) fail("в этой модели нет энкодера диктора: соберите модель из Base-чекпойнта", "unsupported");
    const y = this.y;
    const { mel, frames: T } = await this.mel(pcm);
    let h = vRelu(await vConvSame(this.blk0, mel, T, y));
    const outs = [];
    for (const b of this.blocks) { h = await this._se(b, h, T); outs.push(h); }
    let Ccat = 0; for (const b of this.blocks) Ccat += b.tdnn2.cout;
    const cat = new Float32Array(T * Ccat);
    let off = 0;
    for (let i = 0; i < outs.length; i++) { const C = this.blocks[i].tdnn2.cout; for (let t = 0; t < T; t++) cat.set(outs[i].subarray(t * C, t * C + C), t * Ccat + off); off += C; }
    const m = vRelu(await vConvSame(this.mfa, cat, T, y)), C = this.mfa.cout;
    const stats = (w) => {
      const mu = new Float64Array(C), sd = new Float64Array(C);
      for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) mu[c] += (w ? w[t * C + c] : 1 / T) * m[t * C + c];
      for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) { const d = m[t * C + c] - mu[c]; sd[c] += (w ? w[t * C + c] : 1 / T) * d * d; }
      for (let c = 0; c < C; c++) sd[c] = Math.sqrt(Math.max(sd[c], 1e-12));
      return [mu, sd];
    };
    let [mu, sd] = stats(null);
    const attIn = new Float32Array(T * 3 * C);
    for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) { attIn[t * 3 * C + c] = m[t * C + c]; attIn[t * 3 * C + C + c] = mu[c]; attIn[t * 3 * C + 2 * C + c] = sd[c]; }
    const a1 = await vConvSame(this.aspTdnn, attIn, T, y);
    vRelu(a1); for (let i = 0; i < a1.length; i++) a1[i] = Math.tanh(a1[i]);
    const att = await vConvSame(this.aspConv, a1, T, y);
    for (let c = 0; c < C; c++) {
      let mx = -Infinity;
      for (let t = 0; t < T; t++) mx = Math.max(mx, att[t * C + c]);
      let s = 0;
      for (let t = 0; t < T; t++) { const e = Math.exp(att[t * C + c] - mx); att[t * C + c] = e; s += e; }
      for (let t = 0; t < T; t++) att[t * C + c] = att[t * C + c] / s;
    }
    [mu, sd] = stats(att);
    const pooled = new Float32Array(2 * C);
    for (let c = 0; c < C; c++) { pooled[c] = mu[c]; pooled[C + c] = sd[c]; }
    return vMM(this.fc.w, pooled, 1, y);
  }

  async _transformer(x, T) {
    const y = this.y, H = this.H, heads = this.heads, kvh = this.kvHeads, hd = this.hd, qd = heads * hd, kd = kvh * hd, rep = heads / kvh, half = hd >> 1;
    const scl = f32r(1 / Math.sqrt(f32r(hd)));
    const h = new Float32Array(T * H), att = new Float32Array(T * qd);
    const cs = new Float32Array(T * half), sn = new Float32Array(T * half);
    for (let t = 0; t < T; t++) for (let i = 0; i < half; i++) { const f = f32r(this.invFreq[i] * t); cs[t * half + i] = Math.cos(f); sn[t * half + i] = Math.sin(f); }
    const rope = (v, nh, t) => { for (let hh = 0; hh < nh; hh++) { const o = t * nh * hd + hh * hd; for (let i = 0; i < half; i++) { const a = v[o + i], b = v[o + i + half]; v[o + i] = a * cs[t * half + i] - b * sn[t * half + i]; v[o + i + half] = b * cs[t * half + i] + a * sn[t * half + i]; } } };
    for (const l of this.xl) {
      for (let t = 0; t < T; t++) vLayerNorm(x, t * H, l.ln1w, l.ln1b, H, this.normEps, h, t * H);
      const q = await vMM(l.q, h, T, y), k = await vMM(l.k, h, T, y), v = await vMM(l.v, h, T, y);
      for (let t = 0; t < T; t++) { rope(q, heads, t); rope(k, kvh, t); }
      const sc = new Float32Array(T);
      for (let t = 0; t < T; t++) {
        for (let hh = 0; hh < heads; hh++) {
          const kh = Math.floor(hh / rep), qo = t * qd + hh * hd;
          let mx = -Infinity;
          for (let s = 0; s <= t; s++) { let a = 0; const ko = s * kd + kh * hd; for (let d = 0; d < hd; d++) a += q[qo + d] * k[ko + d]; a *= scl; sc[s] = a; if (a > mx) mx = a; }
          let sum = 0;
          for (let s = 0; s <= t; s++) { sc[s] = Math.exp(sc[s] - mx); sum += sc[s]; }
          const op = t * qd + hh * hd;
          for (let d = 0; d < hd; d++) att[op + d] = 0;
          for (let s = 0; s <= t; s++) { const w = sc[s] / sum, vo = s * kd + kh * hd; for (let d = 0; d < hd; d++) att[op + d] += w * v[vo + d]; }
        }
        if ((t & 31) === 0) await y.tick();
      }
      const o = await vMM(l.o, att, T, y);
      for (let t = 0; t < T; t++) for (let c = 0; c < H; c++) x[t * H + c] += l.sAttn[c] * o[t * H + c];
      for (let t = 0; t < T; t++) vLayerNorm(x, t * H, l.ln2w, l.ln2b, H, this.normEps, h, t * H);
      const m1 = await vMM(l.fc1, h, T, y);
      for (let i = 0; i < m1.length; i++) { const z = m1[i]; m1[i] = 0.5 * z * (1 + vErf(z * Math.SQRT1_2)); }
      const o2 = await vMM(l.fc2, m1, T, y);
      for (let t = 0; t < T; t++) for (let c = 0; c < H; c++) x[t * H + c] += l.sMlp[c] * o2[t * H + c];
    }
  }

  static _nearest(cb, r, ro) {
    const n = cb.rows, d = cb.cols, e = cb.data;
    let best = 0, bd = Infinity;
    for (let i = 0; i < n; i++) { let s = 0; const eo = i * d; for (let j = 0; j < d; j++) { const z = r[ro + j] - e[eo + j]; s += z * z; } if (s < bd) { bd = s; best = i; } }
    return best;
  }

  // Коды образца: Int16Array [frames][validQ]
  async encode(pcm, onProgress) {
    if (!this.hasEnc) fail("в этой модели нет энкодера речевого токенайзера: соберите модель из Base-чекпойнта", "unsupported");
    const y = this.y, n = pcm.length;
    let T = n;
    let r = await vConvMimi(this.conv0, pcm, T, "zero", y); let x = r.y; T = r.T;
    let step = 0; const total = this.stages.length + 3;
    for (const st of this.stages) {
      for (const rb of st.res) {
        const a = vElu(x.slice());
        const b = vElu((await vConvMimi(rb.c1, a, T, "zero", y)).y);
        const c = (await vConvMimi(rb.c2, b, T, "zero", y)).y;
        for (let i = 0; i < x.length; i++) x[i] += c[i];
      }
      vElu(x);
      r = await vConvMimi(st.down, x, T, "zero", y); x = r.y; T = r.T;
      if (onProgress) onProgress(++step / total);
    }
    vElu(x);
    r = await vConvMimi(this.convLast, x, T, "zero", y); x = r.y; T = r.T;
    if (onProgress) onProgress(++step / total);
    await this._transformer(x, T);
    if (onProgress) onProgress(++step / total);
    r = await vConvMimi(this.downsample, x, T, "replicate", y); x = r.y; T = r.T;
    const H = this.H;
    const sem = this.semIn ? await vMM(this.semIn, x, T, y) : x, ac = this.acIn ? await vMM(this.acIn, x, T, y) : x;
    const sdim = this.semIn ? this.semIn.rows : H, adim = this.acIn ? this.acIn.rows : H;
    const nsem = this.semCb.length, nac = this.validQ - nsem, vq = this.validQ;
    const keep = Math.min(T, Math.ceil(n / this.downRate));
    const codes = new Int16Array(keep * vq);
    for (let t = 0; t < keep; t++) {
      const rr = sem.slice(t * sdim, (t + 1) * sdim);
      for (let i = 0; i < nsem; i++) {
        const id = QwenVoiceEncoder._nearest(this.semCb[i], rr, 0); codes[t * vq + i] = id;
        const e = this.semCb[i].data; for (let j = 0; j < sdim; j++) rr[j] -= e[id * sdim + j];
      }
      const ra = ac.slice(t * adim, (t + 1) * adim);
      for (let i = 0; i < nac; i++) {
        const id = QwenVoiceEncoder._nearest(this.acCb[i], ra, 0); codes[t * vq + nsem + i] = id;
        const e = this.acCb[i].data; for (let j = 0; j < adim; j++) ra[j] -= e[id * adim + j];
      }
      if ((t & 7) === 0) await y.tick();
    }
    if (onProgress) onProgress(1);
    return { codes, frames: keep, codebooks: vq };
  }
}
function vLayerNorm(x, xo, w, b, n, eps, out, oo) {
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[xo + i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) { const d = x[xo + i] - mean; v += d * d; }
  v /= n;
  const inv = f32r(1 / Math.sqrt(v + eps)), m = f32r(mean);
  for (let i = 0; i < n; i++) out[oo + i] = (x[xo + i] - m) * inv * w[i] + (b ? b[i] : 0);
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
      g.run1d(rope, [ws.q, ws.k, ws.cs], [R, N, H, HD, 0, 0, k.pe_attn_head || 0, 0], R * H * (HD >> 1));
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
    const pcm = istftSame(h, T, V.n_fft, V.hop, V.center);
    this.timing.vocos_ms += performance.now() - t0;
    return pcm;
  },
});

// ============================================================================
// Конвертер весов прямо в браузере: чекпойнт F5-TTS (safetensors / torch .pt) + вокодер Vocos (.bin / safetensors)
// + vocab.txt → байты .tmod, тот же формат, что у python -m vox.build_f5 (проверяется побайтно по тензорам).
// Читалки: safetensors, zip, мини-интерпретатор pickle (достаточный для torch.save state_dict).
// ============================================================================

// ---------- f32 → f16 / bf16 (округление к чётному, как numpy / vox.build) ----------
const _f32b = new DataView(new ArrayBuffer(4));
function f32ToF16Bits(x) {
  _f32b.setFloat32(0, x);
  const u = _f32b.getUint32(0);
  const sign = (u >>> 16) & 0x8000, e = (u >>> 23) & 0xff, m = u & 0x7fffff;
  if (e === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
  let ne = e - 127 + 15;
  if (ne >= 31) return sign | 0x7c00;
  if (ne <= 0) {
    if (ne < -10) return sign;
    const mm = m | 0x800000, shift = 14 - ne;
    let h = mm >>> shift;
    const rem = mm & ((1 << shift) - 1), half = 1 << (shift - 1);
    if (rem > half || (rem === half && (h & 1))) h++;
    return sign | h;
  }
  let h = (ne << 10) | (m >>> 13);
  const rem = m & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++;   // перенос в экспоненту корректен
  return sign | h;
}
function f32ToBf16Bits(x) {
  _f32b.setFloat32(0, x);
  const u = _f32b.getUint32(0);
  return (((u + 0x7fff + ((u >>> 16) & 1)) >>> 0) >>> 16) & 0xffff;
}
function rintEven(x) {
  const r = Math.round(x);
  return (Math.abs(x % 1) === 0.5 && (r & 1)) ? r - 1 : r;
}

// ---------- чтение safetensors из байтов ----------
function parseSafetensors(bytes) {
  if (bytes.length < 8) fail("safetensors: файл слишком мал", "format");
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const n = Number(dv.getBigUint64(0, true));
  if (n > bytes.length - 8) fail("safetensors: заголовок вылезает за файл (недокачан?)", "format");
  const hdr = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + n)));
  const meta = hdr.__metadata__ || {};
  delete hdr.__metadata__;
  const base = 8 + n, out = new Map();
  out.metadata = meta;
  for (const k of Object.keys(hdr)) {
    const t = hdr[k];
    out.set(k, { shape: t.shape.slice(), dtype: t.dtype, bytes: bytes.subarray(base + t.data_offsets[0], base + t.data_offsets[1]) });
  }
  return out;
}
const ST_SIZE = { F32: 4, F16: 2, BF16: 2, F64: 8 };
// сырые байты тензора → Float32Array
function rawToF32(raw, dtype, numel) {
  const need = (ST_SIZE[dtype] || 0) * numel;
  if (!ST_SIZE[dtype]) fail("неподдерживаемый тип тензора: " + dtype, "format");
  if (raw.length < need) fail("тензор короче, чем требует форма", "format");
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const out = new Float32Array(numel);
  switch (dtype) {
    case "F32": for (let i = 0; i < numel; i++) out[i] = dv.getFloat32(i * 4, true); break;
    case "F16": for (let i = 0; i < numel; i++) out[i] = f16ToF32(dv.getUint16(i * 2, true)); break;
    case "BF16": for (let i = 0; i < numel; i++) out[i] = bf16ToF32(dv.getUint16(i * 2, true)); break;
    case "F64": for (let i = 0; i < numel; i++) out[i] = dv.getFloat64(i * 8, true); break;
  }
  return out;
}

// ---------- zip (то, что делает torch.save): stored и deflate ----------
class ZipReader {
  constructor(bytes) {
    this.bytes = bytes;
    this.dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let e = bytes.length - 22;
    while (e >= 0 && this.dv.getUint32(e, true) !== 0x06054b50) e--;
    if (e < 0) fail("это не zip-архив", "format");
    const n = this.dv.getUint16(e + 10, true);
    let p = this.dv.getUint32(e + 16, true);
    if (n === 0xffff || p === 0xffffffff) fail("zip64 не поддерживается", "unsupported");
    this.entries = new Map();
    for (let i = 0; i < n; i++) {
      if (this.dv.getUint32(p, true) !== 0x02014b50) fail("zip: повреждён центральный каталог", "format");
      const method = this.dv.getUint16(p + 10, true), csize = this.dv.getUint32(p + 20, true), usize = this.dv.getUint32(p + 24, true);
      const nl = this.dv.getUint16(p + 28, true), el = this.dv.getUint16(p + 30, true), cl = this.dv.getUint16(p + 32, true);
      const lho = this.dv.getUint32(p + 42, true);
      const name = new TextDecoder().decode(bytes.subarray(p + 46, p + 46 + nl));
      this.entries.set(name, { method, csize, usize, lho });
      p += 46 + nl + el + cl;
    }
  }
  names() { return Array.from(this.entries.keys()); }
  has(name) { return this.entries.has(name); }
  async read(name) {
    const en = this.entries.get(name);
    if (!en) fail("в архиве нет " + name, "format");
    const lh = en.lho, nl = this.dv.getUint16(lh + 26, true), el = this.dv.getUint16(lh + 28, true);
    const start = lh + 30 + nl + el;
    const raw = this.bytes.subarray(start, start + en.csize);
    if (en.method === 0) return raw;
    if (en.method !== 8) fail("zip: метод сжатия " + en.method + " не поддерживается", "unsupported");
    if (typeof DecompressionStream === "undefined") fail("для сжатого zip нужен DecompressionStream", "unsupported");
    const ds = new DecompressionStream("deflate-raw");
    const w = ds.writable.getWriter();
    w.write(raw); w.close();
    return new Uint8Array(await new Response(ds.readable).arrayBuffer());
  }
}

// ---------- pickle: минимальный интерпретатор для state_dict ----------
class PyGlobal { constructor(m, n) { this.module = m; this.name = n; } }
class PyTensor { constructor(storage, offset, size, stride) { this.storage = storage; this.offset = offset; this.size = size; this.stride = stride; } }
class PyStorageRef { constructor(cls, key, numel) { this.cls = cls; this.key = key; this.numel = numel; } }
const MARK = { mark: true };
const TORCH_ST = { FloatStorage: "F32", HalfStorage: "F16", BFloat16Storage: "BF16", DoubleStorage: "F64", LongStorage: "I64", IntStorage: "I32", ShortStorage: "I16", CharStorage: "I8", ByteStorage: "U8", BoolStorage: "BOOL" };

function unpickle(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const td = new TextDecoder();
  const st = [], memo = new Map();
  let p = 0;
  const readLine = () => { let e = p; while (buf[e] !== 10) e++; const s = td.decode(buf.subarray(p, e)); p = e + 1; return s; };
  const popMark = () => { let i = st.length - 1; while (st[i] !== MARK) i--; const items = st.slice(i + 1); st.length = i; return items; };
  const reduce = (fn, args) => {
    if (fn instanceof PyGlobal) {
      const k = fn.module + "." + fn.name;
      if (k === "collections.OrderedDict" || k === "builtins.dict") return new Map();
      if (k === "torch._utils._rebuild_tensor_v2" || k === "torch._utils._rebuild_tensor") return new PyTensor(args[0], args[1], args[2], args[3]);
      if (k === "torch._utils._rebuild_parameter") return args[0];
      if (k === "builtins.set" || k === "builtins.frozenset") return [];
      if (k === "torch.serialization._get_layout" || k === "torch._utils._rebuild_device_tensor_from_numpy") return null;
      return { reduced: k, args };
    }
    return { reduced: "?", args };
  };
  for (;;) {
    const op = buf[p++];
    switch (op) {
      case 0x80: p += 1; break;                                  // PROTO
      case 0x95: p += 8; break;                                  // FRAME
      case 0x2e: return st.pop();                                // STOP
      case 0x63: { const m = readLine(), n = readLine(); st.push(new PyGlobal(m, n)); break; }   // GLOBAL
      case 0x93: { const n = st.pop(), m = st.pop(); st.push(new PyGlobal(m, n)); break; }       // STACK_GLOBAL
      case 0x71: memo.set(buf[p++], st[st.length - 1]); break;   // BINPUT
      case 0x72: memo.set(dv.getUint32(p, true), st[st.length - 1]); p += 4; break;   // LONG_BINPUT
      case 0x94: memo.set(memo.size, st[st.length - 1]); break;  // MEMOIZE
      case 0x68: st.push(memo.get(buf[p++])); break;             // BINGET
      case 0x6a: st.push(memo.get(dv.getUint32(p, true))); p += 4; break;   // LONG_BINGET
      case 0x70: memo.set(Number(readLine()), st[st.length - 1]); break;    // PUT
      case 0x67: st.push(memo.get(Number(readLine()))); break;              // GET
      case 0x28: st.push(MARK); break;                           // MARK
      case 0x7d: st.push(new Map()); break;                      // EMPTY_DICT
      case 0x5d: st.push([]); break;                             // EMPTY_LIST
      case 0x29: st.push([]); break;                             // EMPTY_TUPLE
      case 0x74: st.push(popMark()); break;                      // TUPLE
      case 0x85: { const a = st.pop(); st.push([a]); break; }
      case 0x86: { const b = st.pop(), a = st.pop(); st.push([a, b]); break; }
      case 0x87: { const c = st.pop(), b = st.pop(), a = st.pop(); st.push([a, b, c]); break; }
      case 0x6c: st.push(popMark()); break;                      // LIST
      case 0x64: { const it = popMark(), m = new Map(); for (let i = 0; i + 1 < it.length; i += 2) m.set(it[i], it[i + 1]); st.push(m); break; }   // DICT
      case 0x4a: st.push(dv.getInt32(p, true)); p += 4; break;   // BININT
      case 0x4b: st.push(buf[p++]); break;                       // BININT1
      case 0x4d: st.push(dv.getUint16(p, true)); p += 2; break;  // BININT2
      case 0x8a: { const n = buf[p++]; let v = 0n; for (let i = n - 1; i >= 0; i--) v = (v << 8n) | BigInt(buf[p + i]); if (n && (buf[p + n - 1] & 0x80)) v -= 1n << BigInt(8 * n); p += n; st.push(Number(v)); break; }   // LONG1
      case 0x49: case 0x4c: { st.push(Number(readLine().replace(/L$/, ""))); break; }   // INT, LONG (текст)
      case 0x46: st.push(Number(readLine())); break;             // FLOAT
      case 0x47: st.push(dv.getFloat64(p, false)); p += 8; break;   // BINFLOAT
      case 0x58: { const n = dv.getUint32(p, true); p += 4; st.push(td.decode(buf.subarray(p, p + n))); p += n; break; }   // BINUNICODE
      case 0x8c: { const n = buf[p++]; st.push(td.decode(buf.subarray(p, p + n))); p += n; break; }   // SHORT_BINUNICODE
      case 0x8d: { const n = Number(dv.getBigUint64(p, true)); p += 8; st.push(td.decode(buf.subarray(p, p + n))); p += n; break; }
      case 0x56: st.push(readLine()); break;                     // UNICODE (текст)
      case 0x55: { const n = buf[p++]; st.push(td.decode(buf.subarray(p, p + n))); p += n; break; }   // SHORT_BINSTRING
      case 0x54: { const n = dv.getUint32(p, true); p += 4; st.push(td.decode(buf.subarray(p, p + n))); p += n; break; }   // BINSTRING
      case 0x53: st.push(readLine().replace(/^['"]|['"]$/g, "")); break;   // STRING
      case 0x43: { const n = buf[p++]; st.push(buf.slice(p, p + n)); p += n; break; }   // SHORT_BINBYTES
      case 0x42: { const n = dv.getUint32(p, true); p += 4; st.push(buf.slice(p, p + n)); p += n; break; }   // BINBYTES
      case 0x88: st.push(true); break;
      case 0x89: st.push(false); break;
      case 0x4e: st.push(null); break;
      case 0x30: st.pop(); break;                                // POP
      case 0x32: st.push(st[st.length - 1]); break;              // DUP
      case 0x31: popMark(); break;                               // POP_MARK
      case 0x51: {                                               // BINPERSID: ('storage', cls, key, location, numel)
        const pid = st.pop();
        if (!Array.isArray(pid) || pid[0] !== "storage") fail("pickle: неизвестный persistent id", "format");
        const cls = pid[1] instanceof PyGlobal ? pid[1].name : String(pid[1]);
        st.push(new PyStorageRef(cls, String(pid[2]), pid[4]));
        break;
      }
      case 0x52: { const args = st.pop(), fn = st.pop(); st.push(reduce(fn, args)); break; }   // REDUCE
      case 0x81: { const args = st.pop(), cls = st.pop(); st.push(reduce(cls, args)); break; }  // NEWOBJ
      case 0x62: st.pop(); break;                                // BUILD (состояние нам не нужно)
      case 0x73: { const v = st.pop(), k = st.pop(), d = st[st.length - 1]; if (d instanceof Map) d.set(k, v); break; }   // SETITEM
      case 0x75: { const it = popMark(), d = st[st.length - 1]; if (d instanceof Map) for (let i = 0; i + 1 < it.length; i += 2) d.set(it[i], it[i + 1]); break; }   // SETITEMS
      case 0x61: { const v = st.pop(); st[st.length - 1].push(v); break; }      // APPEND
      case 0x65: { const it = popMark(); const l = st[st.length - 1]; if (Array.isArray(l)) for (const v of it) l.push(v); break; }   // APPENDS
      default: fail("pickle: операция 0x" + op.toString(16) + " не поддерживается (файл не в формате torch.save?)", "unsupported");
    }
  }
}

// state_dict из torch-архива (.pt/.pth/.bin). Возвращает Map имя → {shape, get()}.
async function readTorchArchive(bytes, preferKeys) {
  if (!(bytes[0] === 0x50 && bytes[1] === 0x4b)) fail("это не torch-архив: старый формат .pt (до PyTorch 1.6) не поддерживается — пересохраните через safetensors", "unsupported");
  const zip = new ZipReader(bytes);
  const pkl = zip.names().find((n) => /(^|\/)data\.pkl$/.test(n));
  if (!pkl) fail("в архиве нет data.pkl", "format");
  const prefix = pkl.slice(0, pkl.length - "data.pkl".length);
  let obj = unpickle(await zip.read(pkl));
  for (const key of preferKeys || []) {
    if (obj instanceof Map && obj.get(key) instanceof Map) { obj = obj.get(key); break; }
  }
  if (!(obj instanceof Map)) fail("в .pt нет словаря весов", "format");
  const storages = new Map();
  const out = new Map();
  for (const [k, t] of obj) {
    if (!(t instanceof PyTensor) || !(t.storage instanceof PyStorageRef)) continue;
    const dt = TORCH_ST[t.storage.cls];
    if (!dt || !ST_SIZE[dt]) continue;              // нужны только вещественные
    const shape = t.size.slice();
    let expect = 1;
    const stride = new Array(shape.length);
    for (let i = shape.length - 1; i >= 0; i--) { stride[i] = expect; expect *= shape[i]; }
    if (t.stride && shape.length > 1 && t.stride.some((s, i) => shape[i] > 1 && s !== stride[i])) fail("тензор " + k + " не непрерывен (stride): такой .pt не поддерживается", "unsupported");
    const numel = shape.reduce((a, b) => a * b, 1);
    out.set(String(k), {
      shape, dtype: dt,
      async load() {
        const key = t.storage.key;
        if (!storages.has(key)) storages.set(key, await zip.read(prefix + "data/" + key));
        const raw = storages.get(key), es = ST_SIZE[dt];
        return rawToF32(raw.subarray(t.offset * es, (t.offset + numel) * es), dt, numel);
      },
    });
  }
  return out;
}

// ---------- набор весов: имя → {shape, get()} ----------
function stripPrefix(k) {
  for (const p of ["module.", "_orig_mod.", "ema_model.", "model."]) while (k.startsWith(p)) k = k.slice(p.length);
  return k;
}
function sniffKind(bytes) {
  if (bytes.length >= 4 && bytes[0] === 0x56 && bytes[1] === 0x4f && bytes[2] === 0x58 && bytes[3] === 0x4d) return "tmod";
  if (bytes.length >= 4 && bytes[0] === 0x50 && bytes[1] === 0x4b) return "torch";
  if (bytes.length > 16) {
    const n = Number(new DataView(bytes.buffer, bytes.byteOffset, 16).getBigUint64(0, true));
    if (n > 1 && n < bytes.length && bytes[8] === 0x7b) return "safetensors";
  }
  if (bytes.length > 2 && bytes[0] === 0x80) return "torch-legacy";
  return "unknown";
}
async function loadWeightSet(bytes, prefix, keepPrefixes) {
  const kind = sniffKind(bytes);
  const out = new Map();
  if (kind === "safetensors") {
    for (const [k, t] of parseSafetensors(bytes)) {
      const kk = stripPrefix(k);
      if (prefix && !kk.startsWith(prefix)) continue;
      const name = kk.slice(prefix.length);
      if (keepPrefixes && !keepPrefixes.some((q) => name.startsWith(q))) continue;
      if (!ST_SIZE[t.dtype]) continue;
      const numel = t.shape.reduce((a, b) => a * b, 1);
      out.set(name, { shape: t.shape, async get() { return rawToF32(t.bytes, t.dtype, numel); } });
    }
  } else if (kind === "torch") {
    const m = await readTorchArchive(bytes, ["ema_model_state_dict", "model_state_dict", "state_dict"]);
    for (const [k, t] of m) {
      const kk = stripPrefix(k);
      if (prefix && !kk.startsWith(prefix)) continue;
      const name = kk.slice(prefix.length);
      if (keepPrefixes && !keepPrefixes.some((q) => name.startsWith(q))) continue;
      out.set(name, { shape: t.shape, get: t.load });
    }
  } else if (kind === "tmod") fail("ожидались веса чекпойнта, а получен готовый .tmod", "format");
  else fail("неизвестный формат файла весов (нужны .safetensors, .pt, .pth или .bin от torch.save)", "format");
  return out;
}

// ---------- банк mel-фильтров (torchaudio.melscale_fbanks, htk, norm=None; транспонирован: [n_mels][n_freqs]) ----------
function melFilterbank(sr, nFft, nMels, fMin = 0, fMax = null) {
  // Считаем в double. torchaudio делает то же в float32 и расходится с точным значением на ~1e-5 (шум округления f_pts):
  // на звук это не влияет, но бит-в-бит с .tmod из Python банк фильтров не совпадает.
  fMax = fMax || sr / 2;
  const h2m = (f) => 2595 * Math.log10(1 + f / 700), m2h = (m) => 700 * (Math.pow(10, m / 2595) - 1);
  const nF = (nFft >> 1) + 1;
  const allF = new Float64Array(nF);
  for (let i = 0; i < nF; i++) allF[i] = (Math.floor(sr / 2) * i) / (nF - 1);
  const fp = new Float64Array(nMels + 2), m0 = h2m(fMin), m1 = h2m(fMax);
  for (let i = 0; i < nMels + 2; i++) fp[i] = m2h(m0 + ((m1 - m0) * i) / (nMels + 1));
  const out = new Float32Array(nMels * nF);
  for (let m = 0; m < nMels; m++) {
    const d0 = fp[m + 1] - fp[m], d1 = fp[m + 2] - fp[m + 1];
    for (let i = 0; i < nF; i++) out[m * nF + i] = Math.max(0, Math.min(-(fp[m] - allF[i]) / d0, (fp[m + 2] - allF[i]) / d1));
  }
  return out;
}

// ---------- запись .tmod ----------
const TMOD_ALIGN = 4096;
const DT_BYTES = { F32: 4, F16: 2, BF16: 2, I8: 1, U8: 1 };
// blobs: [{name, dtype, shape, fill(u8view)}] ; возвращает Uint8Array
async function writeTmod(manifest, blobs, onProgress) {
  blobs.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  const table = {};
  let off = 0;
  for (const b of blobs) {
    off = Math.ceil(off / TMOD_ALIGN) * TMOD_ALIGN;
    b.nbytes = b.shape.reduce((a, c) => a * c, 1) * DT_BYTES[b.dtype];
    table[b.name] = { dtype: b.dtype, shape: b.shape, offset: off, nbytes: b.nbytes };
    b.offset = off;
    off += b.nbytes;
  }
  const m = Object.assign({}, manifest, { blobs: table });
  const mj = new TextEncoder().encode(JSON.stringify(m, null, 2));
  const dataOff = Math.ceil((64 + mj.length) / TMOD_ALIGN) * TMOD_ALIGN;
  const total = dataOff + off;
  let out;
  try { out = new Uint8Array(total); } catch (e) { fail("не хватает памяти для сборки модели (" + (total / 1048576 | 0) + " МиБ): выберите q8/bf16 или готовый .tmod", "memory"); }
  const dv = new DataView(out.buffer);
  out.set([0x56, 0x4f, 0x58, 0x4d]);          // VOXM
  dv.setUint16(4, 1, true); dv.setUint16(6, 0, true); dv.setUint32(8, 0, true); dv.setUint32(12, 0, true);
  dv.setBigUint64(16, 64n, true); dv.setBigUint64(24, BigInt(mj.length), true);
  dv.setBigUint64(32, BigInt(dataOff), true); dv.setBigUint64(40, BigInt(off), true);
  out.set(mj, 64);
  let done = 0;
  for (const b of blobs) {
    await b.fill(out.subarray(dataOff + b.offset, dataOff + b.offset + b.nbytes));
    done++;
    if (onProgress) onProgress(done / blobs.length);
  }
  return out;
}

// ---------- квантование q8 (как quantize_q8 в vox.build) ----------
function quantizeQ8(w, rows, cols) {
  const nb = cols >> 5;
  const q = new Int8Array(rows * cols), s = new Float32Array(rows * nb);
  for (let r = 0; r < rows; r++) {
    for (let b = 0; b < nb; b++) {
      const o = r * cols + b * 32;
      let mx = 0;
      for (let j = 0; j < 32; j++) { const a = Math.abs(w[o + j]); if (a > mx) mx = a; }
      const sc = f32r(mx / 127);
      s[r * nb + b] = sc;
      if (sc > 0) for (let j = 0; j < 32; j++) q[o + j] = rintEven(f32r(w[o + j] / sc));
    }
  }
  return { q, s };
}

const F5_BLOCK_LINEAR = /^transformer_blocks\.\d+\.(attn\.(to_q|to_k|to_v)\.weight|attn\.to_out\.0\.weight|attn_norm\.linear\.weight|ff\.ff\.0\.0\.weight|ff\.ff\.2\.weight)$/;

// добавляет в список блобов тензор: 2D .weight → dtype (или q8), остальное F32
function addBlob(blobs, name, shape, getter, dtype, q8) {
  const is2d = shape.length === 2 && name.endsWith(".weight");
  if (!is2d) {
    blobs.push({ name, dtype: "F32", shape, async fill(u8) { new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength >> 2).set(await getter()); } });
    return;
  }
  if (q8 && (shape[1] & 31) === 0) {
    const cache = {};
    const get = async () => cache.v || (cache.v = quantizeQ8(await getter(), shape[0], shape[1]));
    blobs.push({ name, dtype: "I8", shape, async fill(u8) { new Int8Array(u8.buffer, u8.byteOffset, u8.byteLength).set((await get()).q); } });
    blobs.push({ name: name + ".scale", dtype: "F32", shape: [shape[0], shape[1] >> 5], async fill(u8) { new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength >> 2).set((await get()).s); cache.v = null; } });
    // порядок заполнения — по имени: ".scale" идёт после основного, кэш освобождается там
    return;
  }
  if (dtype === "F32") blobs.push({ name, dtype, shape, async fill(u8) { new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength >> 2).set(await getter()); } });
  else blobs.push({
    name, dtype, shape,
    async fill(u8) {
      const w = await getter(), o = new Uint16Array(u8.buffer, u8.byteOffset, u8.byteLength >> 1);
      if (dtype === "F16") for (let i = 0; i < w.length; i++) o[i] = f32ToF16Bits(w[i]);
      else for (let i = 0; i < w.length; i++) o[i] = f32ToBf16Bits(w[i]);
    },
  });
}

async function sha256Hex(parts) {
  if (typeof crypto === "undefined" || !crypto.subtle) return "";
  const hex = (b) => Array.from(new Uint8Array(b), (x) => x.toString(16).padStart(2, "0")).join("");
  let all = "";
  for (const [name, bytes] of parts) all += name + ":" + hex(await crypto.subtle.digest("SHA-256", bytes)) + "\n";
  return hex(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(all)));
}

// ---------- сборка F5 ----------
// o: { ckpt: Uint8Array, ckptName, vocab: string, vocos: Uint8Array, vocosName, dtype: "bf16"|"f16"|"f32", quant: ""|"q8", vocosDtype (default "f32"),
//      vocosConfig: text of the vocoder's config.yaml (or vocosPadding: "center"|"same"; default center),
//      tokenizer: "pinyin"|"char", variant, dimHead, pinyin: {char: syllable}|null, mel, hash: bool, onProgress(stage, frac) }
// F5TTS_Base / F5TTS_Small (v0): text_mask_padding=false, pe_attn_head=1; F5TTS_v1_*: true / all heads. The weights do not say which, so the
// config is chosen by opts.modelCfg ("v0" | "v1") or guessed from the file/repo name; unknown -> v1.
// vocab.txt: one token per line; CRLF/CR are line breaks (as in f5_tts get_tokenizer, which reads the file in text mode).
function splitVocab(text) {
  const lines = String(text).replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
  if (lines.length && lines[lines.length - 1] === "") lines.pop();
  return lines;
}
// ISTFT padding of the Vocos head from its config.yaml ("center" for charactr/vocos-mel-24khz, ISTFTHead default "same");
// no config -> the official F5-TTS vocoder (center).
function parseVocosPadding(yamlText) {
  if (yamlText == null) return "center";
  const lines = String(yamlText).replace(/\r\n/g, "\n").split("\n");
  const i = lines.findIndex((l) => /^head:\s*$/.test(l));
  if (i < 0) return "center";
  let pad = "same";
  for (let j = i + 1; j < lines.length && (lines[j] === "" || /^[ \t]/.test(lines[j])); j++) {
    const m = /padding:\s*['"]?(\w+)/.exec(lines[j]);
    if (m) { pad = m[1]; break; }
  }
  if (pad !== "center" && pad !== "same") fail("Vocos: padding '" + pad + "' не поддерживается (center, same)", "unsupported");
  return pad;
}
function guessF5Cfg(...names) {
  for (const n of names) {
    const s = String(n || "");
    if (/F5TTS_v1/i.test(s)) return "v1";
    if (/F5TTS_(Base|Small)/i.test(s)) return "v0";
  }
  return "v1";
}
async function convertF5(o) {
  const modelCfg = o.modelCfg || guessF5Cfg(o.ckptName, o.variant);
  if (modelCfg !== "v0" && modelCfg !== "v1") fail("modelCfg: v0 или v1", "arg");
  const maskPadding = o.textMaskPadding != null ? !!o.textMaskPadding : modelCfg === "v1";
  const peAttnHead = o.peAttnHead != null ? (o.peAttnHead || null) : (modelCfg === "v0" ? 1 : null);
  const dtype = (o.dtype || "bf16").toUpperCase(), quant = o.quant || "", tokenizer = o.tokenizer || "pinyin", dimHead = o.dimHead || 64;
  if (!["F32", "F16", "BF16"].includes(dtype)) fail("dtype: f32, f16 или bf16", "arg");
  if (quant && quant !== "q8") fail("quant: пусто или q8", "arg");
  const prog = o.onProgress || (() => {});
  prog("read", 0);
  const dit = await loadWeightSet(o.ckpt, "transformer.");
  if (!dit.size) fail("в чекпойнте нет тензоров transformer.* — это не модель F5-TTS", "format");
  const voc = await loadWeightSet(o.vocos, "", ["backbone.", "head.out."]);
  if (!voc.size) fail("в файле вокодера нет backbone.* / head.out.* — это не Vocos", "format");
  const sh = (m, k) => { const t = m.get(k); if (!t) fail("в весах нет " + k, "format"); return t.shape; };
  const dim = sh(dit, "time_embed.time_mlp.2.weight")[0];
  let depth = 0, convLayers = 0;
  for (const k of dit.keys()) {
    let m = /^transformer_blocks\.(\d+)\./.exec(k); if (m) depth = Math.max(depth, +m[1] + 1);
    m = /^text_embed\.text_blocks\.(\d+)\./.exec(k); if (m) convLayers = Math.max(convLayers, +m[1] + 1);
  }
  const [textRows, textDim] = sh(dit, "text_embed.text_embed.weight");
  const inner = sh(dit, "transformer_blocks.0.attn.to_q.weight")[0];
  const heads = inner / dimHead;
  const ffMult = sh(dit, "transformer_blocks.0.ff.ff.0.0.weight")[0] / dim;
  const melDim = sh(dit, "proj_out.weight")[0];
  const convMult = convLayers ? sh(dit, "text_embed.text_blocks.0.pwconv1.weight")[0] / textDim : 2;
  for (const k of dit.keys()) {
    if (k.startsWith("long_skip_connection")) fail("F5-TTS: long_skip_connection пока не поддерживается", "unsupported");
    if (k.includes(".q_norm.")) fail("F5-TTS: qk_norm пока не поддерживается", "unsupported");
  }
  if (sh(dit, "input_embed.proj.weight")[1] !== 2 * melDim + textDim) fail("неожиданная форма input_embed.proj", "format");
  const vocabList = splitVocab(o.vocab);
  const mel = Object.assign({ sample_rate: 24000, n_fft: 1024, hop: 256, win: 1024, n_mels: melDim }, o.mel || {});
  const vdim = sh(voc, "backbone.embed.weight")[0], vinter = sh(voc, "backbone.convnext.0.pwconv1.weight")[0];
  let vlayers = 0;
  for (const k of voc.keys()) { const m = /^backbone\.convnext\.(\d+)\./.exec(k); if (m) vlayers = Math.max(vlayers, +m[1] + 1); }
  const vout = sh(voc, "head.out.weight")[0];
  const vocosCfg = { dim: vdim, intermediate: vinter, layers: vlayers, n_fft: vout - 2, hop: mel.hop,
    padding: o.vocosPadding || parseVocosPadding(o.vocosConfig) };
  if (vocosCfg.padding !== "center" && vocosCfg.padding !== "same") fail("vocosPadding: center или same", "arg");

  const blobs = [];
  for (const [k, t] of dit) addBlob(blobs, "dit." + k, t.shape, t.get, dtype, quant === "q8" && F5_BLOCK_LINEAR.test(k));
  // Vocos stays f32 by default, as in python -m vox.build_f5 (--vocos-dtype): small, but bf16 weights there add waveform error
  for (const [k, t] of voc) addBlob(blobs, "vocos." + k, t.shape, t.get, (o.vocosDtype || "f32").toUpperCase(), false);
  const fb = melFilterbank(mel.sample_rate, mel.n_fft, mel.n_mels);
  blobs.push({ name: "mel.fb", dtype: "F32", shape: [mel.n_mels, (mel.n_fft >> 1) + 1], async fill(u8) { new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength >> 2).set(fb); } });

  const tok = { type: tokenizer, vocab: vocabList };
  if (tokenizer === "pinyin" && o.pinyin) tok.pinyin = o.pinyin;
  const gen = { nfe_step: 32, cfg_strength: 2.0, sway_sampling_coef: -1.0, speed: 1.0, cross_fade_duration: 0.15, target_rms: 0.1, use_epss: true };
  Object.assign(gen, o.defaults || {});
  const sha = o.hash === false ? "" : await sha256Hex([[o.ckptName || "model.safetensors", o.ckpt], [o.vocosName || "pytorch_model.bin", o.vocos]]);
  const manifest = {
    format: "1.0",
    model: { family: "f5-tts", variant: o.variant || "f5-tts", type: "tts", weights_sha256: sha },
    target: { backend: "cpu", arch: "any", features: [] },
    audio: { sample_rate: mel.sample_rate, samples_per_frame: mel.hop, frame_rate: mel.sample_rate / mel.hop, num_codebooks: 1, codebook_size: 1 },
    stages: [{ name: "dit", kind: "flow_matching" }, { name: "vocos", kind: "vocoder" }],
    precision: { weights: o.dtype ? o.dtype.toLowerCase() : "bf16", quant: quant || "none" },
    generation: gen,
    f5: { dim, depth, heads, dim_head: dimHead, ff_mult: ffMult, text_dim: textDim, text_rows: textRows, mel_dim: melDim, conv_layers: convLayers, conv_mult: convMult,
      text_mask_padding: maskPadding, pe_attn_head: peAttnHead, qk_norm: null, mel, vocos: vocosCfg, tokenizer: tok },
  };
  if (vocabList.length + 1 !== textRows && o.onWarn) o.onWarn("словарь: " + vocabList.length + " записей, а в таблице эмбеддингов " + textRows + " строк (ожидалось на одну больше)");
  prog("read", 1);
  return writeTmod(manifest, blobs, (f) => prog("convert", f));
}

// ============================================================================
// Откуда берётся модель: готовый .tmod, чекпойнт (safetensors / torch) по прямой ссылке, репозиторий Hugging Face
// (ссылка, hf:org/name, org/name), File/Blob/байты, либо объект {ckpt, vocab, vocos}. Всё, что не .tmod, конвертируется
// здесь же, в браузере. Результат конвертации кэшируется (Cache API), чтобы в следующий раз не скачивать и не конвертировать.
// ============================================================================
const HF_HOST = "https://huggingface.co";
const DEFAULT_VOCOS = "charactr/vocos-mel-24khz";
const CKPT_EXT = /\.(safetensors|pt|pth)$/i;

function hfHeaders(opts) { return opts && opts.token ? { Authorization: "Bearer " + opts.token } : {}; }

// "https://huggingface.co/org/name[/tree|blob|resolve/rev/path]" | "hf:org/name[@rev][/path]" | "org/name" → { repo, rev, path, kind }
function parseHfRef(s, base) {
  if (typeof s !== "string") return null;
  const host = (base || HF_HOST).replace(/\/+$/, "");
  s = s.trim();
  let m;
  if ((m = /^hf:\/?\/?([\w.-]+\/[\w.-]+)(?:@([\w.-]+))?(?:\/(.*))?$/.exec(s))) return { repo: m[1], rev: m[2] || "main", path: m[3] || "", kind: m[3] ? "path" : "repo" };
  const esc = host.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  if ((m = new RegExp("^" + esc + "/([\\w.-]+/[\\w.-]+)(?:/(tree|blob|resolve)/([^/?#]+)(?:/([^?#]*))?)?/?(?:[?#].*)?$").exec(s))) {
    const k = m[2];
    return { repo: m[1], rev: m[3] || "main", path: decodeURIComponent(m[4] || ""), kind: k === "tree" || !k ? (m[4] ? "dir" : "repo") : "file" };
  }
  if ((m = /^huggingface\.co\/([\w.-]+\/[\w.-]+)(?:\/(tree|blob|resolve)\/([^/?#]+)(?:\/(.*))?)?$/.exec(s)))
    return parseHfRef("https://huggingface.co/" + s.slice("huggingface.co/".length), base);
  if (/^[\w-][\w.-]*\/[\w-][\w.-]*$/.test(s) && !/\.(tmod|safetensors|pt|pth|bin|json|txt|wav|mp3|html|js)$/i.test(s) && !/^(\.|~)/.test(s)) return { repo: s, rev: "main", path: "", kind: "repo" };
  return null;
}

async function hfList(ref, opts) {
  const host = (opts.hfHost || HF_HOST).replace(/\/+$/, "");
  const h = hfHeaders(opts);
  const get = async (u) => {
    const r = await fetch(u, { headers: h });
    if (r.status === 401 || r.status === 403) fail("Hugging Face: доступ к " + ref.repo + " закрыт (нужен токен: opts.token)", "network");
    if (r.status === 404) fail("Hugging Face: репозиторий или ветка не найдены: " + ref.repo + "@" + ref.rev, "network");
    if (!r.ok) fail("Hugging Face: HTTP " + r.status + " для " + u, "network");
    return r.json();
  };
  let files = null;
  try {
    const t = await get(host + "/api/models/" + ref.repo + "/tree/" + encodeURIComponent(ref.rev) + "?recursive=1&expand=0");
    if (Array.isArray(t)) files = t.filter((e) => e.type === "file").map((e) => ({ path: e.path, size: e.size || (e.lfs && e.lfs.size) || 0 }));
  } catch (e) { if (e instanceof VoxError && e.code === "network" && /закрыт|не найдены/.test(e.message)) throw e; }
  if (!files) {
    const info = await get(host + "/api/models/" + ref.repo + "/revision/" + encodeURIComponent(ref.rev));
    files = (info.siblings || []).map((s) => ({ path: s.rfilename, size: s.size || 0 }));
  }
  if (!files.length) fail("в репозитории " + ref.repo + " нет файлов", "network");
  return files;
}
const hfUrl = (ref, path, opts) => (opts.hfHost || HF_HOST).replace(/\/+$/, "") + "/" + ref.repo + "/resolve/" + encodeURIComponent(ref.rev) + "/" + path.split("/").map(encodeURIComponent).join("/");

const dirOf = (p) => (p.includes("/") ? p.slice(0, p.lastIndexOf("/")) : "");
const baseOf = (p) => p.slice(p.lastIndexOf("/") + 1);

// Выбор файлов из списка репозитория. Возвращает { kind: "tmod"|"f5", ... , alternatives }
function planFromFiles(files, ref, opts) {
  const paths = files.map((f) => f.path).filter((p) => !ref.path || ref.kind === "file" || p === ref.path || p.startsWith(ref.path.replace(/\/$/, "") + "/"));
  if (ref.kind === "file") {
    const p = ref.path;
    if (/\.tmod$/i.test(p)) return { kind: "tmod", tmod: p };
    if (CKPT_EXT.test(p)) return planCkpt(files.map((f) => f.path), p, ref, opts);
    fail("не знаю, что делать с файлом " + p + ": нужен .tmod, .safetensors или .pt", "format");
  }
  // Qwen3-TTS: config.json + speech_tokenizer/
  if (paths.includes("config.json") && paths.some((p) => p.startsWith("speech_tokenizer/")) && !ref.path) return { kind: "qwen3", files: paths };
  const tm = paths.filter((p) => /\.tmod$/i.test(p));
  if (tm.length) {
    if (opts.file) { const f = tm.find((p) => baseOf(p) === opts.file || p === opts.file); if (f) return { kind: "tmod", tmod: f, alternatives: tm }; }
    const rank = (p) => { const b = baseOf(p).toLowerCase(); return /q8/.test(b) ? 0 : /bf16/.test(b) ? 1 : /f16/.test(b) ? 2 : 3; };
    const best = tm.slice().sort((a, b) => rank(a) - rank(b) || (a < b ? -1 : 1))[0];
    return { kind: "tmod", tmod: best, alternatives: tm };
  }
  const ck = paths.filter((p) => CKPT_EXT.test(p) && !/(vocos|optimizer|ema_only|\.index\.json)/i.test(p) && !/(^|\/)(pytorch_model|vocos)/i.test(p));
  if (!ck.length) fail("в " + ref.repo + " не нашлось ни .tmod, ни чекпойнта F5-TTS (.safetensors/.pt)", "format");
  // группы по папкам; в каждой нужен vocab.txt
  const groups = new Map();
  for (const p of ck) { const d = dirOf(p); if (!groups.has(d)) groups.set(d, []); groups.get(d).push(p); }
  const hasVocab = (d) => paths.includes((d ? d + "/" : "") + "vocab.txt") || paths.includes("vocab.txt");
  let dirs = Array.from(groups.keys()).filter(hasVocab);
  if (!dirs.length) dirs = Array.from(groups.keys());
  const score = (d) => (/v1[_-]?base/i.test(d) ? 0 : /base/i.test(d) ? 1 : /small/i.test(d) ? 3 : 2);
  dirs.sort((a, b) => score(a) - score(b) || (a < b ? -1 : 1));
  const dir = dirs[0];
  const pick = groups.get(dir).slice().sort((a, b) => {
    const na = Number((/(\d{3,})/.exec(baseOf(a)) || [0, 0])[1]), nb = Number((/(\d{3,})/.exec(baseOf(b)) || [0, 0])[1]);
    const sa = /\.safetensors$/i.test(a) ? 0 : 1, sb = /\.safetensors$/i.test(b) ? 0 : 1;
    return sa - sb || nb - na || (a < b ? -1 : 1);
  })[0];
  return planCkpt(paths, pick, ref, opts, dirs.map((d) => d || "."));
}
function planCkpt(paths, ckpt, ref, opts, alt) {
  const d = dirOf(ckpt);
  const vocab = [(d ? d + "/" : "") + "vocab.txt", "vocab.txt"].find((p) => paths.includes(p)) || paths.find((p) => baseOf(p) === "vocab.txt") || null;
  const vocos = paths.find((p) => /(^|\/)vocos[^/]*\/(pytorch_model\.bin|model\.safetensors)$/i.test(p)) || null;
  return { kind: "f5", ckpt, vocab, vocos, variant: d ? baseOf(d) : ref.repo.split("/")[1], alternatives: alt };
}

// ---------- кэш результата конвертации ----------
const CACHE_NAME = "vox.js/models";
async function cacheGet(key) {
  try {
    if (typeof caches === "undefined") return null;
    const c = await caches.open(CACHE_NAME), r = await c.match("https://vox.cache/" + encodeURIComponent(key));
    return r ? new Uint8Array(await r.arrayBuffer()) : null;
  } catch (e) { return null; }
}
async function cachePut(key, bytes) {
  try {
    if (typeof caches === "undefined") return false;
    const c = await caches.open(CACHE_NAME);
    await c.put("https://vox.cache/" + encodeURIComponent(key), new Response(bytes, { headers: { "content-type": "application/octet-stream" } }));
    return true;
  } catch (e) { return false; }
}
async function cacheClear() { try { return typeof caches !== "undefined" && (await caches.delete(CACHE_NAME)); } catch (e) { return false; } }

// ---------- загрузка файла по любому описанию ----------
async function fetchAny(src, opts, label) {
  const p = opts.onProgress ? (f, a, b) => opts.onProgress(f, a, b, label) : null;
  if (typeof src === "string" && opts.token && (/^https:\/\/([\w-]+\.)?huggingface\.co\//.test(src) || (opts.hfHost && src.startsWith(opts.hfHost.replace(/\/+$/, "") + "/")))) {
    const r = await fetch(src, { headers: hfHeaders(opts) });
    return fetchBytes(r, p);
  }
  return fetchBytes(src, p);
}
async function fetchText(src, opts, label) {
  if (typeof src === "string" && src.includes("\n")) return src;   // это уже сам текст словаря
  return new TextDecoder().decode(await fetchAny(src, opts, label));
}

function guessTokenizer(vocabText) {
  let n = 0;
  for (const l of vocabText.split("\n")) if (/^[a-z]{1,6}[1-5]$/.test(l)) n++;
  return n >= 100 ? "pinyin" : "char";
}
let _pinyinCache = null;
async function loadPinyin(opts) {
  if (opts.pinyin === false) return null;
  if (opts.pinyin && typeof opts.pinyin === "object" && !(opts.pinyin instanceof Uint8Array)) return opts.pinyin;
  if (_pinyinCache) return _pinyinCache;
  const url = typeof opts.pinyin === "string" ? opts.pinyin : (typeof VOX_BASE_URL === "string" && VOX_BASE_URL ? VOX_BASE_URL + "pinyin.json" : null);
  if (!url) return null;
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error("HTTP " + r.status);
    _pinyinCache = await r.json();
    return _pinyinCache;
  } catch (e) {
    if (opts.onStatus) opts.onStatus("таблица пиньиня недоступна (" + url + "): китайский текст не будет озвучиваться");
    return null;
  }
}

// Главная функция: что угодно → байты .tmod
async function acquireTmod(src, opts = {}, keepSource = false) {
  const r = await acquireTmodRaw(src, opts);
  if (r && r.qwenSource && !keepSource) return qwenSourceToTmod(r.qwenSource, (f) => { if (opts.onProgress) opts.onProgress(f, 0, 0, "convert"); });
  return r;
}
async function acquireTmodRaw(src, opts = {}) {
  const status = (s) => { if (opts.onStatus) opts.onStatus(s); };
  const cvtOpts = () => ({ dtype: opts.dtype || "bf16", quant: opts.quant || "", tokenizer: opts.tokenizer, hash: opts.hash, modelCfg: opts.modelCfg, textMaskPadding: opts.textMaskPadding, peAttnHead: opts.peAttnHead, vocosPadding: opts.vocosPadding, vocosDtype: opts.vocosDtype });
  const cacheKeyFor = (parts) => "tmod/3|" + JSON.stringify(parts) + "|" + JSON.stringify(cvtOpts());   // 3: Vocos in f32 since 0.14.1
  const useCache = opts.cache !== false;
  // config.yaml вокодера (ради padding головы ISTFT); не нашёлся — конвертер считает, что это charactr/vocos-mel-24khz (center)
  const vocosConfigAt = async (url) => { try { return await fetchText(url, Object.assign({}, opts, { onProgress: null }), "config.yaml"); } catch (e) { return null; } };
  let vocosConfig = opts.vocosConfig != null ? opts.vocosConfig : null;
  const convertF5From = async (ckptBytes, ckptName, vocabText, vocosBytes, vocosName, variant) => {
    vocabText = splitVocab(vocabText).join("\n") + "\n";
    status("конвертация весов в формат vox (в браузере)…");
    const tokenizer = opts.tokenizer || guessTokenizer(vocabText);
    const pinyin = tokenizer === "pinyin" ? await loadPinyin(opts) : null;
    return convertF5({
      ckpt: ckptBytes, ckptName, vocab: vocabText, vocos: vocosBytes, vocosName, dtype: opts.dtype || "bf16", quant: opts.quant || "",
      tokenizer, variant: opts.variant || variant, vocosConfig, vocosPadding: opts.vocosPadding, vocosDtype: opts.vocosDtype, modelCfg: opts.modelCfg, textMaskPadding: opts.textMaskPadding, peAttnHead: opts.peAttnHead, pinyin, dimHead: opts.dimHead, mel: opts.mel, hash: opts.hash, onWarn: status,
      onProgress: (st, f) => { if (opts.onProgress && st === "convert") opts.onProgress(f, 0, 0, "convert"); },
    });
  };
  const wrapBytes = async (bytes, label, extra) => {
    const kind = sniffKind(bytes);
    if (kind === "tmod") return bytes;
    if (kind === "safetensors" || kind === "torch") {
      const vocabSrc = extra && extra.vocab != null ? extra.vocab : opts.vocab;
      if (vocabSrc == null) fail("это чекпойнт, а не .tmod: укажите словарь (opts.vocab — ссылка, текст или файл vocab.txt)", "arg");
      const vocabText = vocabSrc instanceof Uint8Array || vocabSrc instanceof ArrayBuffer || (typeof Blob !== "undefined" && vocabSrc instanceof Blob)
        ? new TextDecoder().decode(await fetchBytes(vocabSrc)) : await fetchText(vocabSrc, opts, "vocab.txt");
      let vocosBytes;
      const vs = extra && extra.vocos != null ? extra.vocos : opts.vocos;
      if (vs != null) vocosBytes = await fetchAny(typeof vs === "string" && parseHfRef(vs, opts.hfHost) && !/^https?:/.test(vs) ? vs : vs, opts, "vocos");
      else {
        const ref = { repo: DEFAULT_VOCOS, rev: "main", path: "pytorch_model.bin", kind: "file" };
        status("скачиваю вокодер Vocos (" + DEFAULT_VOCOS + ")…");
        vocosBytes = await fetchAny(hfUrl(ref, "pytorch_model.bin", opts), opts, "vocos");
        if (vocosConfig == null && !opts.vocosPadding) vocosConfig = await vocosConfigAt(hfUrl(ref, "config.yaml", opts));
      }
      return convertF5From(bytes, opts.ckptName || label || "model.safetensors", vocabText, vocosBytes, opts.vocosName || "pytorch_model.bin", opts.variant || "f5-tts");
    }
    fail("неизвестный формат файла: ни .tmod, ни safetensors, ни torch", "format");
  };

  // 1) объект {ckpt, vocab, vocos}
  if (src && typeof src === "object" && !(src instanceof Uint8Array) && !(src instanceof ArrayBuffer) && !(typeof Blob !== "undefined" && src instanceof Blob) && !(typeof Response !== "undefined" && src instanceof Response)) {
    if (src.files && typeof src.files === "object" && !src.ckpt) {   // папка Qwen3-TTS: { "config.json": File, "model.safetensors": File, … }
      status("Qwen3-TTS: конвертирую веса по тензорам (в браузере)…");
      return { qwenSource: await openQwenSource(qwenFsFromMap(src.files, opts), src.name || "qwen3-tts", opts) };
    }
    if (!src.ckpt) fail("объект-источник должен содержать ckpt (и vocab, vocos)", "arg");
    status("загрузка чекпойнта…");
    const cb = await fetchAny(src.ckpt, opts, "ckpt");
    return wrapBytes(cb, (src.ckpt && src.ckpt.name) || (typeof src.ckpt === "string" ? baseOf(src.ckpt) : "model.safetensors"), { vocab: src.vocab, vocos: src.vocos });
  }
  // 2) строка: HF или прямая ссылка
  if (typeof src === "string") {
    const ref = parseHfRef(src, opts.hfHost);
    if (ref) {
      const key = cacheKeyFor(["hf", ref.repo, ref.rev, ref.path]);
      if (useCache && !opts.refresh) { const c = await cacheGet(key); if (c) { status("модель найдена в кэше браузера"); return c; } }
      status("Hugging Face: читаю список файлов " + ref.repo + "…");
      const files = await hfList(ref, opts);
      const plan = planFromFiles(files, ref, opts);
      if (opts.onPlan) opts.onPlan(plan);
      let out;
      if (plan.kind === "tmod") {
        status("скачиваю " + plan.tmod + "…");
        out = await fetchAny(hfUrl(ref, plan.tmod, opts), opts, plan.tmod);
      } else if (plan.kind === "f5") {
        if (!plan.vocab) fail("в репозитории нет vocab.txt рядом с чекпойнтом " + plan.ckpt + " (можно передать opts.vocab)", "format");
        status("скачиваю " + plan.ckpt + "…");
        const cb = await fetchAny(hfUrl(ref, plan.ckpt, opts), opts, plan.ckpt);
        const vt = opts.vocab != null ? await fetchText(opts.vocab, opts, "vocab.txt") : await fetchText(hfUrl(ref, plan.vocab, opts), opts, "vocab.txt");
        let vb;
        if (opts.vocos != null) vb = await fetchAny(opts.vocos, opts, "vocos");
        else if (plan.vocos) {
          vb = await fetchAny(hfUrl(ref, plan.vocos, opts), opts, "vocos");
          if (vocosConfig == null && !opts.vocosPadding) vocosConfig = await vocosConfigAt(hfUrl(ref, (dirOf(plan.vocos) ? dirOf(plan.vocos) + "/" : "") + "config.yaml", opts));
        } else {
          status("скачиваю вокодер Vocos (" + DEFAULT_VOCOS + ")…");
          vb = await fetchAny(hfUrl({ repo: DEFAULT_VOCOS, rev: "main" }, "pytorch_model.bin", opts), opts, "vocos");
          if (vocosConfig == null && !opts.vocosPadding) vocosConfig = await vocosConfigAt(hfUrl({ repo: DEFAULT_VOCOS, rev: "main" }, "config.yaml", opts));
        }
        out = await convertF5From(cb, baseOf(plan.ckpt), vt, vb, opts.vocosName || (plan.vocos ? baseOf(plan.vocos) : "pytorch_model.bin"), plan.variant);
      } else if (plan.kind === "qwen3") {
        status("Qwen3-TTS: загружаю и конвертирую веса по тензорам (в браузере)…");
        return { qwenSource: await openQwenSource(qwenFsFromHf(ref, plan.files, opts), ref.repo.split("/")[1], opts) };
      }
      if (useCache) await cachePut(key, out);
      return out;
    }
    // прямая ссылка
    const key = cacheKeyFor(["url", src]);
    if (useCache && !opts.refresh && !/\.tmod(\?|#|$)/i.test(src)) { const c = await cacheGet(key); if (c) { status("модель найдена в кэше браузера"); return c; } }
    status("скачиваю " + src + "…");
    const bytes = await fetchAny(src, opts, baseOf(src.split(/[?#]/)[0]));
    const kind = sniffKind(bytes);
    if (kind === "tmod") return bytes;
    let vocab = opts.vocab;
    if (vocab == null && /^https?:/.test(src)) vocab = src.slice(0, src.lastIndexOf("/") + 1) + "vocab.txt";   // vocab.txt рядом с чекпойнтом
    const out = await wrapBytes(bytes, baseOf(src.split(/[?#]/)[0]), { vocab });
    if (useCache) await cachePut(key, out);
    return out;
  }
  // 3) байты / Blob / File / Response
  const isFile = typeof File !== "undefined" && src instanceof File;
  const bytes = await fetchAny(src, opts, isFile ? src.name : "model");
  return wrapBytes(bytes, isFile ? src.name : "model.safetensors", {});
}

// ============================================================================
// Qwen3-TTS: загрузка чекпойнта HF (safetensors) прямо в браузере. Те же преобразования, что в python/vox/build.py
// (свёртки под im2col, кодбук = embedding_sum / cluster_usage, q8, словарь/merges токенизатора), но без сборки целого .tmod
// в памяти: источник отдаёт тензоры по требованию (HTTP Range по заголовку safetensors), каждый сразу уходит в GPU и забывается.
// Интерфейс — как у Tmod (has / info / raw / f32 / rawAsync / release) плюс prepare / warm.
// ============================================================================
const QWEN_TALKER_DEFAULTS = {
  vocab_size: 3072, hidden_size: 1024, intermediate_size: 2048, num_hidden_layers: 20, num_attention_heads: 16, num_key_value_heads: 2,
  rms_norm_eps: 1e-6, rope_theta: 10000, num_code_groups: 32, text_hidden_size: 2048, text_vocab_size: 151936, codec_eos_token_id: 4198,
  codec_think_id: 4202, codec_nothink_id: 4203, codec_think_bos_id: 4204, codec_think_eos_id: 4205, codec_pad_id: 4196, codec_bos_id: 4197,
};
const QWEN_PRED_DEFAULTS = {
  vocab_size: 2048, hidden_size: 1024, intermediate_size: 3072, num_hidden_layers: 5, num_attention_heads: 16, num_key_value_heads: 8,
  head_dim: 128, rms_norm_eps: 1e-6, rope_theta: 10000, num_code_groups: 32,
};
const QWEN_CODEC_DEFAULTS = {
  codebook_size: 2048, hidden_size: 1024, latent_dim: 1024, rope_theta: 10000, num_attention_heads: 16, num_key_value_heads: 16,
  sliding_window: 72, intermediate_size: 3072, layer_scale_initial_scale: 0.01, rms_norm_eps: 1e-5, num_hidden_layers: 8,
  num_quantizers: 16, upsample_rates: [8, 5, 4, 3], upsampling_ratios: [2, 2], decoder_dim: 1536,
};
const QWEN_IDS_DEFAULTS = { im_start_token_id: 151644, im_end_token_id: 151645, tts_pad_token_id: 151671, tts_bos_token_id: 151672, tts_eos_token_id: 151673 };
const QWEN_SPK_DEFAULTS = { mel_dim: 128, enc_dim: 1024, enc_channels: [512, 512, 512, 512, 1536], enc_kernel_sizes: [5, 3, 3, 3, 1], enc_dilations: [1, 2, 3, 4, 1],
  enc_attention_channels: 128, enc_res2net_scale: 8, enc_se_channels: 128, sample_rate: 24000 };
const QWEN_MIMI_KEYS = ["sampling_rate", "audio_channels", "hidden_size", "num_filters", "num_residual_layers", "upsampling_ratios", "kernel_size", "last_kernel_size",
  "residual_kernel_size", "dilation_growth_rate", "use_causal_conv", "pad_mode", "compress", "codebook_size", "codebook_dim", "num_quantizers", "use_conv_shortcut",
  "vector_quantization_hidden_dimension", "num_semantic_quantizers", "num_hidden_layers", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim",
  "hidden_act", "norm_eps", "rope_theta", "frame_rate"];
const QWEN_MIMI_DEFAULTS = { sampling_rate: 24000, audio_channels: 1, hidden_size: 512, num_filters: 64, num_residual_layers: 1, upsampling_ratios: [8, 6, 5, 4], kernel_size: 7,
  last_kernel_size: 3, residual_kernel_size: 3, dilation_growth_rate: 2, use_causal_conv: true, pad_mode: "constant", compress: 2, codebook_size: 2048, codebook_dim: 256,
  num_quantizers: 32, use_conv_shortcut: false, vector_quantization_hidden_dimension: 256, num_semantic_quantizers: 1, num_hidden_layers: 8, intermediate_size: 2048,
  num_attention_heads: 8, num_key_value_heads: 8, head_dim: null, hidden_act: "gelu", norm_eps: 1e-5, rope_theta: 10000.0, frame_rate: 12.5 };
const QWEN_LINEAR_Q8 = new RegExp("(\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)\\.weight$)|(^talker\\.codec_head\\.weight$)|" +
  "(^talker\\.code_predictor\\.lm_head\\.\\d+\\.weight$)|(^talker\\.code_predictor\\.small_to_mtp_projection\\.weight$)|(^talker\\.text_projection\\.linear_fc[12]\\.weight$)");
const qMerged = (defaults, given) => { const o = Object.assign({}, defaults); for (const k of Object.keys(given || {})) if (given[k] != null) o[k] = given[k]; return o; };

// ---------- чтение диапазонов ----------
class Sem {
  constructor(n) { this.n = n; this.q = []; }
  async run(fn) {
    if (this.n <= 0) await new Promise((r) => this.q.push(r)); else this.n--;
    try { return await fn(); } finally { const nx = this.q.shift(); if (nx) nx(); else this.n++; }
  }
}

// Файл, из которого можно читать куски: байты в памяти, Blob/File или URL с поддержкой Range
class RangeFile {
  constructor(label, size, readFn) { this.label = label; this.size = size; this._read = readFn; }
  async read(a, b) { if (b <= a) return new Uint8Array(0); return this._read(a, b); }
  static fromBytes(label, bytes) { return new RangeFile(label, bytes.length, async (a, b) => bytes.subarray(a, b)); }
  static fromBlob(label, blob) { return new RangeFile(label, blob.size, async (a, b) => new Uint8Array(await blob.slice(a, b).arrayBuffer())); }
  // URL с поддержкой Range. Читает крупными блоками (по умолчанию 8 МиБ, чтобы не платить за запрос на каждый из сотен тензоров:
  // у HF каждый запрос — это ещё и редирект на CDN и предзапрос CORS) и запоминает адрес CDN после первого редиректа.
  static async fromUrl(label, url, opts, tally) {
    opts = opts || {};
    const net = opts._net || (opts._net = { requests: 0, bytes: 0, ms: 0, redirects: 0, cached: 0 });
    const BLOCK = Math.max(1 << 20, (opts.blockMiB || 8) * 1048576), KEEP = Math.max(4, opts.keepBlocks || 12);
    let target = url, whole = null, total = 0, etag = "";
    // Кэш блоков в Cache API: повторная загрузка той же модели не качает гигабайты заново. Ключ включает размер файла и ETag (без них кэш не используется).
    let store = null;
    const cacheKey = (i) => url + (url.includes("?") ? "&" : "?") + "voxblock=" + i + "&bs=" + BLOCK + "&size=" + total + "&etag=" + encodeURIComponent(etag);
    const openStore = async () => { if (store === null) { try { store = opts.cache === false || typeof caches === "undefined" || !total || !etag ? false : await caches.open("vox-blocks-v1"); } catch (e) { store = false; } } return store; };
    const headers = (u, extra) => Object.assign({}, qwenAuth(u, opts), extra || {});
    const fetchRange = async (a, b) => {
      for (let attempt = 0; ; attempt++) {
        const t0 = performance.now();
        let r;
        try { r = await fetch(target, { headers: headers(target, { Range: "bytes=" + a + "-" + (b - 1) }) }); }
        catch (e) {
          if (target !== url) { target = url; continue; }
          if (attempt < 2) { await new Promise((q) => setTimeout(q, 400 * (attempt + 1))); continue; }
          throw e;
        }
        if (target === url && r.redirected && r.url && r.url !== url) { target = r.url; net.redirects++; }   // подписанный адрес CDN живёт около часа
        if ((r.status === 403 || r.status === 400) && target !== url) { target = url; continue; }
        if (r.status === 401 || r.status === 403) fail("Hugging Face: доступ закрыт (нужен токен: opts.token): " + label, "network");
        if (!r.ok) {
          if (attempt < 2 && r.status >= 500) { await new Promise((q) => setTimeout(q, 400 * (attempt + 1))); continue; }
          fail("не удалось скачать " + label + ": HTTP " + r.status, "network");
        }
        const buf = new Uint8Array(await r.arrayBuffer());
        net.requests++; net.bytes += buf.length; net.ms += performance.now() - t0;
        if (!total) { const m = /\/(\d+)\s*$/.exec(r.headers.get("content-range") || ""); if (m) total = Number(m[1]); etag = r.headers.get("etag") || ""; }
        return { buf, partial: r.status === 206 };
      }
    };
    const blocks = new Map();   // индекс блока → Promise<Uint8Array>
    const block = (i) => {
      let p = blocks.get(i);
      if (p) { blocks.delete(i); blocks.set(i, p); return p; }   // обновить порядок (LRU)
      p = (async () => {
        const st = await openStore();
        if (st) {
          try { const hit = await st.match(cacheKey(i)); if (hit) { net.cached++; return new Uint8Array(await hit.arrayBuffer()); } } catch (e) { /* ок */ }
        }
        const { buf, partial } = await fetchRange(i * BLOCK, (i + 1) * BLOCK);
        if (!partial) { whole = buf; return buf.subarray(i * BLOCK, (i + 1) * BLOCK); }   // сервер не понимает Range
        if (st && buf.length) st.put(cacheKey(i), new Response(buf.slice())).catch(() => {});   // в фоне; при нехватке места просто не сохранится
        return buf;
      })();
      blocks.set(i, p);
      p.catch(() => blocks.delete(i));
      while (blocks.size > KEEP) blocks.delete(blocks.keys().next().value);
      return p;
    };
    const get = async (a, b) => {
      if (whole) return whole.subarray(a, b);
      if (b - a >= BLOCK) {   // крупный тензор (эмбеддинги): блоки пачками по четыре, в общий список недавних они не попадают
        const out = new Uint8Array(b - a), j0 = Math.floor(a / BLOCK), j1 = Math.floor((b - 1) / BLOCK);
        for (let j = j0; j <= j1; j += 4) {
          const ids = []; for (let q = j; q <= Math.min(j1, j + 3); q++) ids.push(q);
          const got = await Promise.all(ids.map((q) => block(q)));
          ids.forEach((q, k) => {
            const bl = got[k], lo = Math.max(a, q * BLOCK), hi = Math.min(b, q * BLOCK + bl.length);
            out.set(bl.subarray(lo - q * BLOCK, hi - q * BLOCK), lo - a);
            blocks.delete(q);
          });
          if (whole) return whole.subarray(a, b);
        }
        return out;
      }
      const i0 = Math.floor(a / BLOCK), i1 = Math.floor((b - 1) / BLOCK);
      if (i0 === i1) { const bl = await block(i0); return whole ? whole.subarray(a, b) : bl.subarray(a - i0 * BLOCK, b - i0 * BLOCK); }
      const out = new Uint8Array(b - a);
      for (let i = i0; i <= i1; i++) {
        const bl = await block(i), lo = Math.max(a, i * BLOCK), hi = Math.min(b, i * BLOCK + bl.length);
        out.set(bl.subarray(lo - i * BLOCK, hi - i * BLOCK), lo - a);
      }
      return out;
    };
    const head = (await fetchRange(0, 8)).buf;
    const n = Number(new DataView(head.buffer, head.byteOffset, 8).getBigUint64(0, true));
    const f = new RangeFile(label, 0, get);
    f.headerLen = n; f.net = net;
    return f;
  }
}
function qwenAuth(url, opts) {
  if (!opts || !opts.token) return {};
  const host = (opts.hfHost || HF_HOST).replace(/\/+$/, "");
  return url.startsWith(host + "/") || /^https:\/\/([\w-]+\.)?huggingface\.co\//.test(url) ? { Authorization: "Bearer " + opts.token } : {};
}

// заголовок safetensors из RangeFile → Map имя → { dtype, shape, a, b, file }
async function readStHeader(file) {
  const h8 = await file.read(0, 8);
  const n = Number(new DataView(h8.buffer, h8.byteOffset, 8).getBigUint64(0, true));
  if (n <= 0 || n > 100 * 1024 * 1024) fail("safetensors: странный размер заголовка в " + file.label + " (это не safetensors?)", "format");
  const hdr = JSON.parse(new TextDecoder().decode(await file.read(8, 8 + n)));
  const base = 8 + n, out = new Map();
  for (const k of Object.keys(hdr)) {
    if (k === "__metadata__") continue;
    const t = hdr[k];
    out.set(k, { dtype: t.dtype, shape: t.shape.slice(), a: base + t.data_offsets[0], b: base + t.data_offsets[1], file });
  }
  return out;
}

// ---------- поэлементное преобразование типов без промежуточного f32-массива ----------
const U16 = (raw) => {
  if ((raw.byteOffset & 1) === 0) return new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength >> 1);
  return new Uint16Array(raw.slice().buffer, 0, raw.byteLength >> 1);
};
function convertRaw(raw, src, dst) {
  if (src === dst) return raw;
  const numel = raw.byteLength / ST_SIZE[src];
  if (!ST_SIZE[src] || !ST_SIZE[dst]) fail("неподдерживаемый тип: " + src + " → " + dst, "format");
  if (dst === "F32") {
    const out = new Float32Array(numel);
    if (src === "BF16") { const u = U16(raw), w = new Uint32Array(out.buffer); for (let i = 0; i < numel; i++) w[i] = u[i] << 16; }
    else { const f = rawToF32(raw, src, numel); out.set(f); }
    return new Uint8Array(out.buffer);
  }
  const out = new Uint16Array(numel);
  if (src === "F32") {
    const f = (raw.byteOffset & 3) === 0 ? new Float32Array(raw.buffer, raw.byteOffset, numel) : new Float32Array(raw.slice().buffer, 0, numel);
    if (dst === "F16") for (let i = 0; i < numel; i++) out[i] = f32ToF16Bits(f[i]); else for (let i = 0; i < numel; i++) out[i] = f32ToBf16Bits(f[i]);
  } else {
    const u = U16(raw);
    if (src === "BF16") { for (let i = 0; i < numel; i++) out[i] = f32ToF16Bits(bf16ToF32(u[i])); }   // BF16 → F16
    else { for (let i = 0; i < numel; i++) out[i] = f32ToBf16Bits(f16ToF32(u[i])); }                   // F16 → BF16
  }
  return new Uint8Array(out.buffer);
}
// f32-массив → байты нужного типа (F32/F16/BF16)
function f32ToRaw(f, dst) {
  if (dst === "F32") return new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
  const out = new Uint16Array(f.length);
  if (dst === "F16") for (let i = 0; i < f.length; i++) out[i] = f32ToF16Bits(f[i]); else for (let i = 0; i < f.length; i++) out[i] = f32ToBf16Bits(f[i]);
  return new Uint8Array(out.buffer);
}

// ---------- источник ----------
class QwenHfSource {
  constructor() {
    this.manifest = null; this.table = new Map(); this.pre = new Map();
    this.prom = new Map(); this.done = new Map(); this.sem = new Sem(6);
    this.fetched = 0; this.total = 0; this.onProgress = null;
    this._tally = (n) => { this.fetched += n; if (this.onProgress) this.onProgress(Math.min(1, this.fetched / Math.max(1, this.total)), this.fetched, this.total, "weights"); };
  }
  has(name) { return this.table.has(name); }
  info(name) {
    const e = this.table.get(name);
    if (!e) fail("в чекпойнте нет тензора: " + name, "format");
    return e;
  }
  raw(name) {
    const r = this.pre.get(name);
    if (!r) fail("тензор ещё не загружен (используйте rawAsync): " + name, "internal");
    return r;
  }
  f32(name) {
    const r = this.done.get(name);
    if (!r) fail("тензор ещё не подготовлен: " + name, "internal");
    return new Float32Array(r.buffer, r.byteOffset, r.byteLength >> 2);
  }
  _get(name) {
    if (this.pre.has(name)) return Promise.resolve(this.pre.get(name));
    let p = this.prom.get(name);
    if (!p) {
      const e = this.info(name);
      p = this.sem.run(async () => { const r = await e.produce(this); await _macro(); return r; }).then((r) => { this.done.set(name, r); return r; });   // после каждого тензора страница получает слово (ввод, отрисовка)
      this.prom.set(name, p);
      p.catch(() => {});
    }
    return p;
  }
  async prepare(name) { await this._get(name); }
  warm(names) { for (const n of names) if (this.table.has(n)) this._get(n); }
  async rawAsync(name) {
    if (this.pre.has(name)) return this.pre.get(name);
    const e = this.info(name);
    // у q8-пары главный тензор считает и масштаб; ждём того, кто его произведёт
    const r = await this._get(e.owner || name);
    const out = e.owner ? this.done.get(name) : r;
    if (!out) fail("внутренняя ошибка: нет данных тензора " + name, "internal");
    return out;
  }
  release(name) { this.prom.delete(name); this.done.delete(name); const e = this.table.get(name); if (e && e.companion) { this.prom.delete(e.companion); this.done.delete(e.companion); } }
}

// Собирает QwenHfSource. fs: { has(path), text(path), file(path) → Promise<RangeFile> }
async function openQwenSource(fs, repoName, opts = {}) {
  const status = (s) => { if (opts.onStatus) opts.onStatus(s); };
  const dtype = (opts.dtype || "bf16").toUpperCase(), quant = opts.quant || "", codecDtype = (opts.codecDtype || "f32").toUpperCase();
  if (!["F32", "F16", "BF16"].includes(dtype)) fail("dtype: f32, f16 или bf16", "arg");
  if (!["F32", "F16", "BF16"].includes(codecDtype)) fail("codecDtype: f32, f16 или bf16", "arg");
  if (quant && quant !== "q8") fail("в браузере для Qwen3-TTS доступно квантование q8 (q4 — только в python -m vox.build)", "unsupported");
  const src = new QwenHfSource();
  src.onProgress = opts.onProgress ? (f, a, b, l) => opts.onProgress(f, a, b, l) : null;
  const tally = src._tally;

  status("читаю config.json…");
  const cfg = JSON.parse(await fs.text("config.json"));
  const tcfg = cfg.talker_config || {}, pcfg = tcfg.code_predictor_config || {};
  const tc = qMerged(QWEN_TALKER_DEFAULTS, Object.fromEntries(Object.entries(tcfg).filter(([k]) => k !== "code_predictor_config")));
  const pc = qMerged(QWEN_PRED_DEFAULTS, pcfg);
  if (pc.num_code_groups == null) pc.num_code_groups = tc.num_code_groups;
  if (!tc.head_dim) tc.head_dim = Math.trunc(tc.hidden_size / tc.num_attention_heads);
  tc.codec_language_id = tcfg.codec_language_id || {};
  const scfg = JSON.parse(await fs.text("speech_tokenizer/config.json"));
  const cc = qMerged(QWEN_CODEC_DEFAULTS, scfg.decoder_config || {});
  if (cc.codebook_dim == null) fail("speech_tokenizer/config.json: нет decoder_config.codebook_dim", "format");
  const idm = qMerged(QWEN_IDS_DEFAULTS, cfg);
  const ids = { im_start: idm.im_start_token_id, im_end: idm.im_end_token_id, tts_pad: idm.tts_pad_token_id, tts_bos: idm.tts_bos_token_id, tts_eos: idm.tts_eos_token_id };

  // --- токенизатор (как tokenizer_blobs) ---
  status("токенизатор…");
  const vocab = JSON.parse(await fs.text("vocab.json"));
  const toks = new Map();
  for (const t of Object.keys(vocab)) toks.set(Number(vocab[t]), t);
  const specials = [];
  if (fs.has("tokenizer_config.json")) {
    const added = JSON.parse(await fs.text("tokenizer_config.json")).added_tokens_decoder || {};
    for (const i of Object.keys(added)) { toks.set(Number(i), added[i].content); if (added[i].special) specials.push(added[i].content); }
  }
  const merges = (await fs.text("merges.txt")).split("\n").map((l) => l.replace(/\r$/, "")).filter((l) => l.trim() && !l.startsWith("#version"));
  const vtxt = Array.from(toks.keys()).sort((a, b) => a - b).map((i) => i + "\t" + toks.get(i)).join("\n");
  const enc = new TextEncoder();
  src.pre.set("tokenizer.vocab", enc.encode(vtxt));
  src.pre.set("tokenizer.merges", enc.encode(merges.join("\n")));
  for (const n of ["tokenizer.vocab", "tokenizer.merges"]) src.table.set(n, { dtype: "U8", shape: [src.pre.get(n).length], nbytes: src.pre.get(n).length, offset: 0 });

  // --- заголовки safetensors ---
  status("читаю заголовки safetensors…");
  const mainTensors = new Map();
  if (fs.has("model.safetensors.index.json")) {
    const wm = JSON.parse(await fs.text("model.safetensors.index.json")).weight_map;
    for (const shard of Array.from(new Set(Object.values(wm)))) {
      const f = await fs.file(shard, tally);
      for (const [k, v] of await readStHeader(f)) mainTensors.set(k, v);
    }
  } else {
    for (const [k, v] of await readStHeader(await fs.file("model.safetensors", tally))) mainTensors.set(k, v);
  }
  const codecTensors = await readStHeader(await fs.file("speech_tokenizer/model.safetensors", tally));
  src.fetched = 0;   // чтение заголовков в прогресс не входит
  src.net = fs.net || null;

  // --- описания блобов ---
  let lazy = false;   // тензоры создания голоса: в таблице есть, но в прогресс загрузки не входят и заранее не качаются
  const add = (name, dt, shape, srcBytes, produce, extra) => {
    const nb = shape.reduce((a, c) => a * c, 1) * ({ F32: 4, F16: 2, BF16: 2, I8: 1, U8: 1 }[dt]);
    src.table.set(name, Object.assign({ dtype: dt, shape, nbytes: nb, offset: 0, produce, srcBytes }, extra || {}));
    if (!lazy) src.total += srcBytes;
  };
  const readRaw = async (t) => { const r = await t.file.read(t.a, t.b); tally(r.length); return r; };
  const readF32 = async (t) => rawToF32(await readRaw(t), t.dtype, t.shape.reduce((a, c) => a * c, 1));
  const need = (t, name) => { if (!["F32", "F16", "BF16"].includes(t.dtype)) fail("тензор " + name + ": тип " + t.dtype + " не поддерживается", "format"); };
  const own = (u8) => { if ((u8.byteOffset & 3) === 0) return u8; return u8.slice(); };

  const vec = (name, t) => { need(t, name); add(name, "F32", t.shape, t.b - t.a, async () => own(convertRaw(await readRaw(t), t.dtype, "F32"))); };
  const mat = (name, t, dt, q8) => {
    need(t, name);
    const [rows, cols] = t.shape;
    if (q8 && t.shape.length === 2 && (cols & 31) === 0) {
      const sname = name + ".scale";
      add(name, "I8", [rows, cols], t.b - t.a, async (S) => {
        const w = await readF32(t), { q, s } = quantizeQ8(w, rows, cols);
        const sb = new Uint8Array(s.buffer, s.byteOffset, s.byteLength);
        S.done.set(sname, sb);
        return new Uint8Array(q.buffer, q.byteOffset, q.byteLength);
      }, { companion: sname });
      add(sname, "F32", [rows, cols >> 5], 0, null, { owner: name });
      return;
    }
    add(name, dt, t.shape, t.b - t.a, async () => own(convertRaw(await readRaw(t), t.dtype, dt)));
  };
  const matX = (name, shape, srcBytes, dt, fill) => {   // матрица, считаемая из f32 (преобразования свёрток, кодбуки)
    add(name, dt, shape, srcBytes, async () => own(f32ToRaw(await fill(), dt)));
  };

  // talker.*
  for (const [name, t] of mainTensors) {
    if (!name.startsWith("talker.")) continue;
    if (t.shape.length === 1) vec(name, t);
    else if (name.endsWith("_embedding.weight") || name.includes(".codec_embedding.") || name.includes("text_embedding")) mat(name, t, dtype, false);
    else mat(name, t, dtype, quant === "q8" && QWEN_LINEAR_Q8.test(name));
  }

  // codec: decoder.* из speech_tokenizer/model.safetensors
  const doneCb = new Set();
  for (const [name, t] of codecTensors) {
    if (!name.startsWith("decoder.")) continue;
    const key = name.slice("decoder.".length), out = "codec." + key, shape = t.shape;
    let m;
    if ((m = /quantizer\..*\.vq\.layers\.\d+\._codebook\.(cluster_usage|embedding_sum)$/.exec(key))) {
      const base = key.slice(0, key.lastIndexOf("."));
      if (doneCb.has(base)) continue;
      doneCb.add(base);
      const es = codecTensors.get("decoder." + base + ".embedding_sum"), cu = codecTensors.get("decoder." + base + ".cluster_usage");
      if (!es || !cu) fail("в кодеке нет пары embedding_sum / cluster_usage: " + base, "format");
      need(es, base); need(cu, base);
      const [rows, cols] = es.shape;
      matX("codec." + base + ".embedding", [rows, cols], (es.b - es.a) + (cu.b - cu.a), codecDtype, async () => {
        const e = await readF32(es), u = await readF32(cu);
        for (let r = 0; r < rows; r++) { const d = Math.max(u[r], f32r(1e-5)); for (let c = 0; c < cols; c++) e[r * cols + c] = f32r(e[r * cols + c] / d); }
        return e;
      });
      continue;
    }
    if (/quantizer\..*(input_proj|project_out)/.test(key)) continue;
    need(t, name);
    const bytes = t.b - t.a;
    if (/quantizer\..*output_proj\.weight$/.test(key)) {
      matX(out, [shape[0], shape[1]], bytes, codecDtype, async () => readF32(t));   // [a,b,1] → [a,b]: данные те же
    } else if (/(^upsample\.\d+\.0\.conv\.weight$)|(^decoder\.\d+\.block\.1\.conv\.weight$)/.test(key)) {
      const [cin, cout, K] = shape;   // ConvTranspose1d [cin, cout, K] → [K*cout, cin]
      matX(out, [K * cout, cin], bytes, codecDtype, async () => {
        const w = await readF32(t), o = new Float32Array(w.length);
        for (let ci = 0; ci < cin; ci++) for (let co = 0; co < cout; co++) for (let k = 0; k < K; k++) o[(k * cout + co) * cin + ci] = w[(ci * cout + co) * K + k];
        return o;
      });
    } else if (key.includes("dwconv.conv.weight")) {
      add(out, "F32", [shape[0], shape[2]], bytes, async () => own(convertRaw(await readRaw(t), t.dtype, "F32")));   // [C,1,K] → [C,K]
    } else if (key.endsWith(".weight") && shape.length === 3) {
      const [cout, cin, K] = shape;   // Conv1d [cout, cin, K] → [cout, K*cin]
      matX(out, [cout, K * cin], bytes, codecDtype, async () => {
        const w = await readF32(t), o = new Float32Array(w.length);
        for (let co = 0; co < cout; co++) for (let ci = 0; ci < cin; ci++) for (let k = 0; k < K; k++) o[co * K * cin + k * cin + ci] = w[(co * cin + ci) * K + k];
        return o;
      });
    } else if (shape.length === 1) vec(out, t);
    else if (shape.length === 2 && key.endsWith(".weight")) mat(out, t, codecDtype, false);
    else vec(out, t);
  }

  // создание голоса в браузере: энкодер диктора (ECAPA-TDNN, из основного чекпойнта) и энкодер Mimi (из speech_tokenizer)
  const voice = {};
  const kmajor = (w, cout, cin, K) => { const o = new Float32Array(w.length); for (let co = 0; co < cout; co++) for (let ci = 0; ci < cin; ci++) for (let k = 0; k < K; k++) o[co * K * cin + k * cin + ci] = w[(co * cin + ci) * K + k]; return o; };
  lazy = true;
  let hasSpk = false, hasEnc = false;
  for (const [name, t] of mainTensors) {
    if (!name.startsWith("speaker_encoder.")) continue;
    hasSpk = true;
    const out = "spk." + name.slice("speaker_encoder.".length);
    if (t.shape.length === 3) matX(out, [t.shape[0], t.shape[1] * t.shape[2]], t.b - t.a, "F32", async () => kmajor(await readF32(t), t.shape[0], t.shape[1], t.shape[2]));
    else vec(out, t);
  }
  if (hasSpk) {
    const fb = slaneyMelBasis(24000, 1024, 128, 0, 12000);
    add("spk.mel_basis", "F32", [128, 513], 0, async () => new Uint8Array(fb.buffer));
    voice.speaker_encoder = qMerged(QWEN_SPK_DEFAULTS, cfg.speaker_encoder_config || {});
  }
  const doneEnc = new Set();
  for (const [name, t] of codecTensors) {
    if (!name.startsWith("encoder.")) continue;
    const key = name.slice("encoder.".length);
    if (key.endsWith(".initialized")) continue;
    hasEnc = true;
    const m = /^(quantizer\..*\.layers\.\d+\.codebook)\.(cluster_usage|embed_sum)$/.exec(key);
    if (m) {
      const base = m[1];
      if (doneEnc.has(base)) continue;
      doneEnc.add(base);
      const es = codecTensors.get("encoder." + base + ".embed_sum"), cu = codecTensors.get("encoder." + base + ".cluster_usage");
      if (!es || !cu) fail("в энкодере кодека нет пары embed_sum / cluster_usage: " + base, "format");
      const [rows, cols] = es.shape;
      matX("enc." + base + ".embed", [rows, cols], (es.b - es.a) + (cu.b - cu.a), "F32", async () => {
        const e = await readF32(es), u = await readF32(cu);
        for (let r = 0; r < rows; r++) { const d = Math.max(u[r], f32r(1e-5)); for (let c = 0; c < cols; c++) e[r * cols + c] = f32r(e[r * cols + c] / d); }
        return e;
      });
      continue;
    }
    const out = "enc." + key;
    if (t.shape.length === 3) matX(out, [t.shape[0], t.shape[1] * t.shape[2]], t.b - t.a, "F32", async () => kmajor(await readF32(t), t.shape[0], t.shape[1], t.shape[2]));
    else if (t.shape.length === 2) mat(out, t, "F32", false);
    else vec(out, t);
  }
  if (hasEnc) {
    const ec = qMerged(QWEN_MIMI_DEFAULTS, Object.fromEntries(Object.entries(scfg.encoder_config || {}).filter(([k]) => QWEN_MIMI_KEYS.includes(k))));
    if (!ec.head_dim) ec.head_dim = Math.trunc(ec.hidden_size / ec.num_attention_heads);
    ec.valid_quantizers = scfg.encoder_valid_num_quantizers != null ? scfg.encoder_valid_num_quantizers : tc.num_code_groups;
    ec.downsample_rate = scfg.encode_downsample_rate != null ? scfg.encode_downsample_rate : 1920;
    voice.encoder = ec;
  }
  lazy = false;

  // --- манифест ---
  const gen = {};
  const g = fs.has("generate_config.json") ? JSON.parse(await fs.text("generate_config.json")) : {};
  for (const [a, b] of [["temperature", "temperature"], ["top_k", "top_k"], ["top_p", "top_p"], ["repetition_penalty", "repetition_penalty"],
    ["subtalker_temperature", "sub_temperature"], ["subtalker_top_k", "sub_top_k"], ["subtalker_top_p", "sub_top_p"], ["max_new_tokens", "max_frames"]]) if (a in g) gen[b] = g[a];
  if (gen.chunk_frames == null) gen.chunk_frames = 4;
  const sr = scfg.output_sample_rate || 24000;
  const spf = [...cc.upsample_rates, ...cc.upsampling_ratios].reduce((a, b) => a * b, 1);
  const variant = ["tokenizer_type", "tts_model_size", "tts_model_type"].filter((k) => cfg[k]).map((k) => String(cfg[k])).join("-").toLowerCase() || repoName;
  src.manifest = {
    format: "1.0",
    model: { family: "qwen3-tts", variant, type: cfg.tts_model_type || "", weights_sha256: "" },
    target: { backend: "cpu", arch: "any", features: [] },
    audio: { sample_rate: sr, samples_per_frame: spf, frame_rate: sr / spf, num_codebooks: tc.num_code_groups, codebook_size: pc.vocab_size },
    stages: [{ name: "talker", kind: "autoregressive" }, { name: "code_predictor", kind: "subloop" }, { name: "decoder", kind: "streaming_conv", context: "left_only" }],
    precision: { weights: dtype.toLowerCase(), quant: quant || "none", codec: codecDtype.toLowerCase() },
    generation: gen,
    tokenizer: { specials },
    qwen3: Object.assign({ talker: tc, predictor: pc, codec: cc, ids }, voice),
    blobs: {},
  };
  for (const [n, e] of src.table) src.manifest.blobs[n] = { dtype: e.dtype, shape: e.shape, offset: 0, nbytes: e.nbytes };
  return src;
}

// ---------- файловые системы для openQwenSource ----------
// репозиторий HF (или совместимый хост); paths — список файлов репозитория
function qwenFsFromHf(ref, paths, opts) {
  opts = Object.assign({}, opts, { _net: { requests: 0, bytes: 0, ms: 0, redirects: 0, cached: 0 } });
  const set = new Set(paths), cache = new Map();
  return {
    net: opts._net,
    has: (p) => set.has(p),
    async text(p) {
      if (!set.has(p)) fail("в репозитории " + ref.repo + " нет файла " + p + ": это не чекпойнт Qwen3-TTS?", "format");
      const url = hfUrl(ref, p, opts), r = await fetch(url, { headers: hfHeaders(opts) });
      if (r.status === 401 || r.status === 403) fail("Hugging Face: доступ к " + ref.repo + " закрыт (нужен токен: opts.token)", "network");
      if (!r.ok) fail("не удалось скачать " + p + ": HTTP " + r.status, "network");
      return r.text();
    },
    async file(p, tally) {
      if (!set.has(p)) fail("в репозитории " + ref.repo + " нет файла " + p, "format");
      if (!cache.has(p)) cache.set(p, RangeFile.fromUrl(p, hfUrl(ref, p, opts), opts, tally));
      return cache.get(p);
    },
  };
}
// словарь { "config.json": File | Blob | Uint8Array | строка-URL, "speech_tokenizer/model.safetensors": … }
function qwenFsFromMap(map, opts) {
  opts = Object.assign({}, opts, { _net: { requests: 0, bytes: 0, ms: 0, redirects: 0, cached: 0 } });
  const get = (p) => map[p];
  const net = opts._net;
  const toBytes = async (v) => (typeof v === "string" && !/^https?:|^\/|^\.\//.test(v) ? new TextEncoder().encode(v)
    : v instanceof Uint8Array ? v : typeof Blob !== "undefined" && v instanceof Blob ? new Uint8Array(await v.arrayBuffer()) : fetchBytes(v));
  return {
    net,
    has: (p) => get(p) != null,
    async text(p) { if (get(p) == null) fail("нет файла " + p, "format"); return new TextDecoder().decode(await toBytes(get(p))); },
    async file(p, tally) {
      const v = get(p);
      if (v == null) fail("нет файла " + p, "format");
      if (v instanceof Uint8Array) return RangeFile.fromBytes(p, v);
      if (typeof Blob !== "undefined" && v instanceof Blob) return RangeFile.fromBlob(p, v);
      if (typeof v === "string") return RangeFile.fromUrl(p, v, opts, tally);
      fail("неподдерживаемый источник для " + p, "arg");
    },
  };
}

// Весь источник → байты .tmod (для Vox.convert: сохранить и отдавать файлом). Нужна память под всю модель.
async function qwenSourceToTmod(src, onProgress) {
  const blobs = [];
  for (const [name, e] of src.table) {
    blobs.push({ name, dtype: e.dtype, shape: e.shape.slice(), async fill(u8) { u8.set(await src.rawAsync(name)); if (!src.pre.has(name)) src.release(name); } });
  }
  // масштаб q8 берётся из пары главного тензора: порядок заполнения — по имени, ".scale" идёт после основного
  const m = Object.assign({}, src.manifest);
  delete m.blobs;
  return writeTmod(m, blobs, onProgress);
}

// Банк mel-фильтров Slaney (как slaney_mel_basis в python/vox/build.py и librosa.filters.mel(htk=False, norm="slaney"))
function slaneyMelBasis(sr, nFft, nMels, fMin, fMax) {
  const fSp = 200.0 / 3, minLogHz = 1000.0, minLogMel = 15.0, logstep = Math.log(6.4) / 27.0;
  const h2m = (f) => (f >= minLogHz ? minLogMel + Math.log(Math.max(f, 1e-10) / minLogHz) / logstep : f / fSp);
  const m2h = (m) => (m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fSp * m);
  const nb = 1 + (nFft >> 1);
  const lin = (a, b, n) => { const o = new Float64Array(n), st = (b - a) / (n - 1); for (let i = 0; i < n; i++) o[i] = a + i * st; o[n - 1] = b; return o; };
  const fft = lin(0, sr / 2.0, nb);
  const melPts = lin(h2m(fMin), h2m(fMax), nMels + 2), melF = Array.from(melPts, m2h);
  const w = new Float32Array(nMels * nb);
  for (let i = 0; i < nMels; i++) {
    const d0 = melF[i + 1] - melF[i], d1 = melF[i + 2] - melF[i + 1], enorm = 2.0 / (melF[i + 2] - melF[i]);
    for (let k = 0; k < nb; k++) {
      const lower = -(melF[i] - fft[k]) / d0, upper = (melF[i + 2] - fft[k]) / d1;
      w[i * nb + k] = Math.max(0, Math.min(lower, upper)) * enorm;
    }
  }
  return w;
}

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

// Склейка кусков кроссфейдом: в хвост уходят последние cfMax отсчётов, остальное можно отдавать. Те же формулы, что synth_chunk в ядре.
class Splicer {
  constructor(cfMax) { this.cfMax = cfMax; this.tail = new Float32Array(0); this.fadeIn = false; }
  push(wave, first) {
    const cfMax = this.cfMax;
    if (this.fadeIn && cfMax > 0) {   // после паузы между отрезками: нарастание вместо кроссфейда с предыдущим
      const n = Math.min(cfMax, wave.length);
      for (let i = 0; i < n; i++) wave[i] *= i / n;
      this.fadeIn = false;
    }
    if (cfMax === 0) return wave;
    const tail = this.tail;
    let buf;
    if (tail.length === 0 && first) buf = wave;
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
    this.tail = buf.slice(buf.length - keep);
    return buf.subarray(0, buf.length - keep);
  }
  // Отдать хвост. fade=true — с затуханием до нуля (отрезок закончился, продолжения пока нет)
  release(fade) {
    const t = this.tail;
    this.tail = new Float32Array(0);
    if (fade && t.length) { const n = t.length; for (let i = 0; i < n; i++) t[i] *= 1 - (i + 1) / n; this.fadeIn = true; }
    return t;
  }
}

class VoxModel {
  constructor(gpu, tm, f5) {
    this.gpu = gpu; this.f5 = f5;
    this.manifest = tm.manifest;
    this.sampleRate = f5.cfg.sample_rate;
    this.generation = Object.assign({}, DEFAULTS, tm.manifest.generation || {});
    const m = tm.manifest.model || {};
    this.info = {
      name: m.name || "", family: m.family || "f5-tts", variant: m.variant || (tm.manifest.f5 && tm.manifest.f5.variant) || "",
      sampleRate: this.sampleRate, weightBytes: f5.wbytes, device: gpu.info, backend: gpu.isCpu ? "cpu" : "webgpu", precision: tm.manifest.precision || {},
      f5: { text_mask_padding: f5.cfg.mask_padding, pe_attn_head: f5.cfg.pe_attn_head || null },
    };
  }

  static async load(src, opts = {}) {
    const t0 = performance.now();
    const bytes = await acquireTmod(src, opts, true);
    const tm = bytes && bytes.qwenSource ? bytes.qwenSource : new Tmod(bytes);
    const fam = (tm.manifest.model || {}).family;
    if (fam && fam !== "f5-tts" && fam !== "qwen3-tts") fail("vox.js умеет семейства f5-tts и qwen3-tts, а в файле: " + fam, "unsupported");
    if (opts.onStatus) opts.onStatus("выбор устройства (WebGPU или процессор)…");
    const gpu = opts.gpu || await createDevice(opts);
    if (opts.onStatus) opts.onStatus((gpu.isCpu ? "подготовка весов для процессора…" : "загрузка весов в GPU…"));
    if (fam === "qwen3-tts") {
      const q = await QwenGpu.create(gpu, tm);
      const qm = new QwenModel(gpu, tm, q);
      qm.info.loadMs = performance.now() - t0;
      if (tm.net) qm.info.net = { requests: tm.net.requests, mib: +(tm.net.bytes / 1048576).toFixed(1), redirects: tm.net.redirects, cachedBlocks: tm.net.cached, netMs: Math.round(tm.net.ms), mibPerSec: tm.net.ms ? +(tm.net.bytes / 1048576 / (tm.net.ms / 1000)).toFixed(1) : 0 };
      return qm;
    }
    if (gpu.precompile) {
      const k = tm.manifest.f5 || {}, mf = tm.manifest.precision || {};
      const wt = mf.quant === "q8" ? [0, 1, 2, 3] : mf.quant === "q4" ? [0, 1, 2, 4] : [0, 1, 2];
      const L = [["attention", "main", { HD: k.dim_head }], ["rope", "main"], ["conv1d", "main"], ["grn", "sumsq"], ["grn", "stats"], ["grn", "apply"], ["ln", "main"]];
      for (const w of wt) L.push(["gemm", "main", { WTYPE: w }]);
      for (const e of ["concat_in", "euler", "euler1", "zero_rows"]) L.push(["ew", e]);
      await gpu.precompile(L);
    }
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

  // Параметры нарезки для голоса: нормированный текст образца, длина образца в кадрах, предел символов куска, длина кроссфейда.
  _plan(voice, s) {
    const k = this.f5.cfg;
    let rt = voice.refText;
    if (!rt.endsWith(". ") && !rt.endsWith("。")) rt += rt.endsWith(".") ? " " : ". ";
    const refAudioLen = Math.floor(voice.samples / k.mel.hop);
    const refSec = f32r(voice.samples / k.sample_rate);
    let maxChars = s.maxChars;
    if (maxChars <= 0) maxChars = Math.max(1, Math.trunc(blen(rt) / refSec * (22 - refSec) * s.speed));
    const cfMax = Math.floor(f32r(Math.max(0, s.crossFade) * k.sample_rate));
    return { rt, refAudioLen, refSec, maxChars, cfMax };
  }

  // Один кусок текста → звук (mel → Vocos → выравнивание громкости). idx — сквозной номер куска (поток шума).
  async _chunkWave(voice, s, P, gen, idx, opts) {
    const k = this.f5.cfg;
    let ref = P.rt;
    if (ref.charCodeAt(ref.length - 1) < 0x80) ref += " ";
    const ids = this.f5.tok.encode(ref + gen);
    const localSpeed = blen(gen) < 10 ? f32r(0.3) : s.speed;
    let dur = P.refAudioLen + Math.trunc(P.refAudioLen / blen(ref) * blen(gen) / localSpeed);
    dur = Math.max(dur, Math.max(ids.length, voice.frames) + 1);
    dur = Math.min(dur, 65536);
    let noise;
    if (opts.noise) noise = typeof opts.noise === "function" ? opts.noise(idx, dur * k.mel_dim) : opts.noise[idx];
    if (!noise) noise = f5Noise(s.seed, idx, dur * k.mel_dim);
    if (noise.length < dur * k.mel_dim) fail("переданный шум короче нужного для куска " + idx, "arg");
    const mel = await this.f5.sample(voice.mel, voice.frames, ids, dur, { nfe: s.nfe, cfg: s.cfg, sway: s.sway, epss: s.epss }, noise,
      { signal: opts.signal, onStep: opts.onStep ? (a, b) => opts.onStep({ chunk: idx, chunks: opts.chunksTotal || 0, step: a, steps: b }) : null });
    const from = P.refAudioLen;
    const wave = await this.f5.vocode(mel.subarray(from * k.mel_dim), dur - from);
    if (voice.rms < s.targetRms && voice.rms > 0) { const g = f32r(voice.rms / s.targetRms); for (let i = 0; i < wave.length; i++) wave[i] *= g; }
    return wave;
  }

  async *stream(text, opts = {}) {
    const voice = opts.voice;
    if (!(voice instanceof Voice)) fail("передайте голос: opts.voice = await model.cloneVoice(...)", "arg");
    if (voice.melDim !== this.f5.cfg.mel_dim) fail("голос создан для другой модели", "arg");
    if (typeof text !== "string" || !text.trim()) fail("пустой текст", "arg");
    const s = this._opts(opts), k = this.f5.cfg;
    const P = this._plan(voice, s);
    const chunks = chunkText(text, P.maxChars);
    const sp = new Splicer(P.cfMax);
    const t0 = performance.now();
    this.f5.timing = { dit_ms: 0, vocos_ms: 0, text_ms: 0 };   // статистика — за этот запуск
    let produced = 0;
    for (let idx = 0; idx < chunks.length; idx++) {
      if (opts.signal && opts.signal.aborted) fail("синтез отменён", "aborted");
      const wave = await this._chunkWave(voice, s, P, chunks[idx], idx, Object.assign({}, opts, { chunksTotal: chunks.length }));
      const piece = sp.push(wave, idx === 0);
      produced += piece.length;
      if (piece.length) yield piece;
    }
    const tail = sp.release(false);
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

// ============================================================================
// Qwen3-TTS в браузере: голос (.tvoice / готовый диктор / описание), сессия генерации (порт engine.cpp + backend_qwen3.cpp),
// детерминированный сэмплинг (Philox), потоковая выдача и адаптер для двустороннего стриминга.
// ============================================================================
const QWEN_DEFAULTS = { temperature: 0.9, top_p: 1.0, top_k: 50, repetition_penalty: 1.05, sub_temperature: 0.9, sub_top_p: 1.0, sub_top_k: 50, max_frames: 2048, chunk_frames: 4 };

const QWEN_ISO = { zh: "chinese", en: "english", ru: "russian", de: "german", fr: "french", es: "spanish", it: "italian", pt: "portuguese", ja: "japanese", ko: "korean" };
function qwenLanguage(l) { l = String(l || "auto").toLowerCase(); return QWEN_ISO[l] || l; }

// Голос Qwen3-TTS: клон (icl — с кодами образца, xvector_only — только вектор диктора), готовый диктор или описание голоса.
class QVoice {
  constructor(d) { Object.assign(this, { mode: "icl", name: "", speaker: "", instruct: "", refText: "", language: "auto", sampling: {}, speakerEmb: new Float32Array(0), refCodes: null, refFrames: 0, refCodebooks: 0, refTextTokens: null }, d); }
  // .tvoice (safetensors, создаётся `vox voice create` или Python) → голос
  static from(bytes) {
    if (bytes instanceof ArrayBuffer) bytes = new Uint8Array(bytes);
    const t = parseSafetensors(bytes), meta = t.metadata || {};
    if (!meta.voice || !meta.format) fail("это не файл .tvoice (нет метаданных)", "format");
    if (String(meta.format)[0] !== "1") fail("формат .tvoice " + meta.format + " не поддерживается этой сборкой", "format");
    const j = JSON.parse(meta.voice);
    const mode = j.mode === "xvector_only" ? "xvector_only" : j.mode === "speaker" ? "speaker" : "icl";
    const take = (name, Ctor, dtype) => {
      const e = t.get(name);
      if (!e) fail(".tvoice: нет тензора " + name, "format");
      if (e.dtype !== dtype) fail(".tvoice: " + name + " должен быть " + dtype, "format");
      const c = e.bytes.slice();
      return { a: new Ctor(c.buffer, 0, c.byteLength / Ctor.BYTES_PER_ELEMENT), shape: e.shape };
    };
    const v = new QVoice({ mode, name: j.name || "voice", speaker: j.speaker || "", instruct: j.instruct || "", refText: j.ref_text || "", language: j.language || "auto",
      sampling: j.sampling || {}, model: j.model || {}, speakerEmb: take("speaker_emb", Float32Array, "F32").a });
    if (mode === "icl") {
      const c = take("ref_codes", Int16Array, "I16");
      v.refCodes = c.a; v.refFrames = c.shape[0]; v.refCodebooks = c.shape[1];
      v.refTextTokens = take("ref_text_tokens", Int32Array, "I32").a;
    }
    return v;
  }
  // Байты .tvoice (safetensors; тот же формат, что у `vox voice create`) — голос можно сохранить и открыть в любой сборке vox
  toTvoice() {
    const te = new TextEncoder();
    const meta = { format: "1.0", voice: JSON.stringify({
      format: "1.0", name: this.name || "voice", mode: this.mode, ref_text: this.refText || "", speaker: this.speaker || "", instruct: this.instruct || "", language: this.language || "auto",
      model: this.model || { family: "qwen3-tts", variant: "", weights_sha256: "" }, sampling: this.sampling || {}, kv_fingerprints: [], speaker_emb_dim: this.speakerEmb.length,
      ref_frames: this.refFrames || 0, ref_codebooks: this.refCodebooks || 0, has_ref_audio: false, features: [], created: new Date().toISOString().slice(0, 10) }) };
    const ts = [["speaker_emb", "F32", [this.speakerEmb.length], new Uint8Array(this.speakerEmb.buffer, this.speakerEmb.byteOffset, this.speakerEmb.byteLength)]];
    if (this.mode === "icl" && this.refFrames > 0) {
      ts.push(["ref_codes", "I16", [this.refFrames, this.refCodebooks], new Uint8Array(this.refCodes.buffer, this.refCodes.byteOffset, this.refCodes.byteLength)]);
      ts.push(["ref_text_tokens", "I32", [this.refTextTokens.length], new Uint8Array(this.refTextTokens.buffer, this.refTextTokens.byteOffset, this.refTextTokens.byteLength)]);
    }
    const hdr = { __metadata__: meta };
    let off = 0;
    for (const [n, dt, shape, b] of ts) { hdr[n] = { dtype: dt, shape, data_offsets: [off, off + b.length] }; off += b.length; }
    let hb = te.encode(JSON.stringify(hdr));
    const padn = (8 - (hb.length % 8)) % 8;
    if (padn) { const t = new Uint8Array(hb.length + padn); t.set(hb); t.fill(32, hb.length); hb = t; }
    const out = new Uint8Array(8 + hb.length + off);
    new DataView(out.buffer).setBigUint64(0, BigInt(hb.length), true);
    out.set(hb, 8);
    let o = 8 + hb.length;
    for (const [, , , b] of ts) { out.set(b, o); o += b.length; }
    return out;
  }
}

// ---------- сэмплинг (порт sampling.hpp) ----------
function qUniform(k0, k1, frame, cb) {
  return (philoxBlock(frame >>> 0, Math.floor(frame / 4294967296) >>> 0, cb >>> 0, 0, k0, k1) >>> 8) * (1 / 16777216);
}
// logits: Float32Array (может быть изменён: штраф за повтор). p: { temperature, top_p, top_k, repetition_penalty } (float32-значения).
function qSample(logits, p, u, history) {
  const n = logits.length;
  if (!n) return -1;
  if (history && p.repetition_penalty !== 1) {
    const seen = new Uint8Array(n);
    for (const t of history) {
      if (t >= 0 && t < n && !seen[t]) { seen[t] = 1; const l = logits[t]; logits[t] = l > 0 ? l / p.repetition_penalty : l * p.repetition_penalty; }
    }
  }
  if (p.temperature <= 0) {
    let best = 0;
    for (let i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  const k = p.top_k > 0 && p.top_k < n ? p.top_k : n;
  // k лучших по (логит убыв., индекс возр.)
  let idx;
  if (k === n) { idx = Array.from({ length: n }, (_, i) => i); idx.sort((a, b) => (logits[a] > logits[b] ? -1 : logits[a] < logits[b] ? 1 : a - b)); }
  else {
    idx = [];
    const better = (a, b) => logits[a] > logits[b] || (logits[a] === logits[b] && a < b);
    for (let i = 0; i < n; i++) {
      if (idx.length === k && !better(i, idx[k - 1])) continue;
      let lo = 0, hi = idx.length;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (better(idx[mid], i)) lo = mid + 1; else hi = mid; }
      idx.splice(lo, 0, i);
      if (idx.length > k) idx.pop();
    }
  }
  const invT = 1 / p.temperature, mx = logits[idx[0]];
  const probs = new Float64Array(idx.length);
  let sum = 0;
  for (let i = 0; i < idx.length; i++) { probs[i] = Math.exp((logits[idx[i]] - mx) * invT); sum += probs[i]; }
  for (let i = 0; i < probs.length; i++) probs[i] /= sum;
  let keep = probs.length;
  if (p.top_p > 0 && p.top_p < 1) {
    let c = 0;
    for (let i = 0; i < probs.length; i++) { c += probs[i]; if (c >= p.top_p) { keep = i + 1; break; } }
  }
  let total = 0;
  for (let i = 0; i < keep; i++) total += probs[i];
  const target = u * total;
  let acc = 0;
  for (let i = 0; i < keep; i++) { acc += probs[i]; if (target < acc) return idx[i]; }
  return idx[keep - 1];
}

// ---------- сессия: один текст → поток кадров → поток звука ----------
class QwenSession {
  constructor(model, voice, opts = {}) {
    const q = model.q, tc = q.tc, g = model.generation;
    this.model = model; this.q = q; this.voice = voice; this.opts = opts;
    const f = Math.fround;
    const pick = (key, ...alts) => { for (const a of [opts[key], voice.sampling && voice.sampling[key], g[key]]) if (a != null) return Number(a); return QWEN_DEFAULTS[key]; };
    this.talker = { temperature: f(pick("temperature")), top_p: f(pick("top_p")), top_k: Math.trunc(pick("top_k")), repetition_penalty: f(pick("repetition_penalty")) };
    this.sub = { temperature: f(pick("sub_temperature")), top_p: f(pick("sub_top_p")), top_k: Math.trunc(pick("sub_top_k")), repetition_penalty: 1 };
    let seed = Math.trunc(opts.seed != null ? opts.seed : (voice.sampling && voice.sampling.seed) || 0);
    if (!seed) seed = 1;
    this.k0 = seed >>> 0; this.k1 = Math.floor(seed / 4294967296) >>> 0;
    this.chunk = Math.max(1, Math.trunc(pick("chunk_frames")));
    this.maxFrames = Math.max(1, Math.trunc(pick("max_frames")));
    this.signal = opts.signal || null;
    this.noEos = !!opts.noEos;
    this.minEos = 2;
    this.suppressFrom = Math.max(0, tc.vocab - 1024); this.suppressTo = tc.vocab;
    this.textStream = []; this.textN = 0;
    this.prefilled = false; this.framesDone = 0; this.frame = 0;
    this.pending = null; this.codes = new Int32Array(tc.groups); this.flat = []; this.buffered = 0; this.history0 = [];
    this.finished = false; this.flushed = false; this.textEnded = false;
    this.ts = null; this.cs = null; this.warmed = false;
    this.stats = { prefill_ms: 0, talker_ms: 0, predictor_ms: 0, codec_ms: 0, frames: 0, prefill_rows: 0 };
    this.ctxFrames = Math.trunc(opts.decoder_context_frames || g.decoder_context_frames || 50);
  }

  _abortCheck() { if (this.signal && this.signal.aborted) fail("синтез отменён", "aborted"); }

  // begin(): префикс (роль + кодек-строки [+ ICL])
  async init() {
    const q = this.q, tc = q.tc, D = tc.hidden, v = this.voice, model = this.model;
    const named = v.mode === "speaker";
    if (!named && v.speakerEmb.length !== D) fail("вектор диктора в голосе имеет размер " + v.speakerEmb.length + ", а модель ждёт " + D, "format");
    const sp = await q.textProject([q.ids.tts_bos, q.ids.tts_eos, q.ids.tts_pad]);
    this.bos = sp.slice(0, D); this.eos = sp.slice(D, 2 * D); this.pad = sp.slice(2 * D, 3 * D);
    let lang = qwenLanguage(this.opts.language || v.language), langId = -1;
    if (lang !== "auto") {
      if (tc.lang[lang] === undefined) fail("язык '" + lang + "' не поддерживается этой моделью (есть: " + Object.keys(tc.lang).join(", ") + ")", "arg");
      langId = tc.lang[lang];
    }
    let spkRow = -1;
    if (named && v.speaker) {
      const key = v.speaker.toLowerCase();
      if (tc.spk_id[key] === undefined) fail("диктора '" + v.speaker + "' нет в модели", "arg");
      spkRow = tc.spk_id[key];
      const dl = tc.spk_dialect[key];
      if ((lang === "chinese" || lang === "auto") && typeof dl === "string") {
        if (tc.lang[dl] === undefined) fail("диалект '" + dl + "' диктора '" + v.speaker + "' не имеет id языка", "format");
        langId = tc.lang[dl];
      }
    }
    const prefix = langId < 0 ? [tc.nothink, tc.think_bos, tc.think_eos] : [tc.think, tc.think_bos, langId, tc.think_eos];
    const hasSpk = !named || spkRow >= 0;
    const C = prefix.length + (hasSpk ? 3 : 2);
    const cie = new Float32Array(C * D);
    let r = 0;
    for (const id of prefix) q.codecRow(id, cie.subarray(r++ * D, r * D));
    if (named) { if (spkRow >= 0) q.codecRow(spkRow, cie.subarray(r++ * D, (r) * D)); }
    else { cie.set(v.speakerEmb, r * D); r++; }
    q.codecRow(tc.pad, cie.subarray(r++ * D, r * D));
    q.codecRow(tc.bos, cie.subarray(r++ * D, r * D));
    // инструкция
    let ins = [];
    if (named && v.instruct) ins = model.tok.encode("<|im_start|>user\n" + v.instruct + "<|im_end|>\n");
    const I = ins.length;
    const role = model.tok.encode("<|im_start|>assistant\n");
    if (role.length !== 3) fail("префикс роли должен быть из 3 токенов", "format");
    this.headRows = I + 3 + (C - 1);
    this.head = new Float32Array(this.headRows * D);
    const proj = await q.textProject(ins.concat(role));
    this.head.set(proj, 0);
    for (let i = 0; i < C - 1; i++) {
      const t = i < C - 2 ? this.pad : this.bos, o = (I + 3 + i) * D;
      for (let d = 0; d < D; d++) this.head[o + d] = t[d] + cie[i * D + d];
    }
    if (v.mode === "icl") {
      if (!v.refCodes || v.refCodebooks !== tc.groups) fail("голос ICL должен содержать коды образца с " + tc.groups + " кодбуками", "format");
      if (!v.refTextTokens || v.refTextTokens.length < 6) fail("в голосе нет токенов текста образца", "format");
      this.refCodes = v.refCodes; this.refFrames = v.refFrames;
      this.codecLens = 1 + v.refFrames;
      this.codecPart = new Float32Array(this.codecLens * D);
      q.codecRow(tc.bos, this.codecPart.subarray(0, D));
      const tmp = new Float32Array(D);
      for (let f = 0; f < v.refFrames; f++) {
        const dst = (1 + f) * D;
        for (let g = 0; g < tc.groups; g++) {
          const id = v.refCodes[f * v.refCodebooks + g];
          if (g === 0) q.codecRow(id, tmp); else q.row(q.predEmb[g - 1], id, tmp);
          for (let d = 0; d < D; d++) this.codecPart[dst + d] += tmp[d];
        }
      }
      await this.appendText(Array.from(v.refTextTokens.subarray(3, v.refTextTokens.length - 2)));
    } else {
      this.refFrames = 0; this.refCodes = null;
      this.codecLens = 1;
      this.codecPart = new Float32Array(D);
      q.codecRow(tc.bos, this.codecPart);
    }
    return this;
  }

  async appendText(tokens) {
    if (!tokens.length) return;
    const D = this.q.tc.hidden, proj = await this.q.textProject(tokens);
    for (let i = 0; i < tokens.length; i++) this.textStream.push(proj.subarray(i * D, (i + 1) * D));
    this.textN += tokens.length;
  }
  needsText() {
    if (this.textEnded) return false;
    if (!this.prefilled) return this.textN < this.codecLens;
    return this.textN < this.codecLens + this.framesDone;
  }
  _elem(idx) {
    if (idx < this.textN) return this.textStream[idx];
    if (this.textEnded && idx === this.textN) return this.eos;
    return this.pad;
  }

  async _talkerLogits() {
    const q = this.q, D = q.tc.hidden, t0 = performance.now();
    let logits;
    if (!this.prefilled) {
      if (this.textEnded && this.codecLens === 1 && this.textN === 0) fail("пустой текст", "arg");
      const rows = this.headRows + this.codecLens, x = new Float32Array(rows * D);
      x.set(this.head);
      for (let j = 0; j < this.codecLens; j++) {
        const t = this._elem(j), o = (this.headRows + j) * D;
        for (let d = 0; d < D; d++) x[o + d] = t[d] + this.codecPart[j * D + d];
      }
      const cap = rows + this.maxFrames + 16;
      this.ts = q.newTalkerState(cap);
      logits = await q.talkerStep(this.ts, x, rows);
      this.prefilled = true;
      this.stats.prefill_ms += performance.now() - t0; this.stats.prefill_rows = rows;
    } else {
      const t = this._elem(this.codecLens + this.framesDone - 1), x = new Float32Array(D);
      for (let d = 0; d < D; d++) x[d] = this.pending[d] + t[d];
      logits = await q.talkerStep(this.ts, x, 1);
      this.stats.talker_ms += performance.now() - t0;
    }
    return logits;
  }

  async generate() {
    const q = this.q, tc = q.tc, G = tc.groups, D = tc.hidden;
    while (this.buffered < this.chunk) {
      this._abortCheck();
      if (this.needsText()) return;
      if (this.frame >= this.maxFrames) { this.finished = true; return; }
      const logits = await this._talkerLogits();
      const force = this.opts.force;   // тесты: подставить эталонные коды (teacher forcing) и записать логиты шагов
      if (this.opts.trace) this.opts.trace.talker.push(logits.slice());
      if (force && this.frame >= force.length / G) { this.finished = true; return; }
      for (let i = this.suppressFrom; i < this.suppressTo && i < logits.length; i++) if (i !== tc.eos) logits[i] = -Infinity;
      if ((this.noEos || this.frame < this.minEos) && tc.eos >= 0 && tc.eos < logits.length) logits[tc.eos] = -Infinity;
      const c0 = force ? force[this.frame * G] : qSample(logits, this.talker, qUniform(this.k0, this.k1, this.frame, 0), this.history0);
      if (c0 === tc.eos) { this.finished = true; return; }
      this.history0.push(c0);
      this.codes[0] = c0;
      this.flat.push(c0);
      const t0 = performance.now();
      for (let cb = 1; cb < G; cb++) {
        const lg = await q.predictorLogits(this.ts, cb, cb === 1 ? c0 : this.codes[cb - 1]);
        if (this.opts.trace) this.opts.trace.pred.push(lg.slice());
        const c = force ? force[this.frame * G + cb] : qSample(lg, this.sub, qUniform(this.k0, this.k1, this.frame, cb), null);
        this.codes[cb] = c; this.flat.push(c);
      }
      this.stats.predictor_ms += performance.now() - t0;
      // end_frame: сумма эмбеддингов кадра
      const pend = new Float32Array(D), tmp = new Float32Array(D);
      q.codecRow(this.codes[0], tmp);
      for (let d = 0; d < D; d++) pend[d] += tmp[d];
      for (let g = 1; g < G; g++) { q.row(q.predEmb[g - 1], this.codes[g], tmp); for (let d = 0; d < D; d++) pend[d] += tmp[d]; }
      this.pending = pend;
      this.framesDone++; this.frame++; this.buffered++; this.stats.frames++;
    }
  }

  // Прогрев декодера кодека кодами образца (в оригинале они декодируются вместе с речью и обрезаются)
  async _warm() {
    this.warmed = true;
    const q = this.q, G = q.tc.groups, cap = this.refFrames + this.maxFrames + 16;
    this.cs = q.newCodecState(cap);
    if (!this.refFrames) return;
    const ctx = this.ctxFrames >= 0xFFFFFFFF ? this.refFrames : Math.min(this.refFrames, this.ctxFrames || this.refFrames);
    for (let f = this.refFrames - ctx; f < this.refFrames; f += 16) {
      const n = Math.min(16, this.refFrames - f);
      await q.codecForward(this.cs, this.refCodes.subarray(f * G, (f + n) * G), n);
    }
  }
  async _decodeBuffered() {
    if (!this.buffered) return new Float32Array(0);
    if (!this.warmed) await this._warm();
    const t0 = performance.now(), G = this.q.tc.groups, parts = [];
    const codes = Int16Array.from(this.flat);
    let n = 0;
    for (let f = 0; f < this.buffered; f += QCHUNK_MAX_FRAMES) {
      const m = Math.min(QCHUNK_MAX_FRAMES, this.buffered - f);
      const pcm = await this.q.codecForward(this.cs, codes.subarray(f * G, (f + m) * G), m);
      parts.push(pcm); n += pcm.length;
    }
    this.flat = []; this.buffered = 0;
    this.stats.codec_ms += performance.now() - t0;
    if (parts.length === 1) return parts[0];
    const out = new Float32Array(n);
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
  }

  // Один шаг (аналог одного прохода vox_session_pull). → { pcm?, done?, idle? }
  async step() {
    if (this.finished) {
      if (this.flushed) return { done: true };
      const pcm = await this._decodeBuffered();
      this.flushed = true;
      return { pcm, done: true };
    }
    await this.generate();
    if (this.buffered) return { pcm: await this._decodeBuffered() };      // чанк набран или не хватает текста — отдаём, что есть
    if (this.finished) return this.step();
    return { idle: true };
  }

  dispose() {
    if (this.ts) { this.ts.free(); this.ts = null; }
    if (this.cs) { this.q.freeCodecState(this.cs); this.cs = null; }
  }
}

// ---------- модель ----------
class QwenModel {
  constructor(gpu, tm, q) {
    this.gpu = gpu; this.q = q;
    this.manifest = tm.manifest;
    this.tm = tm;
    this.family = "qwen3-tts";
    this.sampleRate = (tm.manifest.audio || {}).sample_rate || 24000;
    this.generation = Object.assign({}, QWEN_DEFAULTS, tm.manifest.generation || {});
    const m = tm.manifest.model || {};
    this.type = m.type || "base";
    this.tok = new QwenTokenizer(new TextDecoder().decode(tm.raw("tokenizer.vocab")), new TextDecoder().decode(tm.raw("tokenizer.merges")), (tm.manifest.tokenizer || {}).specials || []);
    const tcl = q.tc;
    this.speakers = Object.keys(tcl.spk_id).sort();
    this.languages = ["auto"].concat(Object.keys(tcl.lang).sort());
    const small = /0b6|0\.6b/i.test(m.variant || "");
    this.capabilities = {
      named_speakers: this.type === "custom_voice" && this.speakers.length > 0,
      instruct: (this.type === "custom_voice" && !small) || this.type === "voice_design",
      requires_instruct: this.type === "voice_design",
      clone_from_tvoice: this.type === "base", clone_from_audio: this.type === "base" && !!((tm.manifest.qwen3 || {}).speaker_encoder && (tm.manifest.qwen3 || {}).encoder),
      speakers: this.speakers, languages: this.languages, text_in: "token", audio_out: "frames",
    };
    this.info = { name: m.name || "", family: "qwen3-tts", variant: m.variant || "", type: this.type, sampleRate: this.sampleRate, weightBytes: q.wbytes, device: gpu.info, backend: gpu.isCpu ? "cpu" : "webgpu", precision: tm.manifest.precision || {}, frameRate: this.sampleRate / q.cc.samples_per_frame };
  }

  // ---- голоса ----
  voiceFromFile(bytes) {
    const v = QVoice.from(bytes);
    if (v.model && v.model.family && v.model.family !== "qwen3-tts") fail("голос создан для другого семейства: " + v.model.family, "arg");
    if (v.mode !== "speaker" && v.speakerEmb.length !== this.q.tc.hidden) fail("голос создан для модели другого размера (вектор диктора " + v.speakerEmb.length + ", нужен " + this.q.tc.hidden + ")", "arg");
    return v;
  }
  // Готовый диктор (CustomVoice) и/или описание голоса (VoiceDesign, CustomVoice 1.7B)
  speakerVoice(speaker, instruct, opts = {}) {
    const c = this.capabilities;
    let spk = String(speaker || ""), ins = String(instruct || "");
    if (!spk && !ins) fail("укажите диктора, описание голоса или и то и другое", "arg");
    if (spk) {
      if (!c.named_speakers) fail("в этой модели нет встроенных дикторов (нужна модель CustomVoice или клонирование голоса из записи)", "arg");
      spk = spk.toLowerCase();
      if (!c.speakers.includes(spk)) fail("неизвестный диктор '" + speaker + "'; в модели есть: " + c.speakers.join(", "), "arg");
    }
    if (ins && !c.instruct) fail("эта модель не принимает описание голоса/стиля (умеют VoiceDesign и CustomVoice 1.7B)", "arg");
    if (c.requires_instruct) {
      if (!ins) fail("эта модель (VoiceDesign) требует описание голоса", "arg");
      if (spk) fail("в VoiceDesign нет именованных дикторов: опишите голос текстом", "arg");
    }
    if (!c.named_speakers && !c.requires_instruct) fail("эта модель не создаёт голоса без записи", "arg");
    return new QVoice({ mode: "speaker", name: spk || "designed", speaker: spk, instruct: ins, language: opts.language || "auto" });
  }
  // Голос из записи (Base-модели): вектор диктора (ECAPA-TDNN) + коды образца (Mimi), всё на CPU маленькими порциями.
  // audio: Float32Array (+ opts.sampleRate), AudioBuffer, Blob/File, ArrayBuffer, URL. refText — что сказано в записи (нужен для icl).
  // opts: { xvectorOnly, language, name, onProgress(доля), signal }
  async cloneVoice(audio, refText, opts = {}) {
    if (!this.capabilities.clone_from_audio) fail("эта модель не создаёт голос по записи (нужна Base-модель: " + (this.type === "base" ? "в загруженном файле нет энкодеров голоса, соберите его из чекпойнта HF" : "тип " + this.type + ")"), "unsupported");
    const xv = !!opts.xvectorOnly;
    if (!xv && (typeof refText !== "string" || !refText.trim())) fail("нужен текст образца (refText): что произносится в записи (или xvectorOnly: true)", "arg");
    let pcm, sr;
    if (audio instanceof Float32Array) { pcm = audio; sr = opts.sampleRate; if (!sr) fail("для Float32Array укажите opts.sampleRate", "arg"); }
    else ({ pcm, sampleRate: sr } = await decodeAudio(audio));
    let a = resampleSinc(pcm, sr, 24000);
    if (a.length < 24000) fail("образец слишком короткий: нужно хотя бы пара секунд речи", "arg");
    const maxS = opts.maxSeconds || 30;
    if (a.length > maxS * 24000) a = a.slice(0, maxS * 24000);   // длинный образец только замедляет, а голос от него не лучше
    const prog = opts.onProgress || (() => {});
    const abort = () => { if (opts.signal && opts.signal.aborted) fail("создание голоса отменено", "aborted"); };
    const y = new Yielder(24, abort);
    if (!this._venc) this._venc = await QwenVoiceEncoder.create(this.tm, y);
    this._venc.y = y;
    const enc = this._venc, t0 = performance.now();
    prog(0.02);
    const emb = await enc.speakerEmbedding(a);
    if (emb.length !== this.q.tc.hidden) fail("вектор диктора не совпал с размером talker", "format");
    prog(xv ? 1 : 0.25);
    const d = { mode: xv ? "xvector_only" : "icl", name: opts.name || "", refText: xv ? "" : refText.trim(), language: opts.language || "auto", speakerEmb: emb,
      model: { family: "qwen3-tts", variant: (this.manifest.model || {}).variant || "", weights_sha256: (this.manifest.model || {}).weights_sha256 || "" } };
    if (!xv) {
      const r = await enc.encode(a, (f) => prog(0.25 + 0.75 * f));
      if (r.codebooks !== this.q.tc.groups) fail("энкодер речи дал " + r.codebooks + " кодбуков, а talker ждёт " + this.q.tc.groups, "format");
      d.refCodes = r.codes; d.refFrames = r.frames; d.refCodebooks = r.codebooks;
      d.refTextTokens = Int32Array.from(this.tok.encode("<|im_start|>assistant\n" + refText.trim() + "<|im_end|>\n"));
    }
    const v = new QVoice(d);
    v.seconds = a.length / 24000; v.createMs = performance.now() - t0;
    return v;
  }

  _check(voice) {
    if (!(voice instanceof QVoice)) fail("передайте голос: opts.voice = model.speakerVoice(...) или model.voiceFromFile(bytes)", "arg");
    return voice;
  }

  // Весь текст сразу → поток звука (кусками по chunk_frames кадров)
  async *stream(text, opts = {}) {
    if (typeof text !== "string" || !text.trim()) fail("пустой текст", "arg");
    const voice = this._check(opts.voice), t0 = performance.now();
    const s = new QwenSession(this, voice, opts);
    try {
      await s.init();
      await s.appendText(this.tok.encode(text.normalize("NFC")));
      s.textEnded = true;
      let produced = 0;
      for (;;) {
        const r = await s.step();
        if (r.pcm && r.pcm.length) { produced += r.pcm.length; yield r.pcm; }
        if (r.done) break;
      }
      this.lastRun = { audioSeconds: produced / this.sampleRate, wallMs: performance.now() - t0, frames: s.stats.frames, timing: Object.assign({}, s.stats) };
    } finally { s.dispose(); }
  }
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
    const cfg = { sample_rate: this.sampleRate, n_fft: 1024, hop: 256, win: 1024, n_mels: 100 };
    if (!this._melFb) this._melFb = melFilterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels);
    const r = melLogSpectrogram(cfg, this._melFb, pcm);
    return { mel: r.mel, frames: r.frames, bands: cfg.n_mels };
  }
  dispose() { if (this._venc) this._venc.release(); this.q.dispose(); try { this.gpu.device.destroy(); } catch (e) { /* ок */ } }
}

// ============================================================================
// Публичный объект Vox
// ============================================================================
const Vox = {
  version: VOX_VERSION,
  supported: () => typeof navigator !== "undefined" && !!navigator.gpu,
  // true — есть WebGPU; без него vox.js считает на процессоре (backend: "cpu" включает его принудительно)
  backends: () => ({ webgpu: typeof navigator !== "undefined" && !!navigator.gpu, cpu: true }),
  load: (src, opts) => VoxModel.load(src, opts),
  // Только конвертация: что угодно (HF-ссылка, URL, File, {ckpt,vocab,vocos}) → байты .tmod (можно сохранить и отдавать как файл).
  convert: (src, opts) => acquireTmod(src, opts),
  hf: { parse: parseHfRef, list: (ref, opts) => hfList(typeof ref === "string" ? parseHfRef(ref, (opts || {}).hfHost) : ref, opts || {}) },
  clearCache: cacheClear,
  config: (c) => { if (c && c.baseUrl != null) VOX_BASE_URL = c.baseUrl; return { baseUrl: VOX_BASE_URL }; },
  Voice, QVoice, VoxError, decodeAudio, wav: wavBlob, play: playPcm, StreamPlayer,
  splitText: chunkText,
  _internal: { QwenVoiceEncoder, Yielder, openQwenSource, qwenFsFromHf, qwenFsFromMap, qwenSourceToTmod, QwenTokenizer, qwenPretokenize, QwenGpu, QwenModel, QwenSession, qSample, qUniform, unpickle, readTorchArchive, parseSafetensors, convertF5, splitVocab, parseVocosPadding, melFilterbank, planFromFiles, f32ToF16Bits, f32ToBf16Bits, sniffKind, Gpu, CpuGpu, createDevice, Tmod, F5Gpu, F5Tokenizer, f5Noise, f5Timesteps, melLogSpectrogram, istftSame, resampleSinc, blen },
};

// ============================================================================
// Двусторонний стриминг в браузере: текст подаётся на лету, звук приходит на лету. Та же логика, что в ядре vox
// (vox.h ABI 0.4): политика коммита auto | sentence | manual, idle_ms, flush, правка ещё не озвученного хвоста.
// Дуплекс — общая оболочка; модель подключается адаптером (для F5-TTS — F5Stream, озвучивает куски целиком).
// ============================================================================
const _utf8 = typeof TextEncoder !== "undefined" ? new TextEncoder() : null;
const utf8len = (s) => _utf8.encode(s).length;

// Конец последнего законченного предложения (индекс в строке JS), 0 — нет. Знак конца законченный, если за ним пробел/перевод строки
// или он полноширинный (。！？…); final — конец текста. "3.5", "e.g." и "..." посередине не рвутся. Порт sentence_cut из ядра.
function sentenceCut(t, final) {
  let best = 0;
  const isWs = (c) => c === " " || c === "\t" || c === "\n" || c === "\r";
  for (let i = 0; i < t.length;) {
    const c = t[i];
    const full = c === "。" || c === "！" || c === "？" || c === "…";
    const term = full || c === "." || c === "!" || c === "?" || c === "\n";
    const len = 1;   // все знаки конца — одиночные кодовые единицы UTF-16
    if (term) {
      let j = i + len;
      const wsAfter = j < t.length && isWs(t[j]);
      if (full || wsAfter || final || c === "\n") {
        while (j < t.length && isWs(t[j])) j++;
        const glued = !full && j === i + len && j < t.length && (t[j] === "." || t[j] === "!" || t[j] === "?");
        if (!glued) best = j;
      }
    }
    i += len;
  }
  return best;
}

// Адаптер F5-TTS (порт direct_text / direct_seal / direct_step из backend_f5.cpp): куски озвучиваются целиком.
class F5Stream {
  constructor(model, voice, s, opts) {
    this.model = model; this.voice = voice; this.s = s; this.opts = opts;
    this.P = model._plan(voice, s);
    this.sp = new Splicer(this.P.cfMax);
    this.seg = ""; this.sealed = false; this.nextChunk = 0; this.counter = 0; this.flushedEnd = false;
    this.caps = { text_in: "chunk", audio_out: "chunks", exact: true };
    this.direct = true;
    model.f5.timing = { dit_ms: 0, vocos_ms: 0, text_ms: 0 };
  }
  commit(piece, seal) { this.seg += piece; if (seal) this.sealed = true; }
  seal() { this.sealed = true; }
  safeCut(pending) { return pending.length; }
  async step(ended) {
    const chunks = chunkText(this.seg, this.P.maxChars);
    const avail = (ended || this.sealed) ? chunks.length : Math.max(0, chunks.length - 1);
    if (this.nextChunk < avail) {
      const wave = await this.model._chunkWave(this.voice, this.s, this.P, chunks[this.nextChunk], this.counter, this.opts);
      const piece = this.sp.push(wave, this.counter === 0);
      this.nextChunk++; this.counter++;
      return { pcm: piece };
    }
    if (this.sealed && !ended) {
      const tail = this.sp.release(true);
      this.seg = ""; this.nextChunk = 0; this.sealed = false;
      return { pcm: tail };
    }
    if (!ended) return { idle: true };
    if (!this.flushedEnd) { this.flushedEnd = true; return { pcm: this.sp.release(false), done: true }; }
    return { done: true };
  }
}

// Адаптер авторегрессионных моделей (Qwen3-TTS): текст подаётся токенами, звук идёт покадрово. Высказывание = одна сессия
// генерации; пауза/правка закрывает текущее высказывание (оно дочитывается), следующий текст открывает новое (seed + номер).
class QwenStream {
  constructor(model, voice, opts) {
    this.model = model; this.voice = voice; this.opts = opts;
    this.caps = { text_in: "token", audio_out: "frames", exact: true };
    this.direct = false;
    this.sessions = []; this.open = null; this.count = 0;
    this.baseSeed = Math.trunc(opts.seed != null ? opts.seed : (voice.sampling && voice.sampling.seed) || 1) || 1;
  }
  _new() {
    const o = Object.assign({}, this.opts, { seed: this.baseSeed + this.count++ });
    const e = { sess: new QwenSession(this.model, this.voice, o), queue: [], closing: false, ready: false };
    this.sessions.push(e);
    return e;
  }
  commit(piece) {
    const toks = this.model.tok.encode(piece);
    if (!toks.length) return;
    if (!this.open) this.open = this._new();
    for (const t of toks) this.open.queue.push(t);
  }
  seal() { /* у AR-модели нечего «запечатывать»: хвост слова уже закоммичен, фраза закончится с паузой/end */ }
  safeCut(pending) { return this.model.tok.safeCut(pending); }
  endUtterance() { if (this.open) { this.open.closing = true; this.open = null; } }
  gap() { this.endUtterance(); }
  async step(ended) {
    if (ended) this.endUtterance();
    const e = this.sessions[0];
    if (!e) return ended ? { done: true } : { idle: true };
    const s = e.sess;
    if (!e.ready) { await s.init(); e.ready = true; }
    if (e.queue.length) { const t = e.queue; e.queue = []; await s.appendText(t); }
    if (e.closing) s.textEnded = true;
    const r = await s.step();
    if (r.done) { s.dispose(); this.sessions.shift(); return { pcm: r.pcm }; }
    return r;
  }
  dispose() { for (const e of this.sessions) e.sess.dispose(); this.sessions = []; }
}

class Duplex {
  constructor(model, opts = {}) {
    const voice = opts.voice;
    const commit = opts.commit || "auto";
    if (!["auto", "sentence", "manual"].includes(commit)) fail("commit: auto | sentence | manual", "arg");
    const onConflict = opts.on_conflict || "keep";
    if (!["keep", "raise", "respeak"].includes(onConflict)) fail("on_conflict: keep | raise | respeak", "arg");
    this.model = model; this.voice = voice;
    this.commit = commit; this.idleMs = Math.max(0, Number(opts.idle_ms) || 0); this.gapMs = Math.max(0, Number(opts.utterance_gap_ms) || 0);
    this.onConflict = onConflict;
    this.sampleRate = model.sampleRate;
    this.ac = new AbortController();
    if (opts.signal) opts.signal.addEventListener("abort", () => this.cancel());
    this.be = model._duplexBackend(voice, opts, this.ac.signal);
    this.caps = this.be.caps;
    this.pending = ""; this.committedText = "";
    this.ended = false; this.cancelled = false; this.done = false; this.error = null;
    this.flushes = 0; this.idleFlushes = 0; this.idleArmed = false; this.lastFeed = 0; this.lastText = 0; this.gapFired = true;
    this.utterances = 1;
    this.q = []; this._aw = null; this._wake = null;
    this.stats = { firstAudioMs: null, audioSamples: 0 };
    this._tFirstWrite = null;
    this._pump();
  }

  // ---------- вход ----------
  _commitPending(force) {
    const t = this.pending;
    if (!t) return 0;
    let cut = 0;
    if (force) cut = t.length;
    else if (this.commit === "sentence") cut = sentenceCut(t, false);
    else if (this.commit === "auto") cut = this.be.safeCut(t);
    if (!cut) return 0;
    const piece = t.slice(0, cut);
    this.be.commit(piece, this.be.direct && (force || this.commit !== "auto"));
    this.pending = t.slice(cut);
    this.committedText += piece;
    return cut;
  }
  _poke() { this._ver = (this._ver || 0) + 1; if (this._wake) { const w = this._wake; this._wake = null; w(); } }
  write(text) {
    if (this.ended) fail("текст уже закончен (end())", "state");
    if (this.cancelled) fail("сессия отменена", "state");
    if (!text) return this;
    if (this._tFirstWrite == null) this._tFirstWrite = performance.now();
    this.pending += String(text).normalize("NFC");
    this.lastFeed = this.lastText = performance.now(); this.idleArmed = true; this.gapFired = false;
    this._commitPending(false);
    this._poke();
    return this;
  }
  flush() {
    if (this.ended || this.cancelled) return this;
    if (this._commitPending(true)) this.flushes++;
    else if (this.be.direct) this.be.seal();
    this.idleArmed = false;
    this._poke();
    return this;
  }
  // Режим набора: передаётся ВЕСЬ текст из поля ввода. → { ok, committed, pending, spoken? } (committed — в байтах UTF-8)
  setText(full) {
    if (this.ended) fail("текст уже закончен (end())", "state");
    full = String(full).normalize("NFC");
    if (this._tFirstWrite == null && full) this._tFirstWrite = performance.now();
    const logical = this.committedText + this.pending;
    const cl = this.committedText.length;
    let res;
    if (full.slice(0, cl) === this.committedText) {
      this.pending = full.slice(cl);
      this.lastFeed = this.lastText = performance.now(); this.idleArmed = !!this.pending; this.gapFired = false;
      this._commitPending(false);
      res = { ok: true, committed: utf8len(this.committedText), pending: this.pending };
    } else {
      res = { ok: false, committed: utf8len(this.committedText), spoken: this.committedText, pending: this.pending };
      if (this.onConflict === "raise") fail("правка внутри уже озвученного текста (озвучено: " + JSON.stringify(this.committedText) + ")", "conflict");
      if (this.onConflict === "respeak") {
        let i = 0;
        while (i < Math.min(full.length, cl) && full[i] === logical[i]) i++;
        const j = full.lastIndexOf(" ", i - 1) + 1;     // с начала изменённого слова
        this.flush();
        if (this.be.endUtterance) this.be.endUtterance();
        this.committedText = "";
        this.pending = full.slice(j);
        this._commitPending(false);
        this.utterances++;
        res = { ok: true, committed: utf8len(this.committedText), pending: this.pending, respeakFrom: j };
      }
    }
    this._poke();
    return res;
  }
  end() {
    if (this.ended) return this;
    this._commitPending(true);
    this.ended = true;
    this._poke();
    return this;
  }
  cancel() {
    this.cancelled = true; this.ended = true;
    try { this.ac.abort(); } catch (e) { /* ок */ }
    this._poke();
  }
  info() {
    return { committed_bytes: utf8len(this.committedText), pending: this.pending, pending_bytes: utf8len(this.pending), text_ended: this.ended,
      finished: this.done, commit: this.commit, idle_ms: this.idleMs, flushes: this.flushes, idle_flushes: this.idleFlushes, utterances: this.utterances };
  }

  // ---------- рабочий цикл ----------
  _sleep(ms) {
    return new Promise((res) => {
      const t = ms > 0 ? setTimeout(() => { this._wake = null; res(); }, ms) : null;
      this._wake = () => { if (t) clearTimeout(t); res(); };
    });
  }
  _emit(pcm) {
    if (!pcm || !pcm.length) return;
    if (this.stats.firstAudioMs == null && this._tFirstWrite != null) this.stats.firstAudioMs = performance.now() - this._tFirstWrite;
    this.stats.audioSamples += pcm.length;
    this.q.push(pcm);
    if (this._aw) { const w = this._aw; this._aw = null; w(); }
  }
  async _pump() {
    try {
      for (;;) {
        if (this.cancelled) break;
        // простой: автокоммит накопленного
        if (this.idleMs && this.idleArmed && !this.ended && this.pending && performance.now() - this.lastFeed >= this.idleMs) {
          this.idleArmed = false;
          if (this._commitPending(true)) { this.flushes++; this.idleFlushes++; }
        }
        // пауза между высказываниями: для F5 это то же, что flush (хвост отдаётся, следующий текст — с нуля)
        if (this.gapMs && !this.gapFired && !this.ended && performance.now() - this.lastText >= this.gapMs && (this.committedText || this.pending)) {
          this.gapFired = true;
          if (this.be.gap) { if (this._commitPending(true)) this.flushes++; this.be.gap(); } else this.flush();
        }
        const ver = this._ver || 0;
        const r = await this.be.step(this.ended);
        if (r.pcm) this._emit(r.pcm);
        if (r.done) break;
        if (r.idle && (this._ver || 0) === ver) {   // не засыпаем, если пока шёл шаг, пришёл новый текст (иначе потеряется «толчок»)
          let wait = 0;
          if (this.idleMs && this.idleArmed && this.pending) wait = Math.max(1, this.idleMs - (performance.now() - this.lastFeed));
          if (this.gapMs && !this.gapFired) wait = wait ? Math.min(wait, this.gapMs) : this.gapMs;
          await this._sleep(wait);
        }
      }
    } catch (e) {
      if (!(e instanceof VoxError && e.code === "aborted")) this.error = e;
    } finally {
      if (this.be.dispose) this.be.dispose();
      this.done = true;
      if (this._aw) { const w = this._aw; this._aw = null; w(); }
    }
  }

  // ---------- выход ----------
  // Асинхронный итератор по кускам звука до конца (после end() или cancel()).
  async *audio() {
    for (;;) {
      if (this.q.length) { yield this.q.shift(); continue; }
      if (this.done) { if (this.error) throw this.error; return; }
      await new Promise((res) => { this._aw = res; });
    }
  }
  // Один кусок или null по таймауту/концу (проверяйте .done).
  async read(timeoutMs) {
    if (this.q.length) return this.q.shift();
    if (this.done) { if (this.error) throw this.error; return null; }
    await Promise.race([new Promise((res) => { this._aw = res; }), new Promise((res) => setTimeout(res, timeoutMs == null ? 1e9 : timeoutMs))]);
    this._aw = null;
    return this.q.length ? this.q.shift() : null;
  }
  // Закрывает текст, ждёт, отдаёт всю запись одним массивом.
  async final() {
    this.end();
    const parts = [];
    let n = 0;
    for await (const c of this.audio()) { parts.push(c); n += c.length; }
    const pcm = new Float32Array(n);
    let o = 0;
    for (const p of parts) { pcm.set(p, o); o += p.length; }
    return { pcm, sampleRate: this.sampleRate, seconds: n / this.sampleRate, stats: Object.assign({}, this.stats) };
  }
}

VoxModel.prototype.duplex = function (opts) { return new Duplex(this, opts); };
QwenModel.prototype.duplex = function (opts) { return new Duplex(this, opts); };
VoxModel.prototype._duplexBackend = function (voice, opts, signal) {
  if (!(voice instanceof Voice)) fail("передайте голос: model.duplex({ voice })", "arg");
  if (voice.melDim !== this.f5.cfg.mel_dim) fail("голос создан для другой модели", "arg");
  return new F5Stream(this, voice, this._opts(opts), { signal, onStep: opts.onStep, noise: opts.noise });
};
QwenModel.prototype._duplexBackend = function (voice, opts, signal) {
  this._check(voice);
  return new QwenStream(this, voice, Object.assign({}, opts, { signal }));
};
Vox.Duplex = Duplex;
Vox._internal.sentenceCut = sentenceCut;
Vox._internal.Splicer = Splicer;

if (typeof module === "object" && module.exports) module.exports = Vox;
global.Vox = Vox;
})(typeof globalThis !== "undefined" ? globalThis : window);
