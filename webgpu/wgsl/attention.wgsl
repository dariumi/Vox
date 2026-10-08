// Внимание без маски: O[q] = softmax(Q[q] K^T * scale) V, поток по ключам (онлайн-softmax). Группа — QB запросов одной головы
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
