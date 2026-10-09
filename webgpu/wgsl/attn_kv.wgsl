// Причинное внимание с GQA над KV-кэшем. Запрос i (0..m) стоит на позиции pos0 + i и видит ключи [first, pos0 + i],
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
