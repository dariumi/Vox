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
