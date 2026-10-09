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
