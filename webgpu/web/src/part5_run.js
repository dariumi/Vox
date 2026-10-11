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
