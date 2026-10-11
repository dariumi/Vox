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
