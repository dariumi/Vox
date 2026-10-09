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
