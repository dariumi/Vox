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
