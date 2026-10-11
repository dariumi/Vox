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
// o: { ckpt: Uint8Array, ckptName, vocab: string, vocos: Uint8Array, vocosName, dtype: "bf16"|"f16"|"f32", quant: ""|"q8",
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
  for (const [k, t] of voc) addBlob(blobs, "vocos." + k, t.shape, t.get, dtype, false);
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
