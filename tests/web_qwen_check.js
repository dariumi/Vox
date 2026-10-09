// Сверка конвертации Qwen3-TTS в vox.js с python -m vox.build: каждый тензор побайтно, манифест — по полям.
// node web_qwen_check.js <vox.js> <папка HF> <эталон.tmod> <dtype> [quant]
const fs = require("fs"), path = require("path");
const Vox = require(path.resolve(process.argv[2]));
function readDir(d, pre = "") {
  const m = {};
  for (const f of fs.readdirSync(d, { withFileTypes: true })) {
    const p = path.join(d, f.name);
    if (f.isDirectory()) Object.assign(m, readDir(p, pre + f.name + "/")); else m[pre + f.name] = new Uint8Array(fs.readFileSync(p));
  }
  return m;
}
function parseTmod(b) {
  const dv = new DataView(b.buffer, b.byteOffset);
  const mo = Number(dv.getBigUint64(16, true)), ms = Number(dv.getBigUint64(24, true)), dof = Number(dv.getBigUint64(32, true));
  const man = JSON.parse(new TextDecoder().decode(b.subarray(mo, mo + ms)));
  return { man, get: (n) => { const t = man.blobs[n]; return b.subarray(dof + t.offset, dof + t.offset + t.nbytes); } };
}
(async () => {
  const [, , , dir, ref, dtype, quant] = process.argv;
  const out = await Vox.convert({ files: readDir(dir) }, { dtype, quant: quant || "", cache: false });
  const a = parseTmod(out), b = parseTmod(new Uint8Array(fs.readFileSync(ref)));
  let bad = 0, n = 0, skip = 0;
  for (const k of Object.keys(b.man.blobs)) {
    if (!a.man.blobs[k]) { console.log("нет в JS:", k); bad++; continue; }
    n++;
    const ta = a.man.blobs[k], tb = b.man.blobs[k];
    if (ta.dtype !== tb.dtype || JSON.stringify(ta.shape) !== JSON.stringify(tb.shape)) { console.log("форма/тип:", k, ta.dtype, tb.dtype, ta.shape, tb.shape); bad++; continue; }
    const x = a.get(k), y = b.get(k);
    let same = x.length === y.length;
    for (let i = 0; same && i < x.length; i++) if (x[i] !== y[i]) same = false;
    if (!same) { console.log("различаются байты:", k); bad++; }
  }
  for (const k of Object.keys(a.man.blobs)) if (!b.man.blobs[k]) { console.log("лишнее в JS:", k); bad++; }
  for (const key of ["qwen3", "generation", "audio", "tokenizer", "precision", "model"]) {
    let x = JSON.stringify(a.man[key]), y = b.man[key];
    y = JSON.stringify(y);
    if (x !== y) { console.log("манифест:", key); bad++; }
  }
  console.log((bad ? "FAIL " : "ok ") + n + " тензоров сравнено (включая энкодеры для создания голоса), расхождений: " + bad);
  process.exit(bad ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(2); });
