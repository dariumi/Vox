// node tests/web_cpu_check.js <vox.js> <model.tmod> <out.f32> <speaker|-> <instruct>: синтез на процессорном бэкенде vox.js (без WebGPU)
const fs = require("fs");
const Vox = require(require("path").resolve(process.argv[2]));
(async () => {
  const [, , , tm, out, spk, ins] = process.argv;
  const b = fs.readFileSync(tm);
  const m = await Vox.load(new Uint8Array(b.buffer, b.byteOffset, b.byteLength), { backend: process.env.VOX_BACKEND || "auto" });
  if (m.info.backend !== "cpu") throw new Error("ожидался backend cpu, получили " + m.info.backend);
  const v = m.speakerVoice(spk === "-" ? null : spk, ins, { language: "english" });
  const r = await m.synthesize("Hello, this is a speaker test.", { voice: v, seed: 1, language: "english", max_frames: 10 });
  fs.writeFileSync(out, Buffer.from(r.pcm.buffer, r.pcm.byteOffset, r.pcm.byteLength));
  process.exit(0);
})().catch((e) => { console.error(e); process.exit(1); });
