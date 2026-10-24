// Node.js via koffi (FFI to the C ABI): synthesis to stdout as raw float32 PCM.
//   cd examples/node && npm install && node vox.mjs model.tmod voice.tvoice "Hello!" > out.f32
// Library path: VOX_LIB (default ../../build/libvox.so; macOS: libvox.dylib, Windows: vox.dll).
import koffi from "koffi";

const lib = koffi.load(process.env.VOX_LIB || new URL("../../build/libvox.so", import.meta.url).pathname);
const vox_last_error = lib.func("const char *vox_last_error()");
const vox_gen_params_default = lib.func("void vox_gen_params_default(void *p)");
const vox_engine_open = lib.func("int vox_engine_open(const char *path, void *opts, _Out_ void **out)");
const vox_engine_close = lib.func("void vox_engine_close(void *e)");
const vox_engine_sample_rate = lib.func("uint32_t vox_engine_sample_rate(void *e)");
const vox_voice_load = lib.func("int vox_voice_load(void *e, const char *path, _Out_ void **out)");
const vox_voice_free = lib.func("void vox_voice_free(void *v)");
const vox_session_create = lib.func("int vox_session_create(void *e, void *v, void *p, _Out_ void **out)");
const vox_session_feed_text = lib.func("int vox_session_feed_text(void *s, const char *utf8)");
const vox_session_end_text = lib.func("int vox_session_end_text(void *s)");
const vox_session_pull = lib.func("int vox_session_pull(void *s, float *pcm, size_t cap, _Out_ size_t *written)");
const vox_session_free = lib.func("void vox_session_free(void *s)");

const check = (st, what) => { if (st < 0) throw new Error(`${what}: ${vox_last_error()}`); };

const [model, voicePath, text] = process.argv.slice(2);
if (!text) { console.error("usage: node vox.mjs model.tmod voice.tvoice text"); process.exit(1); }

const out = [null], voice = [null], sess = [null];
check(vox_engine_open(model, null, out), "open");
const engine = out[0];
check(vox_voice_load(engine, voicePath, voice), "voice");

// vox_gen_params: vox_gen_params_default fills struct_size; seed is a uint64 at offset 32
const params = Buffer.alloc(72);
vox_gen_params_default(params);
params.writeBigUInt64LE(42n, 32);
check(vox_session_create(engine, voice[0], params, sess), "session");
check(vox_session_feed_text(sess[0], text), "feed");
check(vox_session_end_text(sess[0]), "end");

const buf = new Float32Array(4096), written = [0n];
let total = 0;
for (;;) {
  const st = vox_session_pull(sess[0], buf, buf.length, written);
  if (st !== 0) { check(st, "pull"); break; }
  const n = Number(written[0]);
  // a copy: writes to a pipe are asynchronous, and buf is refilled by the next pull
  process.stdout.write(Buffer.from(buf.slice(0, n).buffer));
  total += n;
}
console.error(`${total} samples @ ${vox_engine_sample_rate(engine)} Hz`);
vox_session_free(sess[0]); vox_voice_free(voice[0]); vox_engine_close(engine);
