//! Rust через C ABI (без зависимостей): синтез в stdout — сырой float32 PCM.
//!   cd examples/rust && cargo build --release && LD_LIBRARY_PATH=../../build ./target/release/vox-example model.tmod voice.tvoice "Привет!" > out.f32
use std::ffi::{c_char, c_void, CStr, CString};
use std::io::Write;
use std::ptr;

#[repr(C)]
struct GenParams {
    struct_size: u32,
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    top_k: i32,
    sub_temperature: f32,
    sub_top_p: f32,
    sub_top_k: i32,
    seed: u64,
    chunk_frames: u32,
    max_frames: u32,
    language: *const c_char,
    decoder_context_frames: u32,
    decoder_mode: u32,
    extra_json: *const c_char,
}

const VOX_OK: i32 = 0;

extern "C" {
    fn vox_last_error() -> *const c_char;
    fn vox_gen_params_default(p: *mut GenParams);
    fn vox_engine_open(path: *const c_char, opts: *const c_void, out: *mut *mut c_void) -> i32;
    fn vox_engine_close(e: *mut c_void);
    fn vox_engine_sample_rate(e: *const c_void) -> u32;
    fn vox_voice_load(e: *mut c_void, path: *const c_char, out: *mut *mut c_void) -> i32;
    fn vox_voice_free(v: *mut c_void);
    fn vox_session_create(e: *mut c_void, v: *const c_void, p: *const GenParams, out: *mut *mut c_void) -> i32;
    fn vox_session_feed_text(s: *mut c_void, utf8: *const c_char) -> i32;
    fn vox_session_end_text(s: *mut c_void) -> i32;
    fn vox_session_pull(s: *mut c_void, pcm: *mut f32, cap: usize, written: *mut usize) -> i32;
    fn vox_session_free(s: *mut c_void);
}

fn check(status: i32, what: &str) -> Result<(), String> {
    if status < 0 {
        let msg = unsafe { CStr::from_ptr(vox_last_error()) }.to_string_lossy().into_owned();
        return Err(format!("{what}: {msg}"));
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err(format!("usage: {} model.tmod voice.tvoice text", args[0]));
    }
    let (model, voice_path, text) = (CString::new(args[1].as_str()).unwrap(), CString::new(args[2].as_str()).unwrap(), CString::new(args[3].as_str()).unwrap());
    unsafe {
        let (mut e, mut v, mut s) = (ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
        check(vox_engine_open(model.as_ptr(), ptr::null(), &mut e), "open")?;
        check(vox_voice_load(e, voice_path.as_ptr(), &mut v), "voice")?;
        let mut p: GenParams = std::mem::zeroed();
        vox_gen_params_default(&mut p);
        p.seed = 42;
        check(vox_session_create(e, v, &p, &mut s), "session")?;
        check(vox_session_feed_text(s, text.as_ptr()), "feed")?;
        check(vox_session_end_text(s), "end")?;
        let mut buf = vec![0f32; 4096];
        let (mut total, mut out) = (0usize, std::io::stdout().lock());
        loop {
            let mut n = 0usize;
            let st = vox_session_pull(s, buf.as_mut_ptr(), buf.len(), &mut n);
            if st != VOX_OK {
                check(st, "pull")?;
                break; // VOX_DONE (2) или VOX_AGAIN (1) для синхронного режима здесь означает конец
            }
            let bytes = std::slice::from_raw_parts(buf.as_ptr() as *const u8, n * 4);
            out.write_all(bytes).map_err(|e| e.to_string())?;
            total += n;
        }
        eprintln!("{total} samples @ {} Hz", vox_engine_sample_rate(e));
        vox_session_free(s);
        vox_voice_free(v);
        vox_engine_close(e);
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
