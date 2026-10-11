// Where libvox lives: VOX_LIB_DIR (default ../../build). At run time LD_LIBRARY_PATH is needed (macOS: DYLD_LIBRARY_PATH, Windows: PATH).
fn main() {
    let dir = std::env::var("VOX_LIB_DIR").unwrap_or_else(|_| "../../build".into());
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=vox");
    println!("cargo:rerun-if-env-changed=VOX_LIB_DIR");
}
