use bindgen::callbacks::MacroParsingBehavior;
use bindgen::callbacks::ParseCallbacks;
use cc;
use std::collections::HashSet;
use std::env;
use std::path::PathBuf;

const IGNORE_MACROS: [&str; 20] = [
    "FE_DIVBYZERO",
    "FE_DOWNWARD",
    "FE_INEXACT",
    "FE_INVALID",
    "FE_OVERFLOW",
    "FE_TONEAREST",
    "FE_TOWARDZERO",
    "FE_UNDERFLOW",
    "FE_UPWARD",
    "FP_INFINITE",
    "FP_INT_DOWNWARD",
    "FP_INT_TONEAREST",
    "FP_INT_TONEARESTFROMZERO",
    "FP_INT_TOWARDZERO",
    "FP_INT_UPWARD",
    "FP_NAN",
    "FP_NORMAL",
    "FP_SUBNORMAL",
    "FP_ZERO",
    "IPPORT_RESERVED",
];

#[derive(Debug)]
struct IgnoreMacros(HashSet<String>);

impl ParseCallbacks for IgnoreMacros {
    fn will_parse_macro(&self, name: &str) -> MacroParsingBehavior {
        if self.0.contains(name) { MacroParsingBehavior::Ignore } else { MacroParsingBehavior::Default }
    }
}

impl IgnoreMacros {
    fn new() -> Self {
        Self(IGNORE_MACROS.into_iter().map(|s| s.to_owned()).collect())
    }
}

fn main() {
    println!("cargo:rustc-link-search=/usr/local/cuda-12.8/targets/x86_64-linux/lib/");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");

    cc::Build::new()
        .cuda(true)
        .file("./c_src/bindings.c")
        .file("./c_src/ops.cu")
        .flag("-I/usr/local/cuda-12.8/targets/x86_64-linux/include/")
        .flag("-L/usr/local/cuda/targets/x86_64-linux/lib")
        // .flag("-ccbin=g++-12")
        .flag("-std=c++17")
        .flag("-gencode")
        .flag("arch=compute_80,code=sm_80")
        .compile("lib");

    let bindings = bindgen::Builder::default()
        .header("./c_src/bindings.h")
        .clang_arg("-I/usr/local/cuda-12.8/targets/x86_64-linux/include/")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .parse_callbacks(Box::new(IgnoreMacros::new()))
        // Longdouble creates warnings about 16-bit numbers, good to ignore if relevant
        .blocklist_function("strtold")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs")).expect("Couldn't write bindings!");
}
