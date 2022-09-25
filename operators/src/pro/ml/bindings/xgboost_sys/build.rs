extern crate bindgen;
extern crate cmake;

use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let out_dir = &env::var("OUT_DIR").unwrap();
    let target = env::var("TARGET").unwrap();

    let out_dir_pathbuf = PathBuf::from(out_dir);

    let cmake_path = format!("{}/xgboost", out_dir);
    let xg_include_path = format!("{}/xgboost/include", out_dir);
    let xg_rabit_include_path = format!("{}/xgboost/rabit/include", out_dir);
    let xg_dmlc_include_path = format!("{}/xgboost/dmlc-core/include", out_dir);
    let clone_path = format!("{}/xgboost", out_dir);

    if !std::path::Path::new(&xg_dmlc_include_path).exists() {
        // we need to get the source code
        std::process::Command::new("git")
            .args([
                "clone",
                "--recursive",
                "-b",
                "release_1.6.0",
                "https://github.com/dmlc/xgboost",
                &clone_path,
            ])
            .output()
            .expect("Failed to fetch git submodules!");
    }

    // CMake
    let _ = Config::new(cmake_path)
        .uses_cxx11()
        .define("BUILD_STATIC_LIB", "ON")
        .build();

    // CONFIG BINDGEN
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args(&["-x", "c++", "-std=c++11"])
        .clang_arg(format!("-I{}", Path::new(&xg_include_path).display()))
        .clang_arg(format!("-I{}", Path::new(&xg_rabit_include_path).display()))
        .clang_arg(format!("-I{}", Path::new(&xg_dmlc_include_path).display()))
        .generate()
        .expect("Unable to generate bindings.");

    // GENERATE THE BINDINGS
    bindings
        .write_to_file(out_dir_pathbuf.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    // link to appropriate C++ lib
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=dylib=omp");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=dylib=gomp");
    }

    // LINK STUFF (LINUX)
    println!(
        "cargo:rustc-link-search={}",
        out_dir_pathbuf.join("lib").display()
    );
    println!("cargo:rustc-link-lib=xgboost");
    println!("cargo:rustc-link-lib=dmlc");
}
