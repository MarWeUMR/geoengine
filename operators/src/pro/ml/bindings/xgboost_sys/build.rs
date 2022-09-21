extern crate bindgen;
extern crate cmake;

use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};

fn main() {

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let target = env::var("TARGET").unwrap();
    
    let od = out_dir.to_str().unwrap();
    let od_pathbuf = PathBuf::from(od);

    let cmake_path  = format!("{}/xgboost", od);
    let xg_include_path  = format!("{}/xgboost/include", od);
    let xg_rabit_include_path  = format!("{}/xgboost/rabit/include", od);
    let xg_dmlc_include_path  = format!("{}/xgboost/dmlc-core/include", od);
    let clone_path  = format!("{}/xgboost", od);


    if std::path::Path::new(&xg_dmlc_include_path).exists() == false {
        println!("cloning xgboost repo into out_dir ...");
        std::process::Command::new("git")
        .args(["clone", "--recursive", "-b", "release_1.6.0", "https://github.com/dmlc/xgboost", &clone_path])
        .output()
        .expect("Failed to fetch git submodules!");
    } else {
        println!("Found xgboost repo.");
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
        .clang_arg(format!(
            "-I{}",
            Path::new(&xg_include_path).display()
        ))
        .clang_arg(format!(
            "-I{}",
            Path::new(&xg_rabit_include_path).display()
        ))
        .clang_arg(format!(
            "-I{}",
            Path::new(&xg_dmlc_include_path).display()
        ))
        .generate()
        .expect("Unable to generate bindings.");

    // GENERATE THE BINDINGS
    bindings
        .write_to_file(od_pathbuf.join("bindings.rs"))
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
    println!("cargo:rustc-link-search={}", od_pathbuf.join("lib").display());
    println!("cargo:rustc-link-lib=xgboost");
    println!("cargo:rustc-link-lib=dmlc");
}
