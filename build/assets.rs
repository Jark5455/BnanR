use std::{env, fs};
use std::path::Path;
use glob::glob;

pub fn copy_assets() {
    let build_dir_pth = env::var("OUT_DIR").unwrap();
    let build_dir = Path::new(&build_dir_pth).parent().unwrap().parent().unwrap();

    for entry in glob("src/**/*.bpk").expect("glob failed") {
        let path = entry.unwrap();

        println!("cargo:rerun-if-changed={}", path.display());

        let mut components = path.components();
        components.next();

        let assets_path = components.as_path();
        let out_path = build_dir.join(assets_path);
        let out_dir = out_path.parent().unwrap().to_owned();

        if !out_dir.exists() {
            fs::create_dir_all(out_dir).unwrap();
        }

        fs::copy(path, out_path.clone()).unwrap();

        println!("cargo:rerun-if-changed={}", out_path.display());
    }
}