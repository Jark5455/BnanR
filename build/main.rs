mod shaders;
mod assets;

use shaders::compile_shaders;
use assets::copy_assets;

fn main() {

    vcpkg::Config::new()
        .vcpkg_root("target/vcpkg".parse().unwrap())
        .find_package("sdl3").unwrap();

    vcpkg::Config::new()
        .vcpkg_root("target/vcpkg".parse().unwrap())
        .find_package("assimp").unwrap();

    println!("cargo:rerun-if-changed=build/main.rs");
    println!("cargo:rerun-if-changed=build/shaders.rs");
    println!("cargo:rerun-if-changed=build/assets.rs");
    
    compile_shaders();
    copy_assets();
}