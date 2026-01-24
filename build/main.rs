mod shaders;
mod assets;

use shaders::compile_shaders;
use assets::copy_assets;

fn main() {

    vcpkg::find_package("sdl3").unwrap();
    vcpkg::find_package("vulkan").unwrap();
    vcpkg::find_package("vulkan-memory-allocator").unwrap();
    vcpkg::find_package("shaderc").unwrap();
    vcpkg::find_package("assimp").unwrap();
    vcpkg::find_package("meshoptimizer").unwrap();

    println!("cargo:rerun-if-changed=build/main.rs");
    println!("cargo:rerun-if-changed=build/shaders.rs");
    println!("cargo:rerun-if-changed=build/assets.rs");
    
    compile_shaders();
    copy_assets();
}