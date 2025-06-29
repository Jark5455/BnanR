mod shaders;

use shaders::compile_shaders;

fn main() {
    
    println!("cargo:rerun-if-changed=build/main.rs");
    println!("cargo:rerun-if-changed=build/shaders.rs");
    
    compile_shaders();
}