
use std::env;
use std::fs;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use glob::glob;
use shaderc;

pub fn compile_shaders() {

    let build_dir_pth = env::var("OUT_DIR").unwrap();
    let build_dir = Path::new(&build_dir_pth).parent().unwrap().parent().unwrap();

    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_1 as u32);
    options.set_target_spirv(shaderc::SpirvVersion::V1_4);

    for entry in glob("src/BnanR-Sample-Shaders/**/*.*").expect("glob failed") {
        let path = entry.unwrap();
        
        let filename = path.file_name().unwrap().to_str().unwrap().to_owned();
        let ext = path.extension().unwrap().to_str().unwrap().to_owned();

        println!("cargo:rerun-if-changed={}", path.display());
        let source = fs::read_to_string(path.clone()).unwrap().to_owned();

        let shader_kind = match ext.as_str() {
            "vert" => shaderc::ShaderKind::Vertex,
            "frag" => shaderc::ShaderKind::Fragment,
            "comp" => shaderc::ShaderKind::Compute,
            "mesh" => shaderc::ShaderKind::Mesh,
            _ => { panic!("unknown shader type: {}", ext); }
        };

        let binary = compiler.compile_into_spirv(source.as_str(), shader_kind, filename.as_str(), "main", Some(&options)).expect(format!("Error while compile shader: {}", path.as_os_str().to_str().unwrap()).as_str());

        let mut components = path.components();
        components.next();
        
        let shader_path = components.as_path();
        let out_path = build_dir.join(shader_path).with_extension(format!("{}.spv", ext));
        let out_dir = out_path.parent().unwrap().to_owned();

        if !out_dir.exists() {
            fs::create_dir_all(out_dir).unwrap();
        }

        let mut file = OpenOptions::new()
            .truncate(true)
            .write(true)
            .create(true)
            .open(out_path.clone())
            .unwrap();
        
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(binary.as_binary_u8()).unwrap();

        println!("cargo:rerun-if-changed={}", out_path.display());
    }
}