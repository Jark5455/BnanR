use std::io::Write;

use clap::*;
use anyhow::*;
use ash::*;
use cgmath::*;
use russimp::scene::*;

use image::io::Reader as ImageReader;
use exr::prelude::*;

use BnanR::fs::bpk::{BpkArchive, BpkNode};
use BnanR::core::bnan_mesh::Vertex;

#[derive(Parser)]
#[command(name = "bpk")]
#[command(about = "CLI tool for managing .bpk archives", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Create {
        #[arg(help = "Path to the .bpk archive to create")]
        archive_path: std::path::PathBuf,
        
        #[arg(long, help = "Overwrite if exists")]
        force: bool,
    },

    List {
        #[arg(help = "Path to the .bpk archive")]
        archive_path: std::path::PathBuf,
    },

    Add {
        #[arg(help = "Path to the .bpk archive")]
        archive_path: std::path::PathBuf,
        
        #[arg(help = "Path to the file on disk to add")]
        source_path: std::path::PathBuf,
        
        #[arg(help = "Path inside the archive (e.g. 'textures/sky.png')")]
        internal_path: String,
        
        #[arg(long, help = "Overwrite if exists in archive")]
        overwrite: bool,
    },

    AddMesh {
        #[arg(help = "Path to the .bpk archive")]
        archive_path: std::path::PathBuf,

        #[arg(help = "Path to the mesh file on disk")]
        mesh_path: std::path::PathBuf,

        #[arg(help = "Directory inside the archive to place buffers (e.g. 'models/bunny')")]
        internal_dir: String,
    },

    AddImage {
        #[arg(help = "Path to the .bpk archive")]
        archive_path: std::path::PathBuf,
        
        #[arg(help = "Path to the image file on disk")]
        image_path: std::path::PathBuf,
        
        #[arg(help = "Path inside the archive (e.g. 'textures/sky.bnan')")]
        internal_path: String,
    },

    Remove {
        #[arg(help = "Path to the .bpk archive")]
        archive_path: std::path::PathBuf,
        
        #[arg(help = "Path inside the archive to remove")]
        internal_path: String,
    },

    Read {
        #[arg(help = "Path to the .bpk archive")]
        archive_path: std::path::PathBuf,
        
        #[arg(help = "Path inside the archive to read")]
        internal_path: String,
        
        #[arg(short, long, help = "Output file path (default: stdout)")]
        output: Option<std::path::PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Create { archive_path, force } => {
            if archive_path.exists() && !force {
                return Err(anyhow!("File {:?} already exists. Use --force to overwrite.", archive_path));
            }
            let mut archive = BpkArchive::new();
            archive.save(&archive_path)?;
            println!("Created empty archive: {:?}", archive_path);
        }
        Commands::List { archive_path } => {
            let archive = BpkArchive::open(&archive_path)?;
            println!("Contents of {:?}:", archive_path);
            list_recursive(&archive.root, "");
        }
        Commands::Add { archive_path, source_path, internal_path, overwrite } => {
            let mut archive = if archive_path.exists() {
                 BpkArchive::open(&archive_path)?
            } else {
                 BpkArchive::new()
            };

            if !overwrite && archive.get_node(&internal_path).is_some() {
                 return Err(anyhow!("Path '{}' already exists in archive. Use --overwrite to replace.", internal_path));
            }

            let data = std::fs::read(&source_path).context(format!("Failed to read source file {:?}", source_path))?;
            
            archive.add_item(&internal_path, data)?;
            archive.save(&archive_path)?;
            println!("Added '{}' to archive.", internal_path);
        }
        Commands::AddMesh { archive_path, mesh_path, internal_dir } => {
             let mut archive = if archive_path.exists() {
                 BpkArchive::open(&archive_path)?
            } else {
                 BpkArchive::new()
            };
            
            println!("Processing mesh: {:?}", mesh_path);
            let scene = Scene::from_file(
                mesh_path.to_str().unwrap(), 
                vec![
                    PostProcess::CalculateTangentSpace, 
                    PostProcess::Triangulate, 
                    PostProcess::JoinIdenticalVertices,
                    PostProcess::SortByPrimitiveType,
                    PostProcess::MakeLeftHanded,
                    PostProcess::FlipUVs,
                    PostProcess::GenerateSmoothNormals,
                    PostProcess::GenerateUVCoords
                ]
            )?;

            if let Some(mesh) = scene.meshes.first() {
                let positions: Vec<Vector3<f32>> = mesh.vertices.iter().map(|v| Vector3 {x: v.x, y: v.y, z: v.z}).collect();
                let positions_bytes = unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, positions.len() * 12) }.to_vec();
                
                let mut complex_vertices = Vec::with_capacity(mesh.vertices.len());

                if mesh.normals.is_empty() {
                    bail!("Mesh has no normals");
                }

                if mesh.tangents.is_empty() {
                    bail!("Mesh has no tangents");
                }

                for i in 0..mesh.vertices.len() {

                    let normal = Vector3 {x: mesh.normals[i].x, y: mesh.normals[i].y, z: mesh.normals[i].z};
                    let tangent = Vector3 { x: mesh.tangents[i].x, y: mesh.tangents[i].y, z: mesh.tangents[i].z};

                    let uv = if let Some(uvs) = &mesh.texture_coords[0] {
                        if i < uvs.len() {
                            Vector2 { x: uvs[i].x, y: uvs[i].y }
                        } else { Vector2 {x: 0.0, y: 0.0} }
                    } else { Vector2 {x: 0.0, y: 0.0} };
                    
                    complex_vertices.push(Vertex { normal, tangent, uv });
                }
                let vertices_bytes = unsafe { std::slice::from_raw_parts(complex_vertices.as_ptr() as *const u8, complex_vertices.len() * std::mem::size_of::<Vertex>()) }.to_vec();

                let indices: Vec<u32> = mesh.faces.iter().flat_map(|face| face.0.clone()).collect();
                let indices_bytes = unsafe { std::slice::from_raw_parts(indices.as_ptr() as *const u8, indices.len() * 4) }.to_vec();

                let usage_vertex = vk::BufferUsageFlags::VERTEX_BUFFER.as_raw();
                let usage_index = vk::BufferUsageFlags::INDEX_BUFFER.as_raw();
                let props = vk::MemoryPropertyFlags::DEVICE_LOCAL.as_raw(); // Should be device local usually

                archive.add_directory(&internal_dir)?;
                
                let pos_path = format!("{}/positions", internal_dir);
                archive.add_raw_bnan_buffer(&pos_path, positions_bytes, 12, positions.len() as u32, usage_vertex, props)?;

                let vert_path = format!("{}/vertices", internal_dir);
                archive.add_raw_bnan_buffer(&vert_path, vertices_bytes, std::mem::size_of::<Vertex>() as u64, complex_vertices.len() as u32, usage_vertex, props)?;

                let ind_path = format!("{}/indices", internal_dir);
                archive.add_raw_bnan_buffer(&ind_path, indices_bytes, 4, indices.len() as u32, usage_index, props)?;

                archive.save(&archive_path)?;
                println!("Imported mesh to '{}'", internal_dir);

            } else {
                bail!("No meshes found in file");
            }
        }
        Commands::AddImage { archive_path, image_path, internal_path } => {
             let mut archive = if archive_path.exists() {
                 BpkArchive::open(&archive_path)?
            } else {
                 BpkArchive::new()
            };

            let ext = image_path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
            
            if ext == "exr" {
                let image = read_first_flat_layer_from_file(&image_path)?;
                let resolution = image.attributes.display_window.size;
                let width = resolution.width() as u32;
                let height = resolution.height() as u32;
                
                let channels = &image.layer_data.channel_data.list;
                let channel_count = channels.len();
                
                let pixel_count = (width * height) as usize;
                
                let get_sample = |channel_idx: usize, pixel_idx: usize| -> f32 {
                    match &channels[channel_idx].sample_data {
                         FlatSamples::F16(vec) => vec[pixel_idx].to_f32(),
                         FlatSamples::F32(vec) => vec[pixel_idx],
                         FlatSamples::U32(vec) => vec[pixel_idx] as f32,
                    }
                };

                fn find_channel(channels: &SmallVec<[AnyChannel<FlatSamples>; 4]>, names: &[&str]) -> Option<usize> {
                    for (i, c) in channels.iter().enumerate() {
                        if names.iter().any(|&n| c.name.to_string().to_lowercase() == n.to_lowercase() || c.name.to_string().to_lowercase().ends_with(&format!(".{}", n.to_lowercase()))) {
                            return Some(i);
                        }
                    }
                    None
                }

                let r_idx = find_channel(channels, &["R", "Red", "X"]);
                let g_idx = find_channel(channels, &["G", "Green", "Y"]);
                let b_idx = find_channel(channels, &["B", "Blue", "Z"]);
                let a_idx = find_channel(channels, &["A", "Alpha", "W"]);

                let (format, usage_indices) = if r_idx.is_some() && g_idx.is_some() && b_idx.is_some() && a_idx.is_some() {
                    (vk::Format::R32G32B32A32_SFLOAT, vec![r_idx.unwrap(), g_idx.unwrap(), b_idx.unwrap(), a_idx.unwrap()])
                } else if r_idx.is_some() && g_idx.is_some() && b_idx.is_some() {
                    (vk::Format::R32G32B32_SFLOAT, vec![r_idx.unwrap(), g_idx.unwrap(), b_idx.unwrap()])
                } else if r_idx.is_some() && g_idx.is_some() {
                    (vk::Format::R32G32_SFLOAT, vec![r_idx.unwrap(), g_idx.unwrap()])
                } else if r_idx.is_some() {
                    (vk::Format::R32_SFLOAT, vec![r_idx.unwrap()])
                } else if channel_count == 1 {
                    (vk::Format::R32_SFLOAT, vec![0])
                } else if channel_count == 3 {
                    (vk::Format::R32G32B32_SFLOAT, vec![0, 1, 2])
                } else if channel_count == 4 {
                    (vk::Format::R32G32B32A32_SFLOAT, vec![0, 1, 2, 3])
                } else {
                    bail!("Unsure how to map {} channles", channel_count);
                };

                 let mut raw_data = Vec::with_capacity(pixel_count * usage_indices.len() * 4);
                 
                 for i in 0..pixel_count {
                     for &c_idx in &usage_indices {
                         let sample = get_sample(c_idx, i);
                         raw_data.extend_from_slice(&sample.to_ne_bytes());
                     }
                 }
                 
                 let usage = (vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST).as_raw();
                 archive.add_raw_bnan_image(&internal_path, raw_data, width, height, 1, format.as_raw(), usage)?;

            } else {
                let img = ImageReader::open(&image_path)?.decode()?;
                let width = img.width();
                let height = img.height();
                let data = img.to_rgba8().into_raw();
                
                let format = vk::Format::R8G8B8A8_SRGB.as_raw();
                let usage = (vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST).as_raw();

                archive.add_raw_bnan_image(&internal_path, data, width, height, 1, format, usage)?;
            }
            
            archive.save(&archive_path)?;
            println!("Imported image to '{}'", internal_path);
        }
        Commands::Remove { archive_path, internal_path } => {
             let mut archive = BpkArchive::open(&archive_path)?;
             match archive.get_node(&internal_path) {
                 Some(_) => {
                     archive.remove_item(&internal_path)?;
                     archive.save(&archive_path)?;
                     println!("Removed '{}' from archive.", internal_path);
                 }
                 None => println!("Path '{}' not found in archive.", internal_path),
             }
        }
        Commands::Read { archive_path, internal_path, output } => {
            let mut archive = BpkArchive::open(&archive_path)?;
            let data = archive.read_file(&internal_path)?;
            
            if let Some(out_path) = output {
                std::fs::write(&out_path, data)?;
                println!("Extracted '{}' to {:?}", internal_path, out_path);
            } else {
                std::io::stdout().write_all(&data)?;
            }
        }
    }

    Ok(())
}

fn list_recursive(node: &BpkNode, parent_path: &str) {
    if let BpkNode::Directory { children } = node {
        for child in children {
            let full_path = if parent_path.is_empty() {
                child.name.clone()
            } else {
                format!("{}/{}", parent_path, child.name)
            };
            
            match &child.node {
                BpkNode::Directory { .. } => {
                    println!("DIR  {}", full_path);
                    list_recursive(&child.node, &full_path);
                },
                BpkNode::File { uncompressed_size, .. } => {
                    println!("FILE {} ({} bytes)", full_path, uncompressed_size);
                }
            }
        }
    }
}
