use std::fs::File;
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::Path;

use flate2::write::DeflateEncoder;
use flate2::read::DeflateDecoder;
use flate2::Compression;

use anyhow::*;
use ash::*;
use serde::*;
use bincode;

use crate::core::bnan_mesh::{BnanMeshletDAG, BnanMeshletRawData};

const BPK_MAGIC: u32 = 0x004B5042; // "BPK\0" in little endian
const BPK_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BpkImageHeader {
    width: u32,
    height: u32,
    depth: u32,
    format: i32,
    data_len: u64,
    data_checksum: [u8; 16],
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BpkBufferHeader {
    instance_size: u64,
    instance_count: u32,
    data_len: u64,
    data_checksum: [u8; 16],
}

#[derive(Debug, Clone)]
pub enum BpkEntryData {
    OnDisk { offset: u64, size: u64 },
    InMemory(Vec<u8>),
}

#[derive(Debug, Clone)]
pub enum BpkNode {
    File {
        data: BpkEntryData,
    },

    Directory {
        children: Vec<BpkEntry>,
    },
}

#[derive(Debug, Clone)]
pub struct BpkEntry {
    pub name: String,
    pub node: BpkNode,
}

pub struct BpkArchive {
    file: Option<File>,
    pub root: BpkNode,
}

impl BpkArchive {
    pub fn new() -> Self {
        Self {
            file: None,
            root: BpkNode::Directory { children: Vec::new() },
        }
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)?;
        
        let mut magic_buf = [0u8; 4];
        file.read_exact(&mut magic_buf)?;
        if u32::from_le_bytes(magic_buf) != BPK_MAGIC {
            bail!("Invalid BPK file magic");
        }

        let mut version_buf = [0u8; 4];
        file.read_exact(&mut version_buf)?;
        let version = u32::from_le_bytes(version_buf);
        if version != BPK_VERSION {
             bail!("Unsupported BPK version: {}", version);
        }

        let mut count_buf = [0u8; 8];
        file.read_exact(&mut count_buf)?;
        let count = u64::from_le_bytes(count_buf);

        let mut root = BpkNode::Directory { children: Vec::new() };

        for _ in 0..count {
            // Read Path String
            let mut path_len_buf = [0u8; 4];
            file.read_exact(&mut path_len_buf)?;
            let path_len = u32::from_le_bytes(path_len_buf) as usize;

            let mut path_buf = vec![0u8; path_len];
            file.read_exact(&mut path_buf)?;
            let full_path = String::from_utf8(path_buf).context("Invalid UTF-8 in entry path")?;

            let mut type_buf = [0u8; 1];
            file.read_exact(&mut type_buf)?;
            let is_dir = type_buf[0] == 1;

            if is_dir {
                Self::insert_node(&mut root, &full_path, BpkNode::Directory { children: Vec::new() })?;
            } else {
                let mut offset_buf = [0u8; 8];
                file.read_exact(&mut offset_buf)?;
                let offset = u64::from_le_bytes(offset_buf);

                let mut size_buf = [0u8; 8];
                file.read_exact(&mut size_buf)?;
                let size = u64::from_le_bytes(size_buf);

                let node = BpkNode::File {
                    data: BpkEntryData::OnDisk { offset, size },
                };
                
                Self::insert_node(&mut root, &full_path, node)?;
            }
        }
        
        Ok(Self {
            file: Some(file),
            root,
        })
    }

    fn insert_node(root: &mut BpkNode, path: &str, new_node: BpkNode) -> Result<()> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if parts.is_empty() { return Ok(()); }
        
        let mut current_node = root;
        
        for i in 0..parts.len()-1 {
            let part = parts[i];
            
            if let BpkNode::Directory { children } = current_node {
                match children.binary_search_by(|e| e.name.as_str().cmp(part)) {
                    Result::Ok(idx) => {
                         current_node = &mut children[idx].node;
                    },
                    Err(idx) => {
                        let new_dir = BpkEntry {
                            name: part.to_string(),
                            node: BpkNode::Directory { children: Vec::new() },
                        };
                        children.insert(idx, new_dir);
                        current_node = &mut children[idx].node;
                    }
                }
            } else {
                bail!("Path conflict: {} is a file, expected directory", part);
            }
        }
        
        let leaf_name = parts.last().unwrap();
        if let BpkNode::Directory { children } = current_node {
            let entry = BpkEntry {
                name: leaf_name.to_string(),
                node: new_node,
            };
            match children.binary_search_by(|e| e.name.as_str().cmp(leaf_name)) {
                Result::Ok(idx) => children[idx] = entry,
                Err(idx) => children.insert(idx, entry),
            }
        } else {
            bail!("Parent is not a directory");
        }

        Ok(())
    }

    pub fn save<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        
        struct FlatEntry<'a> {
            path: String,
            node: &'a BpkNode,
        }
        
        let mut flat_entries = Vec::new();
        
        fn collect_recursive<'a>(node: &'a BpkNode, parent_path: String, out: &mut Vec<FlatEntry<'a>>) {
             if let BpkNode::Directory { children } = node {
                 for child in children {
                     let mut current_path = parent_path.clone();
                     if !current_path.is_empty() {
                         current_path.push('/');
                     }
                     current_path.push_str(&child.name);
                     
                     out.push(FlatEntry { path: current_path.clone(), node: &child.node });
                     collect_recursive(&child.node, current_path, out);
                 }
             }
        }
        
        collect_recursive(&self.root, String::new(), &mut flat_entries);

        struct PendingWrite {
             data: Vec<u8>,
        }

        let mut pending_writes = Vec::with_capacity(flat_entries.len());
        
        for entry in &flat_entries {
             if let BpkNode::File { data, .. } = entry.node {
                 match data {
                     BpkEntryData::InMemory(data) => {
                        pending_writes.push(Some(PendingWrite { data: data.clone() }));
                     },

                     BpkEntryData::OnDisk { offset, size } => {
                         if let Some(src_file) = &mut self.file {
                             src_file.seek(SeekFrom::Start(*offset))?;
                             let mut buffer = vec![0u8; *size as usize];
                             src_file.read_exact(&mut buffer)?;
                             pending_writes.push(Some(PendingWrite { data: buffer }));
                         } else {
                             bail!("Missing source file for OnDisk entry: {}", entry.path);
                         }
                     }
                 }
             } else {
                 pending_writes.push(None);
             }
        }

        let mut header_size = 4 + 4 + 8; // Magic + Version + Count
        for entry in &flat_entries {
            // PathLen(4) + Path(len) + Type(1)
            header_size += 4 + entry.path.len() + 1;
            if let BpkNode::File { .. } = entry.node {
                // Offset(8) + Size(8)
                header_size += 8 + 8;
            }
        }

        let file = File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);

        writer.write_all(&BPK_MAGIC.to_le_bytes())?;
        writer.write_all(&BPK_VERSION.to_le_bytes())?;
        writer.write_all(&(flat_entries.len() as u64).to_le_bytes())?;

        let mut current_offset = header_size as u64;

        for (i, entry) in flat_entries.iter().enumerate() {
            writer.write_all(&(entry.path.len() as u32).to_le_bytes())?;
            writer.write_all(entry.path.as_bytes())?;
            
            match entry.node {
                BpkNode::Directory { .. } => {
                    writer.write_all(&[1u8])?; // Type 1 = Directory
                },
                BpkNode::File { .. } => {
                    writer.write_all(&[0u8])?; // Type 0 = File

                    let p_write = pending_writes[i].as_ref().unwrap();
                    let size = p_write.data.len() as u64;
                    
                    writer.write_all(&current_offset.to_le_bytes())?;
                    writer.write_all(&size.to_le_bytes())?;

                    current_offset += size;
                }
            }
        }

        for p_write in pending_writes {
            if let Some(pw) = p_write {
                writer.write_all(&pw.data)?;
            }
        }
        
        writer.flush()?;
        Ok(())
    }

    pub fn add_directory(&mut self, path: &str) -> Result<()> {
        Self::insert_node(&mut self.root, path, BpkNode::Directory { children: Vec::new() })
    }

    pub fn add_item(&mut self, path: &str, data: Vec<u8>) -> Result<()> {
        let node = BpkNode::File {
            data: BpkEntryData::InMemory(data),
        };
        
        Self::insert_node(&mut self.root, path, node)
    }

    pub fn remove_item(&mut self, path: &str) -> Result<()> {
         let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
         if parts.is_empty() { return Ok(()); }
         
         let parent_parts = &parts[0..parts.len()-1];
         let leaf_name = parts.last().unwrap();
         
         let mut current_node = &mut self.root;
         for part in parent_parts {
             if let BpkNode::Directory { children } = current_node {
                 match children.binary_search_by(|e| e.name.as_str().cmp(part)) {
                     Result::Ok(idx) => current_node = &mut children[idx].node,
                     Err(_) => return Ok(()), // Not found, nothing to remove
                 }
             } else {
                 return Ok(());
             }
         }
         
         if let BpkNode::Directory { children } = current_node {
              if let Result::Ok(idx) = children.binary_search_by(|e| e.name.as_str().cmp(leaf_name)) {
                  children.remove(idx);
              }
         }
         
         Ok(())
    }

    pub fn get_node(&self, path: &str) -> Option<&BpkNode> {
         let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
         if parts.is_empty() { return Some(&self.root); }
         
         let mut current_node = &self.root;
         for part in parts {
             if let BpkNode::Directory { children } = current_node {
                 match children.binary_search_by(|e| e.name.as_str().cmp(part)) {
                     Result::Ok(idx) => current_node = &children[idx].node,
                     Err(_) => return None,
                 }
             } else {
                 return None;
             }
         }
         Some(current_node)
    }

    pub fn get_node_mut(&mut self, path: &str) -> Option<&mut BpkNode> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if parts.is_empty() { return Some(&mut self.root); }

        let mut current_node = &mut self.root;
        for part in parts {
            if let BpkNode::Directory { children } = current_node {
                match children.binary_search_by(|e| e.name.as_str().cmp(part)) {
                    Result::Ok(idx) => current_node = &mut children[idx].node,
                    Err(_) => return None,
                }
            } else {
                return None;
            }
        }
        Some(current_node)
    }
    
    pub fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        let node_ptr = self.get_node(path).ok_or_else(|| anyhow!("Path not found: {}", path))?;

        let (data_desc) = match node_ptr {
            BpkNode::File { data } => data.clone(),
            BpkNode::Directory { .. } => bail!("{} in a directory, not a file", path),
        };
        
        match data_desc {
             BpkEntryData::InMemory(d) => Ok(d),
             BpkEntryData::OnDisk { offset, size } => {
                  if let Some( file) = &self.file {

                      let mut file_ref = file;
                      file_ref.seek(SeekFrom::Start(offset))?;

                      let mut reader = file_ref.take(size);
                      let mut data: Vec<u8> = Vec::with_capacity(size as usize);
                      reader.read_to_end(&mut data)?;

                      Ok(data)

                      /*
                      let mut decoder = DeflateDecoder::new(reader);
                      let mut uncompressed = Vec::with_capacity(compressed_size as usize * 2); // heuristic estimate
                      decoder.read_to_end(&mut uncompressed)?;
                      
                      let calc_hash = md5::compute(&uncompressed).0;
                      if calc_hash != checksum {
                          bail!("Checksum mismatch for {}", path);
                      }
                      Ok(uncompressed)
                       */



                  } else {
                      bail!("No source file available");
                  }
             }
        }
    }

    pub fn add_image(&mut self, path: &str, width: u32, height: u32, depth: u32, format: vk::Format, data: Vec<u8>) -> Result<()> {

        let checksum = md5::compute(&data).0;

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&data)?;
        let compressed = encoder.finish()?;

        let header = BpkImageHeader {
            width,
            height,
            depth,
            format: format.as_raw(),
            data_len: data.len() as u64,
            data_checksum: checksum
        };
         
        let header_bytes = bincode::serialize(&header)?;
         
        let mut header = Vec::new();
        header.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
        header.write_all(&header_bytes)?;

        let header_path = format!("{}.meta", path);
        self.add_item(&header_path, header)?;
        self.add_item(path, compressed)
    }

    pub fn add_buffer(&mut self, path: &str, instance_size: u64, instance_count: u32, data: Vec<u8>) -> Result<()> {

        let checksum = md5::compute(&data).0;

        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&data)?;
        let compressed = encoder.finish()?;

        let header = BpkBufferHeader {
            instance_size,
            instance_count,
            data_len: data.len() as u64,
            data_checksum: checksum,
        };

        let header_bytes = bincode::serialize(&header)?;

        let mut header = Vec::new();
        header.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
        header.write_all(&header_bytes)?;

        let header_path = format!("{}.meta", path);
        self.add_item(&header_path, header)?;
        self.add_item(path, compressed)
    }

    pub fn load_image(&self, path: &str) -> Result<(BpkImageHeader, Vec<u8>)> {
        let header_path = format!("{}.meta", path);

        let header_blob = self.read_file(header_path.as_str())?;
        let mut header_cursor = std::io::Cursor::new(&header_blob);

        let mut len_buf = [0u8; 4];
        if header_cursor.read_exact(&mut len_buf).is_err() { bail!("Failed to read image header length"); }
        let header_len = u32::from_le_bytes(len_buf) as usize;

        let mut header_buf = vec![0u8; header_len];
        if header_cursor.read_exact(&mut header_buf).is_err() { bail!("Failed to read image header"); }

        let header: BpkImageHeader = bincode::deserialize(&header_buf)?;

        let blob = self.read_file(path)?;
        let mut decoder = DeflateDecoder::new(blob.as_slice());
        let mut uncompressed = Vec::with_capacity(header.data_len as usize);
        decoder.read_to_end(&mut uncompressed)?;

        Ok((header, uncompressed))
    }

    pub fn load_buffer(&self, path: &str) -> Result<(BpkBufferHeader, Vec<u8>)> {

        let header_path = format!("{}.meta", path);
        let header_blob = self.read_file(header_path.as_str())?;
        let mut cursor = std::io::Cursor::new(&header_blob);
        
        let mut len_buf = [0u8; 4];
        if cursor.read_exact(&mut len_buf).is_err() { bail!("Failed to read buffer header length"); }
        let header_len = u32::from_le_bytes(len_buf) as usize;
        
        let mut header_buf = vec![0u8; header_len];
        if cursor.read_exact(&mut header_buf).is_err() { bail!("Failed to read buffer header"); }
        
        let header: BpkBufferHeader = bincode::deserialize(&header_buf)?;

        let blob = self.read_file(path)?;
        let mut decoder = DeflateDecoder::new(blob.as_slice());
        let mut uncompressed = Vec::with_capacity(header.data_len as usize);
        decoder.read_to_end(&mut uncompressed)?;
        
        Ok((header, uncompressed))
    }

    pub fn add_meshlet(&mut self, path: &str, positions: &[u8], vertices: &[u8], triangles: &[u8]) -> Result<()> {
        self.add_directory(path)?;
        
        let pos_path = format!("{}/positions", path);
        self.add_item(&pos_path, positions.to_vec())?;
        
        let vert_path = format!("{}/vertices", path);
        self.add_item(&vert_path, vertices.to_vec())?;
        
        let tri_path = format!("{}/triangles", path);
        self.add_item(&tri_path, triangles.to_vec())?;
        
        Ok(())
    }

    pub fn load_meshlet(&self, meshlet_path: &str) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let pos_path = format!("{}/positions", meshlet_path);
        let positions = self.read_file(&pos_path)?;
        
        let vert_path = format!("{}/vertices", meshlet_path);
        let vertices = self.read_file(&vert_path)?;
        
        let tri_path = format!("{}/triangles", meshlet_path);
        let triangles = self.read_file(&tri_path)?;
        
        Ok((positions, vertices, triangles))
    }

    pub fn add_meshlet_dag(&mut self, base_path: &str, dag: &BnanMeshletDAG, raw_data: &[BnanMeshletRawData]) -> Result<()> {

        self.add_directory(base_path)?;
        let dag_bytes = bincode::serialize(dag)?;
        let meta_path = format!("{}/dag.meta", base_path);
        self.add_item(&meta_path, dag_bytes)?;
        
        for (idx, raw) in raw_data.iter().enumerate() {
            let meshlet_path = format!("{}/meshlet_{}", base_path, idx);
            
            let positions_bytes: Vec<u8> = raw.positions.iter()
                .flat_map(|v| [v.x.to_le_bytes(), v.y.to_le_bytes(), v.z.to_le_bytes()])
                .flatten()
                .collect();
            
            let vertices_bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(
                    raw.vertices.as_ptr() as *const u8,
                    raw.vertices.len() * size_of::<crate::core::bnan_mesh::Vertex>()
                ).to_vec()
            };
            
            self.add_meshlet(&meshlet_path, &positions_bytes, &vertices_bytes, &raw.triangles)?;
        }
        
        Ok(())
    }

    pub fn load_meshlet_dag(&mut self, base_path: &str) -> Result<BnanMeshletDAG> {
        let meta_path = format!("{}/dag.meta", base_path);
        let dag_bytes = self.read_file(&meta_path)?;
        let dag: BnanMeshletDAG = bincode::deserialize(&dag_bytes)?;
        Ok(dag)
    }
}

