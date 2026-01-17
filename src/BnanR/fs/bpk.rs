use std::fs::File;
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::Path;

use flate2::write::DeflateEncoder;
use flate2::read::DeflateDecoder;
use flate2::Compression;

use anyhow::{Result, bail, Context, anyhow};
use ash::vk;
use serde::{Serialize, Deserialize};
use bincode;

use crate::core::bnan_image::BnanImage;
use crate::core::bnan_buffer::BnanBuffer;
use crate::core::bnan_device::BnanDevice;
use crate::core::ArcMut;

const BPK_MAGIC: u32 = 0x004B5042; // "BPK\0" in little endian
const BPK_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct BpkImageHeader {
    width: u32,
    height: u32,
    depth: u32,
    format: i32, // vk::Format
    usage: u32,  // vk::ImageUsageFlags
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct BpkBufferHeader {
    instance_size: u64,
    instance_count: u32,
    usage: u32,      // vk::BufferUsageFlags
    properties: u32, // vk::MemoryPropertyFlags
}

#[derive(Debug, Clone)]
pub enum BpkEntryData {
    OnDisk { offset: u64, compressed_size: u64 },
    InMemory(Vec<u8>),
}

#[derive(Debug, Clone)]
pub enum BpkNode {
    File {
        uncompressed_size: u64,
        checksum: [u8; 16],
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

        let mut count_buf = [0u8; 4];
        file.read_exact(&mut count_buf)?;
        let count = u32::from_le_bytes(count_buf);

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

                let mut comp_size_buf = [0u8; 8];
                file.read_exact(&mut comp_size_buf)?;
                let compressed_size = u64::from_le_bytes(comp_size_buf);

                let mut uncomp_size_buf = [0u8; 8];
                file.read_exact(&mut uncomp_size_buf)?;
                let uncompressed_size = u64::from_le_bytes(uncomp_size_buf);

                let mut checksum = [0u8; 16];
                file.read_exact(&mut checksum)?;

                let node = BpkNode::File {
                    uncompressed_size,
                    checksum,
                    data: BpkEntryData::OnDisk { offset, compressed_size },
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
                    Ok(idx) => {
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
                Ok(idx) => children[idx] = entry,
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
             compressed_data: Vec<u8>,
        }
        let mut pending_writes = Vec::with_capacity(flat_entries.len());
        
        for entry in &flat_entries {
             if let BpkNode::File { data, .. } = entry.node {
                 match data {
                    BpkEntryData::InMemory(raw_data) => {
                        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
                        encoder.write_all(raw_data)?;
                        let compressed = encoder.finish()?;
                        pending_writes.push(Some(PendingWrite { compressed_data: compressed }));
                    },
                    BpkEntryData::OnDisk { offset, compressed_size } => {
                         if let Some(src_file) = &mut self.file {
                             src_file.seek(SeekFrom::Start(*offset))?;
                             let mut buffer = vec![0u8; *compressed_size as usize];
                             src_file.read_exact(&mut buffer)?;
                             pending_writes.push(Some(PendingWrite { compressed_data: buffer }));
                         } else {
                             bail!("Missing source file for OnDisk entry: {}", entry.path);
                         }
                    }
                 }
             } else {
                 pending_writes.push(None);
             }
        }

        let mut header_size = 4 + 4 + 4; // Magic + Version + Count
        for entry in &flat_entries {
            // PathLen(4) + Path(len) + Type(1)
            header_size += 4 + entry.path.len() + 1;
            if let BpkNode::File { .. } = entry.node {
                // Offset(8) + CompSize(8) + UncompSize(8) + Checksum(16)
                header_size += 8 + 8 + 8 + 16;
            }
        }

        let file = File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);

        writer.write_all(&BPK_MAGIC.to_le_bytes())?;
        writer.write_all(&BPK_VERSION.to_le_bytes())?;
        writer.write_all(&(flat_entries.len() as u32).to_le_bytes())?;

        let mut current_offset = header_size as u64;

        for (i, entry) in flat_entries.iter().enumerate() {
            writer.write_all(&(entry.path.len() as u32).to_le_bytes())?;
            writer.write_all(entry.path.as_bytes())?;
            
            match entry.node {
                BpkNode::Directory { .. } => {
                    writer.write_all(&[1u8])?; // Type 1 = Directory
                },
                BpkNode::File { uncompressed_size, checksum, .. } => {
                    writer.write_all(&[0u8])?; // Type 0 = File
                    
                    let p_write = pending_writes[i].as_ref().unwrap();
                    let compressed_size = p_write.compressed_data.len() as u64;
                    
                    writer.write_all(&current_offset.to_le_bytes())?;
                    writer.write_all(&compressed_size.to_le_bytes())?;
                    writer.write_all(&uncompressed_size.to_le_bytes())?;
                    writer.write_all(checksum)?;
                    
                    current_offset += compressed_size;
                }
            }
        }

        for p_write in pending_writes {
            if let Some(pw) = p_write {
                writer.write_all(&pw.compressed_data)?;
            }
        }
        
        writer.flush()?;
        Ok(())
    }

    pub fn add_directory(&mut self, path: &str) -> Result<()> {
        Self::insert_node(&mut self.root, path, BpkNode::Directory { children: Vec::new() })
    }

    pub fn add_item(&mut self, path: &str, data: Vec<u8>) -> Result<()> {
        let checksum = md5::compute(&data).0;
        let node = BpkNode::File {
            uncompressed_size: data.len() as u64,
            checksum,
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
                     Ok(idx) => current_node = &mut children[idx].node,
                     Err(_) => return Ok(()), // Not found, nothing to remove
                 }
             } else {
                 return Ok(());
             }
         }
         
         if let BpkNode::Directory { children } = current_node {
              if let Ok(idx) = children.binary_search_by(|e| e.name.as_str().cmp(leaf_name)) {
                  children.remove(idx);
              }
         }
         
         Ok(())
    }

    pub fn get_node(&mut self, path: &str) -> Option<&mut BpkNode> {
         let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
         if parts.is_empty() { return Some(&mut self.root); }
         
         let mut current_node = &mut self.root;
         for part in parts {
             if let BpkNode::Directory { children } = current_node {
                 match children.binary_search_by(|e| e.name.as_str().cmp(part)) {
                     Ok(idx) => current_node = &mut children[idx].node,
                     Err(_) => return None,
                 }
             } else {
                 return None;
             }
         }
         Some(current_node)
    }
    
    pub fn read_file(&mut self, path: &str) -> Result<Vec<u8>> {
        let node_ptr = self.get_node(path).ok_or_else(|| anyhow!("Path not found: {}", path))?;

        let (checksum, data_desc) = match node_ptr {
            BpkNode::File { checksum, data, .. } => (*checksum, data.clone()),
            BpkNode::Directory { .. } => bail!("{} in a directory, not a file", path),
        };
        
        match data_desc {
             BpkEntryData::InMemory(d) => Ok(d),
             BpkEntryData::OnDisk { offset, compressed_size } => {
                  if let Some(file) = &mut self.file {
                      file.seek(SeekFrom::Start(offset))?;

                      let reader = (file as &mut dyn Read).take(compressed_size);
                      
                      let mut decoder = DeflateDecoder::new(reader);
                      let mut uncompressed = Vec::with_capacity(compressed_size as usize * 2); // heuristic estimate
                      decoder.read_to_end(&mut uncompressed)?;
                      
                      let calc_hash = md5::compute(&uncompressed).0;
                      if calc_hash != checksum {
                          bail!("Checksum mismatch for {}", path);
                      }
                      Ok(uncompressed)
                  } else {
                      bail!("No source file available");
                  }
             }
        }
    }

    pub fn add_bnan_image(&mut self, path: &str, image: &BnanImage, usage_flags: vk::ImageUsageFlags, pixel_data: Vec<u8>) -> Result<()> {
         let header = BpkImageHeader {
             width: image.image_extent.width,
             height: image.image_extent.height,
             depth: image.image_extent.depth,
             format: image.format.as_raw(),
             usage: usage_flags.as_raw(),
         };
         
         let header_bytes = bincode::serialize(&header)?;
         
         let mut blob = Vec::new();
         blob.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
         blob.write_all(&header_bytes)?;
         blob.write_all(&pixel_data)?;
         
         self.add_item(path, blob)
    }

    pub fn load_bnan_image(&mut self, path: &str, device: ArcMut<BnanDevice>) -> Result<(BnanImage, Vec<u8>)> {
        let blob = self.read_file(path)?;
        let mut cursor = std::io::Cursor::new(&blob);
        
        let mut len_buf = [0u8; 4];
        if cursor.read_exact(&mut len_buf).is_err() { bail!("Failed to read image header length"); }
        let header_len = u32::from_le_bytes(len_buf) as usize;
        
        let mut header_buf = vec![0u8; header_len];
        if cursor.read_exact(&mut header_buf).is_err() { bail!("Failed to read image header"); }
        
        let header: BpkImageHeader = bincode::deserialize(&header_buf)?;
        
        let mut pixel_data = Vec::new();
        cursor.read_to_end(&mut pixel_data)?;
        
        let extent = vk::Extent3D { width: header.width, height: header.height, depth: header.depth };
        let format = vk::Format::from_raw(header.format);
        let usage = vk::ImageUsageFlags::from_raw(header.usage);
        
        let image = BnanImage::new(device, format, usage, extent, vk::SampleCountFlags::TYPE_1)?;
        
        Ok((image, pixel_data))
    }

    pub fn add_bnan_buffer(&mut self, path: &str, buffer: &mut BnanBuffer, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags) -> Result<()> {
        let ptr = buffer.map().context("Failed to map buffer for saving. Is it HOST_VISIBLE?")?;
        
        let mut data = vec![0u8; buffer.buffer_size as usize];
        unsafe { std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), buffer.buffer_size as usize); }
        buffer.unmap();
        
        let header = BpkBufferHeader {
            instance_size: buffer.instance_size,
            instance_count: buffer.instance_count,
            usage: usage.as_raw(),
            properties: properties.as_raw(),
        };
        
        let header_bytes = bincode::serialize(&header)?;
        
        let mut blob = Vec::new();
        blob.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
        blob.write_all(&header_bytes)?;
        blob.write_all(&data)?;
        
        self.add_item(path, blob)
    }

    pub fn load_bnan_buffer(&mut self, path: &str, device: ArcMut<BnanDevice>) -> Result<BnanBuffer> {
        let blob = self.read_file(path)?;
        let mut cursor = std::io::Cursor::new(&blob);
        
        let mut len_buf = [0u8; 4];
        if cursor.read_exact(&mut len_buf).is_err() { bail!("Failed to read buffer header length"); }
        let header_len = u32::from_le_bytes(len_buf) as usize;
        
        let mut header_buf = vec![0u8; header_len];
        if cursor.read_exact(&mut header_buf).is_err() { bail!("Failed to read buffer header"); }
        
        let header: BpkBufferHeader = bincode::deserialize(&header_buf)?;
        
        let mut data = Vec::new();
        cursor.read_to_end(&mut data)?;
        
        let usage = vk::BufferUsageFlags::from_raw(header.usage);
        let properties = vk::MemoryPropertyFlags::from_raw(header.properties);
        
        let mut buffer = BnanBuffer::new(device, header.instance_size, header.instance_count, usage, properties)?;
        
        if properties.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
             let ptr = buffer.map()?;
             unsafe {
                 std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len().min(buffer.buffer_size as usize));
             }
             buffer.unmap();
        } else {
            println!("Warning: Loaded BnanBuffer is not HOST_VISIBLE. Data from archive was NOT uploaded to GPU.");
        }
        
        Ok(buffer)
    }

    pub fn add_raw_bnan_buffer(&mut self, path: &str, data: Vec<u8>, instance_size: u64, instance_count: u32, usage: u32, properties: u32) -> Result<()> {
         let header = BpkBufferHeader {
            instance_size,
            instance_count,
            usage,
            properties,
        };
        
        let header_bytes = bincode::serialize(&header)?;
        
        let mut blob = Vec::new();
        blob.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
        blob.write_all(&header_bytes)?;
        blob.write_all(&data)?;
        
        self.add_item(path, blob)
    }

    pub fn add_raw_bnan_image(&mut self, path: &str, pixel_data: Vec<u8>, width: u32, height: u32, depth: u32, format: i32, usage: u32) -> Result<()> {
         let header = BpkImageHeader {
             width,
             height,
             depth,
             format,
             usage,
         };
         
         let header_bytes = bincode::serialize(&header)?;
         
         let mut blob = Vec::new();
         blob.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
         blob.write_all(&header_bytes)?;
         blob.write_all(&pixel_data)?;
         
         self.add_item(path, blob)
    }
}
