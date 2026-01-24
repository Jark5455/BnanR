use ash::*;
use anyhow::*;
use cgmath::*;

use crate::core::ArcMut;
use crate::core::bnan_buffer::BnanBuffer;
use crate::core::bnan_device::{BnanDevice, WorkQueue};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub normal: Vector3<f32>,
    pub tangent: Vector3<f32>,
    pub uv: Vector2<f32>,
}

impl Vertex {
    pub fn binding_descriptions() -> Vec<vk::VertexInputBindingDescription> {
        let mut position_binding = vk::VertexInputBindingDescription::default();
        position_binding.binding = 0;
        position_binding.stride = size_of::<Vector3<f32>>() as u32;
        position_binding.input_rate = vk::VertexInputRate::VERTEX;

        let mut attribute_binding = vk::VertexInputBindingDescription::default();
        attribute_binding.binding = 1;
        attribute_binding.stride = size_of::<Vertex>() as u32;
        attribute_binding.input_rate = vk::VertexInputRate::VERTEX;

        vec![position_binding, attribute_binding]
    }

    pub fn attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        let mut position = vk::VertexInputAttributeDescription::default();
        position.binding = 0;
        position.location = 0;
        position.format = vk::Format::R32G32B32_SFLOAT;
        position.offset = 0;

        let mut normal = vk::VertexInputAttributeDescription::default();
        normal.binding = 1;
        normal.location = 1;
        normal.format = vk::Format::R32G32B32_SFLOAT;
        normal.offset = std::mem::offset_of!(Vertex, normal) as u32;

        let mut tangent = vk::VertexInputAttributeDescription::default();
        tangent.binding = 1;
        tangent.location = 2;
        tangent.format = vk::Format::R32G32B32_SFLOAT;
        tangent.offset = std::mem::offset_of!(Vertex, tangent) as u32;

        let mut uv = vk::VertexInputAttributeDescription::default();
        uv.binding = 1;
        uv.location = 3;
        uv.format = vk::Format::R32G32_SFLOAT;
        uv.offset = std::mem::offset_of!(Vertex, uv) as u32;

        vec![position, normal, tangent, uv]
    }
}

pub struct BnanMesh {
    pub device: ArcMut<BnanDevice>,
    pub positions: BnanBuffer,
    pub vertices: BnanBuffer,
    pub indices: BnanBuffer,
}

impl BnanMesh {

    pub fn new(device: ArcMut<BnanDevice>, positions: &Vec<u8>, vertices: &Vec<u8>, indices: &Vec<u8>) -> Result<Self> {

        let (positions, vertices, indices) = Self::create_buffers(device.clone(), positions, vertices, indices)?;

        Ok(Self {
            device,
            positions,
            vertices,
            indices,
        })
    }

    pub fn bind_position(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device.lock().unwrap().device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.positions.buffer], &[0]);
            self.device.lock().unwrap().device.cmd_bind_index_buffer(command_buffer, self.indices.buffer, 0, vk::IndexType::UINT32);
        }
    }

    pub fn bind_all(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device.lock().unwrap().device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.positions.buffer, self.vertices.buffer], &[0, 0]);
            self.device.lock().unwrap().device.cmd_bind_index_buffer(command_buffer, self.indices.buffer, 0, vk::IndexType::UINT32);
        }
    }

    pub fn draw(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device.lock().unwrap().device.cmd_draw_indexed(command_buffer, self.indices.instance_count, 1, 0, 0, 0)
        }
    }

    fn create_buffers(device: ArcMut<BnanDevice>, positions: &Vec<u8>, vertices: &Vec<u8>, indices: &Vec<u8>) -> Result<(BnanBuffer, BnanBuffer, BnanBuffer)> {
        let mut positions_staging_buffer = BnanBuffer::new(device.clone(), size_of::<u8>() as vk::DeviceSize, positions.len() as u32, vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
        let mut vertices_staging_buffer = BnanBuffer::new(device.clone(), size_of::<u8>() as vk::DeviceSize, vertices.len() as u32, vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
        let mut indices_staging_buffer = BnanBuffer::new(device.clone(), size_of::<u8>() as vk::DeviceSize, indices.len() as u32, vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;

        positions_staging_buffer.map()?;
        positions_staging_buffer.write_to_buffer(positions, 0)?;

        vertices_staging_buffer.map()?;
        vertices_staging_buffer.write_to_buffer(vertices, 0)?;

        indices_staging_buffer.map()?;
        indices_staging_buffer.write_to_buffer(indices, 0)?;

        let positions_buffer = BnanBuffer::new(device.clone(), size_of::<Vector3<f32>>() as vk::DeviceSize, positions_staging_buffer.instance_count / size_of::<Vector3<f32>>() as u32, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let vertices_buffer = BnanBuffer::new(device.clone(), size_of::<Vertex>() as vk::DeviceSize, vertices_staging_buffer.instance_count / size_of::<Vertex>() as u32, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let indices_buffer = BnanBuffer::new(device.clone(), size_of::<u32>() as vk::DeviceSize, indices_staging_buffer.instance_count / size_of::<u32>() as u32, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;


        unsafe {
            let device_guard = device.lock().unwrap();
            let fence = device_guard.device.create_fence(&vk::FenceCreateInfo::default(), None)?;

            let command_buffer = device_guard.begin_commands(WorkQueue::TRANSFER, 1)?;

            let positon_copy_region = vk::BufferCopy::default()
                .size(positions_buffer.buffer_size);

            let vertices_copy_region = vk::BufferCopy::default()
                .size(vertices_buffer.buffer_size);

            let indices_copy_region = vk::BufferCopy::default()
                .size(indices_buffer.buffer_size);

            device_guard.device.cmd_copy_buffer(command_buffer[0], positions_staging_buffer.buffer, positions_buffer.buffer, &[positon_copy_region]);
            device_guard.device.cmd_copy_buffer(command_buffer[0], vertices_staging_buffer.buffer, vertices_buffer.buffer, &[vertices_copy_region]);
            device_guard.device.cmd_copy_buffer(command_buffer[0], indices_staging_buffer.buffer, indices_buffer.buffer, &[indices_copy_region]);

            device_guard.submit_commands(WorkQueue::TRANSFER, command_buffer, None, Some(fence.clone()))?;
            device_guard.device.wait_for_fences(&[fence], true, u64::MAX)?;
        }

        Ok((positions_buffer, vertices_buffer, indices_buffer))

    }
}

// ============================================================================
// Meshlet Data Structures
// ============================================================================

use serde::{Serialize, Deserialize};

/// Single meshlet geometry data with offsets into global buffers
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BnanMeshletData {
    pub position_offset: u32,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub triangle_offset: u32,
    pub triangle_count: u32,
}

/// DAG node at a specific LOD level
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BnanMeshletDAGNode {
    pub meshlet: BnanMeshletData,
    pub lod_level: u32,
    pub child_indices: Vec<u32>,
    pub parent_index: Option<u32>,
    pub bounds: Vector4<f32>,
}

/// Complete DAG for a mesh
#[derive(Serialize, Deserialize, Debug)]
pub struct BnanMeshletDAG {
    pub nodes: Vec<BnanMeshletDAGNode>,
    pub root_indices: Vec<u32>,
    pub leaf_indices: Vec<u32>,
    pub max_lod_level: u32,
}

/// Raw meshlet data for storage (separate from DAG metadata)
pub struct BnanMeshletRawData {
    pub positions: Vec<Vector3<f32>>,
    pub vertices: Vec<Vertex>,
    pub triangles: Vec<u8>,
}