use ash::*;
use cgmath::*;

use crate::core::ArcMut;
use crate::core::bnan_buffer::BnanBuffer;
use crate::core::bnan_device::BnanDevice;

#[repr(C)]
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
        normal.location = 0;
        normal.format = vk::Format::R32G32B32_SFLOAT;
        normal.offset = std::mem::offset_of!(Vertex, normal) as u32;

        let mut tangent = vk::VertexInputAttributeDescription::default();
        tangent.binding = 1;
        tangent.location = 1;
        tangent.format = vk::Format::R32G32B32_SFLOAT;
        tangent.offset = std::mem::offset_of!(Vertex, tangent) as u32;

        let mut uv = vk::VertexInputAttributeDescription::default();
        uv.binding = 1;
        uv.location = 2;
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

    pub fn new(device: ArcMut<BnanDevice>, positions: BnanBuffer, vertices: BnanBuffer, indices: BnanBuffer) -> Self {
        Self {
            device,
            positions,
            vertices,
            indices,
        }
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
}