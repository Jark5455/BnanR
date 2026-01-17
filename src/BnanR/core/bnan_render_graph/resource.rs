use ash::vk;

use crate::core::ArcMut;
use crate::core::bnan_image::BnanImage;
use crate::core::bnan_buffer::BnanBuffer;
use crate::core::bnan_rendering::FRAMES_IN_FLIGHT;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResourceHandle(pub usize);

#[derive(Clone)]
pub enum ResourceType {
    SwapchainImage(ArcMut<BnanImage>),
    Image([ArcMut<BnanImage>; FRAMES_IN_FLIGHT]),
    Buffer([ArcMut<BnanBuffer>; FRAMES_IN_FLIGHT]),
}

pub struct PhysicalResource {
    pub handle: ResourceHandle,
    pub name: String,
    pub resource: ResourceType,
    
    pub current_layout: vk::ImageLayout,
    pub current_access: vk::AccessFlags2,
    pub current_stage: vk::PipelineStageFlags2,
    pub current_queue_family: u32,
}

impl PhysicalResource {
    pub fn new_swapchain_image(handle: ResourceHandle, name: String, image: ArcMut<BnanImage>) -> Self {
        Self {
            handle,
            name,
            resource: ResourceType::SwapchainImage(image),
            current_layout: vk::ImageLayout::UNDEFINED,
            current_access: vk::AccessFlags2::NONE,
            current_stage: vk::PipelineStageFlags2::NONE,
            current_queue_family: vk::QUEUE_FAMILY_IGNORED,
        }
    }
    
    pub fn new_image(handle: ResourceHandle, name: String, images: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT]) -> Self {
        Self {
            handle,
            name,
            resource: ResourceType::Image(images),
            current_layout: vk::ImageLayout::UNDEFINED,
            current_access: vk::AccessFlags2::NONE,
            current_stage: vk::PipelineStageFlags2::NONE,
            current_queue_family: vk::QUEUE_FAMILY_IGNORED,
        }
    }
    
    pub fn new_buffer(handle: ResourceHandle, name: String, buffers: [ArcMut<BnanBuffer>; FRAMES_IN_FLIGHT]) -> Self {
        Self {
            handle,
            name,
            resource: ResourceType::Buffer(buffers),
            current_layout: vk::ImageLayout::UNDEFINED,
            current_access: vk::AccessFlags2::NONE,
            current_stage: vk::PipelineStageFlags2::NONE,
            current_queue_family: vk::QUEUE_FAMILY_IGNORED,
        }
    }
}
