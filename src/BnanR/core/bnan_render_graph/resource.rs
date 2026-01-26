use ash::vk;

use crate::core::ArcMut;
use crate::core::bnan_image::BnanImage;
use crate::core::bnan_buffer::BnanBuffer;
use crate::core::bnan_rendering::FRAMES_IN_FLIGHT;


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceUsage {
    ColorAttachment,
    DepthStencilAttachment,
    DepthStencilResolve,
    ShaderRead,
    StorageRead,
    StorageWrite,
    TransferSrc,
    TransferDst,
    Present,
}

impl ResourceUsage {
    pub fn get_stage_and_access(self) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
        match self {
            ResourceUsage::ColorAttachment => (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),
            ResourceUsage::DepthStencilAttachment => (
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ),
            ResourceUsage::DepthStencilResolve => (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),
            ResourceUsage::ShaderRead => (
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_READ,
            ),
            ResourceUsage::StorageRead => (
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            ResourceUsage::StorageWrite => (
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
            ),
            ResourceUsage::TransferSrc => (
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            ResourceUsage::TransferDst => (
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            ResourceUsage::Present => (
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
            ),
        }
    }
    
    pub fn get_layout(self) -> vk::ImageLayout {
        match self {
            ResourceUsage::ColorAttachment => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ResourceUsage::DepthStencilAttachment => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ResourceUsage::DepthStencilResolve => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ResourceUsage::ShaderRead => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ResourceUsage::StorageRead | ResourceUsage::StorageWrite => vk::ImageLayout::GENERAL,
            ResourceUsage::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ResourceUsage::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ResourceUsage::Present => vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }
}

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
            current_stage: vk::PipelineStageFlags2::NONE,
            current_queue_family: vk::QUEUE_FAMILY_IGNORED,
        }
    }
}
