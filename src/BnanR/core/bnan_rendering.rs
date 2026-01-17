use ash::*;

use crate::core::ArcMut;
use crate::core::bnan_image::BnanImage;

pub struct BnanFrameInfo {
    pub frame_index: usize,
    pub frame_time: f32,
    pub command_pool: vk::CommandPool,
    pub main_command_buffer: vk::CommandBuffer,
    pub swapchain_image: ArcMut<BnanImage>,
}

pub const FRAMES_IN_FLIGHT: usize = 2;

impl BnanFrameInfo {
    pub fn new(command_pool: vk::CommandPool, main_command_buffer: vk::CommandBuffer, swapchain_image: ArcMut<BnanImage>) -> BnanFrameInfo {
        BnanFrameInfo {
            frame_index: 0,
            frame_time: 0.0,
            command_pool,
            main_command_buffer,
            swapchain_image,
        }
    }
}