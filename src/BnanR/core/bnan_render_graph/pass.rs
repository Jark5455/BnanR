use ash::vk;

use crate::core::bnan_rendering::BnanFrameInfo;
use crate::core::bnan_render_graph::resource::ResourceHandle;

pub struct RenderPassResource {
    pub handle: ResourceHandle,
    pub stage: vk::PipelineStageFlags2,
    pub layout: vk::ImageLayout,
    pub resolve_target: Option<ResourceHandle>,
}

pub struct RenderPass {
    pub name: String,
    pub inputs: Vec<RenderPassResource>,
    pub outputs: Vec<RenderPassResource>,
    pub execute: Box<dyn Fn(&vk::CommandBuffer, &BnanFrameInfo)>,
}

impl RenderPass {
    pub fn new(
        name: String,
        inputs: Vec<RenderPassResource>,
        outputs: Vec<RenderPassResource>,
        execute: Box<dyn Fn(&vk::CommandBuffer, &BnanFrameInfo)>,
    ) -> Self {
        Self {
            name,
            inputs,
            outputs,
            execute,
        }
    }
}
