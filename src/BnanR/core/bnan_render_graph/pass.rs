use ash::vk;

use crate::core::bnan_render_graph::graph::BnanRenderGraph;
use crate::core::bnan_rendering::BnanFrameInfo;
use crate::core::bnan_render_graph::resource::ResourceHandle;

pub struct RenderPassResource {
    pub handle: ResourceHandle,
    pub stage: vk::PipelineStageFlags2,
    pub layout: vk::ImageLayout,
    pub resolve_target: Option<ResourceHandle>,
    /// If true, use the previous frame's image instead of the current frame's.
    /// Useful for temporal effects like occlusion culling with Hi-Z from frame N-1.
    pub use_previous_frame: bool,
}

pub struct RenderPass {
    pub name: String,
    pub inputs: Vec<RenderPassResource>,
    pub outputs: Vec<RenderPassResource>,
    pub execute: Box<dyn Fn(&BnanRenderGraph, &BnanFrameInfo)>,
}

impl RenderPass {
    pub fn new(
        name: String,
        inputs: Vec<RenderPassResource>,
        outputs: Vec<RenderPassResource>,
        execute: Box<dyn Fn(&BnanRenderGraph, &BnanFrameInfo)>,
    ) -> Self {
        Self {
            name,
            inputs,
            outputs,
            execute,
        }
    }
}
