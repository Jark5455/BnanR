

use crate::core::bnan_render_graph::graph::BnanRenderGraph;
use crate::core::bnan_rendering::BnanFrameInfo;
use crate::core::bnan_render_graph::resource::{ResourceHandle, ResourceUsage};

pub struct RenderPassResource {
    pub handle: ResourceHandle,
    pub usage: ResourceUsage,
    pub resolve_target: Option<ResourceHandle>,
    pub is_temporal: bool,
}

impl RenderPassResource {
    pub fn new(handle: ResourceHandle, usage: ResourceUsage) -> Self {
        Self {
            handle,
            usage,
            resolve_target: None,
            is_temporal: false,
        }
    }
    
    pub fn with_resolve(handle: ResourceHandle, usage: ResourceUsage, resolve: ResourceHandle) -> Self {
        Self {
            handle,
            usage,
            resolve_target: Some(resolve),
            is_temporal: false,
        }
    }
    
    pub fn temporal(handle: ResourceHandle, usage: ResourceUsage) -> Self {
        Self {
            handle,
            usage,
            resolve_target: None,
            is_temporal: true,
        }
    }
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
