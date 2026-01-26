use ash::vk;
use crate::core::bnan_render_graph::graph::BnanRenderGraph;
use crate::core::bnan_rendering::BnanFrameInfo;
use crate::core::bnan_render_graph::resource::{ResourceHandle, ResourceUsage};

#[derive(Clone)]
pub struct AttachmentConfig {
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub clear_value: vk::ClearValue,
}

impl Default for AttachmentConfig {
    fn default() -> Self {
        Self {
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: vk::ClearValue { color: vk::ClearColorValue { float32: [0.0065, 0.0065, 0.0065, 1.0] } },
        }
    }
}

pub struct RenderPassResource {
    pub handle: ResourceHandle,
    pub usage: ResourceUsage,
    pub resolve_target: Option<ResourceHandle>,
    pub is_temporal: bool,
    pub config: AttachmentConfig,
}

impl RenderPassResource {
    pub fn new(handle: ResourceHandle, usage: ResourceUsage) -> Self {
        let mut config = AttachmentConfig::default();
        if matches!(usage, ResourceUsage::DepthStencilAttachment) {
            config.clear_value = vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 }};
        }
        
        Self {
            handle,
            usage,
            resolve_target: None,
            is_temporal: false,
            config,
        }
    }
    
    pub fn with_resolve(handle: ResourceHandle, usage: ResourceUsage, resolve: ResourceHandle) -> Self {
        let mut config = AttachmentConfig::default();
        if matches!(usage, ResourceUsage::DepthStencilAttachment) {
            config.clear_value = vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 }};
        }
        
        Self {
            handle,
            usage,
            resolve_target: Some(resolve),
            is_temporal: false,
            config,
        }
    }
    
    pub fn temporal(handle: ResourceHandle, usage: ResourceUsage) -> Self {
        Self {
            handle,
            usage,
            resolve_target: None,
            is_temporal: true,
            config: AttachmentConfig::default(),
        }
    }
    
    pub fn set_load_op(mut self, op: vk::AttachmentLoadOp) -> Self {
        self.config.load_op = op;
        self
    }
    
    pub fn set_store_op(mut self, op: vk::AttachmentStoreOp) -> Self {
        self.config.store_op = op;
        self
    }
    
    pub fn set_clear_color(mut self, color: [f32; 4]) -> Self {
        self.config.clear_value = vk::ClearValue { color: vk::ClearColorValue { float32: color } };
        self
    }
    
    pub fn set_clear_depth(mut self, depth: f32, stencil: u32) -> Self {
        self.config.clear_value = vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth, stencil } };
        self
    }

    pub fn create_color_attachment_info(&self, view: vk::ImageView, resolve_view: Option<vk::ImageView>) -> vk::RenderingAttachmentInfo {
        let mut info = self.create_base_attachment_info(view, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        if let Some(rv) = resolve_view {
            info = info.resolve_image_view(rv)
                .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .resolve_mode(vk::ResolveModeFlags::AVERAGE);
        }
        info
    }

    pub fn create_depth_attachment_info(&self, view: vk::ImageView, resolve_view: Option<vk::ImageView>) -> vk::RenderingAttachmentInfo {
        let mut info = self.create_base_attachment_info(view, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        if let Some(rv) = resolve_view {
            info = info.resolve_image_view(rv)
                .resolve_image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .resolve_mode(vk::ResolveModeFlags::MAX);
        }
        info
    }

    fn create_base_attachment_info(&self, view: vk::ImageView, layout: vk::ImageLayout) -> vk::RenderingAttachmentInfo {
        vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(layout)
            .load_op(self.config.load_op)
            .store_op(self.config.store_op)
            .clear_value(self.config.clear_value)
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
