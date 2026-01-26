use std::result::Result::Ok;
use std::collections::HashMap;
use std::time::Instant;

use anyhow::*;
use ash::*;

use crate::core::{make_arcmut, ArcMut};
use crate::core::bnan_device::{BnanBarrierBuilder, BnanDevice};
use crate::core::bnan_swapchain::BnanSwapchain;
use crate::core::bnan_window::BnanWindow;
use crate::core::bnan_rendering::{BnanFrameInfo, FRAMES_IN_FLIGHT}; 
use crate::core::bnan_render_graph::resource::{ResourceHandle, ResourceType, PhysicalResource, ResourceUsage};
use crate::core::bnan_render_graph::pass::RenderPass;
use crate::core::bnan_image::BnanImage;
use crate::core::bnan_render_graph::sync::RenderGraphSync;

pub struct BnanRenderGraph {
    pub device: ArcMut<BnanDevice>,
    pub swapchain: ArcMut<BnanSwapchain>,
    pub window: ArcMut<BnanWindow>,
    pub sync: RenderGraphSync,
    
    pub passes: Vec<RenderPass>,
    pub resources: HashMap<usize, PhysicalResource>,
    pub resource_counter: usize,
    
    pub current_frame: usize,
    
    pub swapchain_resource_handle: Option<ResourceHandle>,
    pub frame_time_reference: Instant,
}

impl BnanRenderGraph {
    pub fn new(window: ArcMut<BnanWindow>, device: ArcMut<BnanDevice>, swapchain: ArcMut<BnanSwapchain>) -> Result<Self> {
        
        let swapchain_image_count = swapchain.lock().unwrap().images.len();
        let sync = RenderGraphSync::new(device.clone(), swapchain_image_count as u32)?;
        
        Ok(Self {
            device,
            swapchain,
            window,
            sync,
            passes: Vec::new(),
            resources: HashMap::new(),
            resource_counter: 0,
            current_frame: 0,
            swapchain_resource_handle: Some(ResourceHandle(0)),
            frame_time_reference: Instant::now(),
        })
    }

    pub fn get_backbuffer_handle(&self) -> ResourceHandle {
        self.swapchain_resource_handle.clone().unwrap()
    }

    pub fn get_image(&self, handle: &ResourceHandle, frame: usize) -> Option<ArcMut<BnanImage>> {
        self.resources.get(&handle.0).and_then(|physical| {
            match &physical.resource {
                ResourceType::Image(images) => Some(images[frame].clone()),
                ResourceType::SwapchainImage(image) => Some(image.clone()),
                _ => None,
            }
        })
    }

    pub fn get_images(&self, handle: &ResourceHandle) -> Option<[ArcMut<BnanImage>; FRAMES_IN_FLIGHT]> {
        self.resources.get(&handle.0).and_then(|physical| {
            match &physical.resource {
                ResourceType::Image(images) => Some(images.clone()),
                _ => None,
            }
        })
    }

    pub fn get_previous_frame_image(&self, handle: &ResourceHandle, current_frame: usize) -> Option<ArcMut<BnanImage>> {
        let prev_frame = (current_frame + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT;
        self.get_image(handle, prev_frame)
    }

    pub fn import_render_image(&mut self, name: &str, images: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT]) -> ResourceHandle {
        self.resource_counter += 1;
        if self.resource_counter <= 1 { self.resource_counter = 1; }
        
        let handle = ResourceHandle(self.resource_counter);
        self.resource_counter += 1;
        
        let physical = PhysicalResource::new_image(handle.clone(), name.to_string(), images);
        self.resources.insert(handle.0, physical);
        
        handle
    }
    
    pub fn update_render_image(&mut self, handle: &ResourceHandle, images: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT]) {
        
        if let Some(physical) = self.resources.get_mut(&handle.0) {
            physical.resource = ResourceType::Image(images);
            physical.current_layout = vk::ImageLayout::UNDEFINED;
            physical.current_stage = vk::PipelineStageFlags2::NONE;
        }
    }
    
    pub fn add_pass(&mut self, pass: RenderPass) {
        self.passes.push(pass);
    }

    fn get_src_access_and_stage(
        current_layout: vk::ImageLayout,
        current_stage: vk::PipelineStageFlags2,
        is_swapchain: bool,
    ) -> (vk::AccessFlags2, vk::PipelineStageFlags2) {
        // Swapchain from UNDEFINED syncs with acquire semaphore at COLOR_ATTACHMENT_OUTPUT
        if is_swapchain && current_layout == vk::ImageLayout::UNDEFINED {
            return (vk::AccessFlags2::NONE, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        }
        
        match current_layout {
            vk::ImageLayout::UNDEFINED => (vk::AccessFlags2::NONE, current_stage),
            vk::ImageLayout::GENERAL => (vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::PipelineStageFlags2::ALL_COMMANDS),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (vk::AccessFlags2::COLOR_ATTACHMENT_WRITE, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
                // Depth resolve uses COLOR_ATTACHMENT_OUTPUT stage with COLOR_ATTACHMENT_WRITE access for resolve
                // But for regular attachment it's DEPTH_STENCIL_ATTACHMENT_WRITE
                // If we are coming from a resolve, current_stage would look like COLOR_ATTACHMENT_OUTPUT (from barrier logic)
                if current_stage == vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT {
                     (vk::AccessFlags2::COLOR_ATTACHMENT_WRITE, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                } else {
                     (vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE, vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                }
            },
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (vk::AccessFlags2::TRANSFER_READ, vk::PipelineStageFlags2::TRANSFER),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => (vk::AccessFlags2::TRANSFER_WRITE, vk::PipelineStageFlags2::TRANSFER),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (vk::AccessFlags2::SHADER_READ, vk::PipelineStageFlags2::FRAGMENT_SHADER),
            vk::ImageLayout::PRESENT_SRC_KHR => (vk::AccessFlags2::NONE, vk::PipelineStageFlags2::NONE),
            _ => (vk::AccessFlags2::NONE, current_stage),
        }
    }
    
    fn get_aspect_mask_for_format(format: vk::Format) -> vk::ImageAspectFlags {
        match format {
            vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
                vk::ImageAspectFlags::DEPTH
            }
            vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT => {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            }
            vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
            _ => vk::ImageAspectFlags::COLOR,
        }
    }
    
    pub fn execute(&mut self) -> Result<()> {
        
        let (swapchain_loader, swapchain_khr, swapchain_images, swapchain_views, swapchain_extent, swapchain_format) = {
             let device = self.device.lock().unwrap();
             let swapchain = self.swapchain.lock().unwrap();

             (
                 khr::swapchain::Device::new(&device.instance, &device.device),
                 swapchain.swapchain,
                 swapchain.images.clone(),
                 swapchain.image_views.clone(),
                 swapchain.extent,
                 swapchain.surface_format.format
             )
        };

        self.sync.wait_and_reset_in_flight(self.current_frame)?;
        
        let acquire_result = unsafe {
            swapchain_loader.acquire_next_image(
                swapchain_khr,
                u64::MAX,
                self.sync.image_available_semaphores[self.current_frame],
                vk::Fence::null()
            )
        };
        
        let image_index = match acquire_result {
            Ok((idx, _)) => idx,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                 self.recreate_swapchain()?;
                 return Ok(());
            }
            Err(e) => return Err(Error::new(e)),
        };
        
        let command_buffer = self.sync.get_command_buffer(self.current_frame);
        let swapchain_image = swapchain_images[image_index as usize];
        let swapchain_view = swapchain_views[image_index as usize];
        
        let swapchain_resource_image = make_arcmut(BnanImage::from_image(
            self.device.clone(),
            swapchain_image,
            swapchain_view,
            swapchain_format,
            vk::Extent3D { width: swapchain_extent.width, height: swapchain_extent.height, depth: 1 }
        ));
        
        if self.swapchain_resource_handle.is_none() {
            let handle = ResourceHandle(self.resource_counter);
            self.resource_counter += 1;
            self.swapchain_resource_handle = Some(handle);
        }
        
        let swapchain_handle = self.swapchain_resource_handle.clone().unwrap();
        
        // Update swapchain physical resource
        let swapchain_physical = PhysicalResource {
            handle: swapchain_handle.clone(),
            name: "Backbuffer".to_string(),
            resource: ResourceType::SwapchainImage(swapchain_resource_image.clone()),
            current_layout: vk::ImageLayout::UNDEFINED,
            current_stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            current_queue_family: vk::QUEUE_FAMILY_IGNORED,
        };
        self.resources.insert(swapchain_handle.0, swapchain_physical);

        {
            let device = self.device.lock().unwrap();
            unsafe {
                device.device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
                let begin_info = vk::CommandBufferBeginInfo::default();
                device.device.begin_command_buffer(command_buffer, &begin_info)?;
            }
        }
        
        let frame_time = self.frame_time_reference.elapsed().as_nanos() as f32 / 1000.0;
        self.frame_time_reference = Instant::now();
        
        let frame_info = BnanFrameInfo {
            frame_index: self.current_frame,
            frame_time, 
            command_pool: vk::CommandPool::null(),
            main_command_buffer: command_buffer,
            swapchain_image: swapchain_resource_image,
        };

        for pass in &self.passes {
            {
                let device = self.device.lock().unwrap();
                let mut barrier_builder = BnanBarrierBuilder::new();

                let mut process_resource = |handle: &ResourceHandle, usage: ResourceUsage, use_frame: usize, builder: &mut BnanBarrierBuilder| {
                    let (required_stage, required_access) = usage.get_stage_and_access();
                    let required_layout = usage.get_layout();

                    if let Some(physical) = self.resources.get_mut(&handle.0) {
                        let is_image = matches!(&physical.resource, ResourceType::SwapchainImage(_) | ResourceType::Image(_));
                        let needs_barrier = (is_image && physical.current_layout != required_layout)
                            || physical.current_stage != required_stage;

                        if needs_barrier {
                            match &physical.resource {
                                ResourceType::SwapchainImage(image_arc) => {
                                    let image = image_arc.lock().unwrap();
                                    let (src_access, src_stage) = Self::get_src_access_and_stage(
                                        physical.current_layout, physical.current_stage, true
                                    );
                                    
                                    // Use raw barrier push for flexibility or use builder's transition
                                    // Here we use push_barrier to have full control over access flags
                                    let barrier = vk::ImageMemoryBarrier2::default()
                                        .old_layout(physical.current_layout)
                                        .new_layout(required_layout)
                                        .image(image.image)
                                        .subresource_range(vk::ImageSubresourceRange {
                                            aspect_mask: vk::ImageAspectFlags::COLOR,
                                            base_mip_level: 0,
                                            level_count: 1,
                                            base_array_layer: 0,
                                            layer_count: 1,
                                        })
                                        .src_access_mask(src_access)
                                        .src_stage_mask(src_stage)
                                        .dst_access_mask(required_access)
                                        .dst_stage_mask(required_stage);

                                    builder.push_barrier(barrier);
                                },

                                ResourceType::Image(images) => {
                                    let image = images[use_frame].lock().unwrap();
                                    let (src_access, src_stage) = Self::get_src_access_and_stage(
                                        physical.current_layout, physical.current_stage, false
                                    );

                                    let aspect_mask = Self::get_aspect_mask_for_format(image.format);

                                    let barrier = vk::ImageMemoryBarrier2::default()
                                        .old_layout(physical.current_layout)
                                        .new_layout(required_layout)
                                        .image(image.image)
                                        .subresource_range(vk::ImageSubresourceRange {
                                            aspect_mask,
                                            base_mip_level: 0,
                                            level_count: image.mip_levels,
                                            base_array_layer: 0,
                                            layer_count: 1,
                                        })
                                        .src_access_mask(src_access)
                                        .src_stage_mask(src_stage)
                                        .dst_access_mask(required_access)
                                        .dst_stage_mask(required_stage);

                                    builder.push_barrier(barrier);
                                },

                                ResourceType::Buffer(_) => {
                                    // todo buffers
                                }
                            }

                            physical.current_layout = required_layout;
                            physical.current_stage = required_stage;
                        }
                    }
                };

                for input in &pass.inputs {
                    let use_frame = if input.is_temporal {
                        (frame_info.frame_index + (FRAMES_IN_FLIGHT - 1)) % FRAMES_IN_FLIGHT
                    } else {
                        frame_info.frame_index
                    };
                    process_resource(&input.handle, input.usage, use_frame, &mut barrier_builder);
                }

                for output in &pass.outputs {
                    process_resource(&output.handle, output.usage, frame_info.frame_index, &mut barrier_builder);
                    if let Some(resolve_handle) = &output.resolve_target {
                        let resolve_usage = match output.usage {
                            ResourceUsage::DepthStencilAttachment => ResourceUsage::DepthStencilResolve,
                            _ => ResourceUsage::ColorAttachment,
                        };
                        process_resource(resolve_handle, resolve_usage, frame_info.frame_index, &mut barrier_builder);
                    }
                }

                barrier_builder.record(&device, command_buffer);
            }

            // --- Dynamic Rendering Setup ---
            let mut is_dynamic_rendering = false;
            let mut color_attachments = Vec::new();
            let mut depth_attachment: Option<vk::RenderingAttachmentInfo> = None;
            let mut render_extent: Option<vk::Extent2D> = None;
             
            for output in &pass.outputs {
                if matches!(output.usage, ResourceUsage::ColorAttachment) {
                    if let Some(physical) = self.resources.get(&output.handle.0) {
                        let image_guard = match &physical.resource {
                            ResourceType::SwapchainImage(image_arc) => Some(image_arc.lock().unwrap()),
                            ResourceType::Image(images) => Some(images[frame_info.frame_index].lock().unwrap()),
                            ResourceType::Buffer(_) => None,
                        };
                        
                        if let Some(image) = image_guard {
                            // Resolve handling
                            let mut resolve_view = None;
                            if let Some(resolve_handle) = &output.resolve_target {
                                if let Some(resolve_physical) = self.resources.get(&resolve_handle.0) {
                                    let resolve_image_guard = match &resolve_physical.resource {
                                        ResourceType::SwapchainImage(image_arc) => Some(image_arc.lock().unwrap()),
                                        ResourceType::Image(images) => Some(images[frame_info.frame_index].lock().unwrap()),
                                        ResourceType::Buffer(_) => None,
                                    };
                                    if let Some(resolve_image) = resolve_image_guard {
                                        resolve_view = Some(resolve_image.image_view);
                                    }
                                }
                            }

                            let attachment = output.create_color_attachment_info(image.image_view, resolve_view);
                            color_attachments.push(attachment);
                            is_dynamic_rendering = true;
                             
                            if render_extent.is_none() {
                                render_extent = Some(vk::Extent2D {
                                    width: image.image_extent.width,
                                    height: image.image_extent.height,
                                });
                            }
                        }
                    }
                } else if matches!(output.usage, ResourceUsage::DepthStencilAttachment) {
                    if let Some(physical) = self.resources.get(&output.handle.0) {
                        let image_guard = match &physical.resource {
                            ResourceType::SwapchainImage(image_arc) => Some(image_arc.lock().unwrap()),
                            ResourceType::Image(images) => Some(images[frame_info.frame_index].lock().unwrap()),
                            ResourceType::Buffer(_) => None,
                        };

                        if let Some(image) = image_guard {
                            // Resolve handling
                            let mut resolve_view = None;
                            if let Some(resolve_handle) = &output.resolve_target {
                                if let Some(resolve_physical) = self.resources.get(&resolve_handle.0) {
                                    let resolve_image_guard = match &resolve_physical.resource {
                                        ResourceType::SwapchainImage(image_arc) => Some(image_arc.lock().unwrap()),
                                        ResourceType::Image(images) => Some(images[frame_info.frame_index].lock().unwrap()),
                                        ResourceType::Buffer(_) => None,
                                    };
                                    if let Some(resolve_image) = resolve_image_guard {
                                        resolve_view = Some(resolve_image.image_view);
                                    }
                                }
                            }

                            depth_attachment = Some(output.create_depth_attachment_info(image.image_view, resolve_view));
                            is_dynamic_rendering = true;
                             
                            if render_extent.is_none() {
                                render_extent = Some(vk::Extent2D {
                                    width: image.image_extent.width,
                                    height: image.image_extent.height,
                                });
                            }
                        }
                    }
                }
            }
             
            if is_dynamic_rendering {
                let extent = render_extent.unwrap_or(swapchain_extent);
                 
                let mut rendering_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D { offset: vk::Offset2D{x:0, y:0}, extent })
                    .layer_count(1)
                    .color_attachments(&color_attachments);
                     
                if let Some(depth) = &depth_attachment {
                    rendering_info = rendering_info.depth_attachment(depth);
                }
                     
                let device = self.device.lock().unwrap();
                unsafe { device.device.cmd_begin_rendering(command_buffer, &rendering_info); }
            }

            match &pass.execute {
                f => f(&self, &frame_info),
            }
             
            if is_dynamic_rendering {
                let device = self.device.lock().unwrap();
                unsafe { device.device.cmd_end_rendering(command_buffer); }
            }
        }

        // --- Final Swapchain Transition ---
        {
            let device = self.device.lock().unwrap();
            let swapchain_handle = self.swapchain_resource_handle.clone().unwrap();
            let physical = self.resources.get(&swapchain_handle.0).unwrap();

            let builder = BnanBarrierBuilder::new()
                .transition_image_layout(swapchain_image, physical.current_layout, vk::ImageLayout::PRESENT_SRC_KHR, None, None)?
                .record(&*device, command_buffer);

            unsafe {
                device.device.end_command_buffer(command_buffer)?;
            }
        }
        
        let mut wait_semaphores_vec = vec![
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.sync.image_available_semaphores[self.current_frame])
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .value(1)
        ];
        
        if self.sync.pending_transfer_signal[self.current_frame] {
            wait_semaphores_vec.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.sync.transfer_finished_semaphores[self.current_frame])
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .value(1)
            );
            self.sync.pending_transfer_signal[self.current_frame] = false;
        }
        
        let signal_semaphores = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.sync.render_finished_semaphores[image_index as usize])
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .value(1)
        ];
        let command_buffer_infos = [
             vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)
        ];

        let submit_info = [vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_semaphores_vec)
            .signal_semaphore_infos(&signal_semaphores)
            .command_buffer_infos(&command_buffer_infos)
        ];
        
        let graphics_queue = self.device.lock().unwrap().graphics_queue;

        {
             let device = self.device.lock().unwrap();
             unsafe {
                device.device.queue_submit2(graphics_queue, &submit_info, self.sync.in_flight_fences[self.current_frame])?;
             }
        }
        
        let wait_semaphores_present = [self.sync.render_finished_semaphores[image_index as usize]];
        let swapchains = [swapchain_khr];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores_present)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            let result = swapchain_loader.queue_present(graphics_queue, &present_info);
             match result {
                Ok(_) => {},
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    self.recreate_swapchain()?;
                }
                Err(e) => return Err(Error::new(e)),
            }
        }
        
        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;
        Ok(())
    }
    
    pub fn recreate_swapchain(&mut self) -> Result<()> {
         let extent = self.window.lock().unwrap().get_window_extent();
         self.swapchain.lock().unwrap().recreate_swapchain(extent)?;
         
         let image_count = self.swapchain.lock().unwrap().images.len();
         self.sync.recreate_semaphores(image_count)?;
         
         Ok(())
    }

    pub fn submit_transfer_work<F>(&mut self, frame: usize, commands: F) -> Result<()>
    where
        F: FnOnce(vk::CommandBuffer, &BnanDevice),
    {
        let device = self.device.lock().unwrap();
        let cmd_buffer = self.sync.transfer_command_buffers[frame];
        
        unsafe {
            device.device.wait_for_fences(&[self.sync.transfer_fences[frame]], true, u64::MAX)?;
            device.device.reset_fences(&[self.sync.transfer_fences[frame]])?;
            device.device.reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())?;
            
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.device.begin_command_buffer(cmd_buffer, &begin_info)?;
        }
        
        commands(cmd_buffer, &device);
        
        unsafe {
            device.device.end_command_buffer(cmd_buffer)?;
            
            let signal_semaphores = [
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.sync.transfer_finished_semaphores[frame])
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .value(1)
            ];
            let command_buffer_infos = [
                vk::CommandBufferSubmitInfo::default().command_buffer(cmd_buffer)
            ];
            
            let submit_info = [vk::SubmitInfo2::default()
                .signal_semaphore_infos(&signal_semaphores)
                .command_buffer_infos(&command_buffer_infos)
            ];
            
            device.device.queue_submit2(device.transfer_queue, &submit_info, self.sync.transfer_fences[frame])?;
        }
        
        self.sync.pending_transfer_signal[frame] = true;
        Ok(())
    }
    
    pub fn has_pending_transfers(&self, frame: usize) -> bool {
        self.sync.pending_transfer_signal[frame]
    }
    
    pub fn get_current_frame(&self) -> usize {
        self.current_frame
    }
    
    pub fn get_queue_indices(&self) -> Result<crate::core::bnan_device::QueueIndices> {
        self.device.lock().unwrap().get_queue_indices()
    }
}
