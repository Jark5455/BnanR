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
use crate::core::bnan_render_graph::resource::{ResourceHandle, ResourceType, PhysicalResource};
use crate::core::bnan_render_graph::pass::RenderPass;
use crate::core::bnan_image::BnanImage;


const CLEAR_COLOR: vk::ClearColorValue = vk::ClearColorValue { float32: [0.0118, 0.0118, 0.0118, 1.0] };
const CLEAR_DEPTH_STENCIL: vk::ClearDepthStencilValue = vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 };

pub struct BnanRenderGraph {
    pub device: ArcMut<BnanDevice>,
    pub swapchain: ArcMut<BnanSwapchain>,
    pub window: ArcMut<BnanWindow>,
    
    pub passes: Vec<RenderPass>,
    pub resources: HashMap<usize, PhysicalResource>,
    pub resource_counter: usize,

    pub command_buffers: [vk::CommandBuffer; FRAMES_IN_FLIGHT],
    pub image_available_semaphores: [vk::Semaphore; FRAMES_IN_FLIGHT],
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: [vk::Fence; FRAMES_IN_FLIGHT],
    
    pub transfer_command_buffers: [vk::CommandBuffer; FRAMES_IN_FLIGHT],
    pub transfer_finished_semaphores: [vk::Semaphore; FRAMES_IN_FLIGHT],
    pub transfer_fences: [vk::Fence; FRAMES_IN_FLIGHT],
    pub pending_transfers: [bool; FRAMES_IN_FLIGHT],
    
    pub current_frame: usize,
    
    pub swapchain_resource_handle: Option<ResourceHandle>,
    pub frame_time_reference: Instant,
}


impl Drop for BnanRenderGraph {
    fn drop(&mut self) {
        unsafe {
             let device = self.device.lock().unwrap();
             device.device.device_wait_idle().unwrap();
             
             for i in 0..FRAMES_IN_FLIGHT {
                device.device.destroy_semaphore(self.image_available_semaphores[i], None);
                device.device.destroy_fence(self.in_flight_fences[i], None);
                
                device.device.destroy_semaphore(self.transfer_finished_semaphores[i], None);
                device.device.destroy_fence(self.transfer_fences[i], None);
             }
             
             for sem in &self.render_finished_semaphores {
                 device.device.destroy_semaphore(*sem, None);
             }
        }
    }
}

impl BnanRenderGraph {
    pub fn new(window: ArcMut<BnanWindow>, device: ArcMut<BnanDevice>, swapchain: ArcMut<BnanSwapchain>) -> Result<Self> {
        
        let swapchain_image_count = swapchain.lock().unwrap().images.len();
        
        let (command_buffers, image_available_semaphores, render_finished_semaphores, in_flight_fences,
             transfer_command_buffers, transfer_finished_semaphores, transfer_fences) = 
            Self::create_sync_objects(device.clone(), swapchain_image_count as u32)?;
        
        Ok(Self {
            device,
            swapchain,
            window,
            passes: Vec::new(),
            resources: HashMap::new(),
            resource_counter: 0,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            transfer_command_buffers,
            transfer_finished_semaphores,
            transfer_fences,
            pending_transfers: [false; FRAMES_IN_FLIGHT],
            current_frame: 0,
            swapchain_resource_handle: Some(ResourceHandle(0)),
            frame_time_reference: Instant::now(),
        })
    }
    
    fn create_sync_objects(device: ArcMut<BnanDevice>, swapchain_image_count: u32) -> Result<(
        [vk::CommandBuffer; FRAMES_IN_FLIGHT],
        [vk::Semaphore; FRAMES_IN_FLIGHT],
        Vec<vk::Semaphore>,
        [vk::Fence; FRAMES_IN_FLIGHT],
        [vk::CommandBuffer; FRAMES_IN_FLIGHT],
        [vk::Semaphore; FRAMES_IN_FLIGHT],
        [vk::Fence; FRAMES_IN_FLIGHT],
    )> {
         let device_guard = device.lock().unwrap();
         
         let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(device_guard.command_pools[0])
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(FRAMES_IN_FLIGHT as u32);
            
         let command_buffers = unsafe { device_guard.device.allocate_command_buffers(&alloc_info)? };
         let command_buffers_array: [vk::CommandBuffer; FRAMES_IN_FLIGHT] = command_buffers.try_into().map_err(|_| anyhow!("Failed"))?;

         let transfer_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(device_guard.command_pools[BnanDevice::TRANSFER_COMMAND_POOL])
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(FRAMES_IN_FLIGHT as u32);
            
         let transfer_command_buffers = unsafe { device_guard.device.allocate_command_buffers(&transfer_alloc_info)? };
         let transfer_command_buffers_array: [vk::CommandBuffer; FRAMES_IN_FLIGHT] = transfer_command_buffers.try_into().map_err(|_| anyhow!("Failed"))?;

         let semaphore_info = vk::SemaphoreCreateInfo::default();
         let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
         
         let mut image_sem = Vec::new();
         let mut render_sem = Vec::new();
         let mut fences = Vec::new();
         let mut transfer_sem = Vec::new();
         let mut transfer_fences_vec = Vec::new();
         
         for _ in 0..FRAMES_IN_FLIGHT {
             unsafe {
                 image_sem.push(device_guard.device.create_semaphore(&semaphore_info, None)?);
                 fences.push(device_guard.device.create_fence(&fence_info, None)?);
                 transfer_sem.push(device_guard.device.create_semaphore(&semaphore_info, None)?);
                 transfer_fences_vec.push(device_guard.device.create_fence(&fence_info, None)?);
             }
         }
         
         for _ in 0..swapchain_image_count {
             unsafe {
                 render_sem.push(device_guard.device.create_semaphore(&semaphore_info, None)?);
             }
         }
         
         Ok((
            command_buffers_array,
            image_sem.try_into().unwrap(),
            render_sem,
            fences.try_into().unwrap(),
            transfer_command_buffers_array,
            transfer_sem.try_into().unwrap(),
            transfer_fences_vec.try_into().unwrap(),
         ))
    }

    pub fn get_backbuffer_handle(&self) -> ResourceHandle {
        self.swapchain_resource_handle.clone().unwrap()
    }

    /// Retrieves the underlying image for a given resource handle and frame index.
    /// Returns None if the handle doesn't exist or isn't an image resource.
    pub fn get_image(&self, handle: &ResourceHandle, frame: usize) -> Option<ArcMut<BnanImage>> {
        self.resources.get(&handle.0).and_then(|physical| {
            match &physical.resource {
                ResourceType::Image(images) => Some(images[frame].clone()),
                ResourceType::SwapchainImage(image) => Some(image.clone()),
                _ => None,
            }
        })
    }

    /// Retrieves all per-frame images for a given resource handle.
    /// Useful for setting up descriptors for all frames in flight before the render loop.
    /// Returns None if the handle doesn't exist or isn't a per-frame image resource.
    pub fn get_images(&self, handle: &ResourceHandle) -> Option<[ArcMut<BnanImage>; FRAMES_IN_FLIGHT]> {
        self.resources.get(&handle.0).and_then(|physical| {
            match &physical.resource {
                ResourceType::Image(images) => Some(images.clone()),
                _ => None,
            }
        })
    }

    /// Retrieves the previous frame's image for a given resource handle.
    /// Useful for temporal effects like occlusion culling with Hi-Z from frame N-1.
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

        {
            let device = self.device.lock().unwrap();
            unsafe {
                device.device.wait_for_fences(&[self.in_flight_fences[self.current_frame]], true, u64::MAX)?;
                device.device.reset_fences(&[self.in_flight_fences[self.current_frame]])?;
            }
        }
        
        let acquire_result = unsafe {
            swapchain_loader.acquire_next_image(
                swapchain_khr,
                u64::MAX,
                self.image_available_semaphores[self.current_frame],
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
        
        let command_buffer = self.command_buffers[self.current_frame];
        let swapchain_image = swapchain_images[image_index as usize];
        let swapchain_view = swapchain_views[image_index as usize];
        
        let swapchain_resource_image = make_arcmut(BnanImage::from_image(
            self.device.clone(),
            swapchain_image,
            swapchain_view,
            swapchain_format,
            vk::Extent3D { width: swapchain_extent.width, height: swapchain_extent.height, depth: 1 }
        ));
        
        // Update the physical resource tracking for swapchain
        if self.swapchain_resource_handle.is_none() {
            let handle = ResourceHandle(self.resource_counter);
            self.resource_counter += 1;
            self.swapchain_resource_handle = Some(handle);
        }
        
        let swapchain_handle = self.swapchain_resource_handle.clone().unwrap();
        
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
             let device = self.device.lock().unwrap();
             let mut barrier_builder = BnanBarrierBuilder::new();
             
             let mut process_resource = |handle: &ResourceHandle, required_stage, required_layout, use_frame: usize, builder: &mut BnanBarrierBuilder| {
                 if let Some(physical) = self.resources.get_mut(&handle.0) {
                     let mut needs_barrier = false;
                     
                     let is_image = matches!(&physical.resource, ResourceType::SwapchainImage(_) | ResourceType::Image(_));
                     if is_image && physical.current_layout != required_layout {
                         needs_barrier = true;
                     }
                     
                     if physical.current_stage != required_stage {
                         needs_barrier = true;
                     }
                     
                     if needs_barrier {
                         match &physical.resource {
                             ResourceType::SwapchainImage(image_arc) => {
                                 let image = image_arc.lock().unwrap();
                                 builder.transition_image_layout(image.image, physical.current_layout, required_layout, None, None).unwrap();
                             },
                             
                             ResourceType::Image(images) => {
                                 let image = images[use_frame].lock().unwrap();
                                 builder.transition_image_layout(image.image, physical.current_layout, required_layout, Some(image.mip_levels), None).unwrap();
                             },
                             
                             ResourceType::Buffer(_) => {
                                 todo!("Buffer barriers")
                             }
                         }
                         
                         physical.current_layout = required_layout;
                         physical.current_stage = required_stage;
                     }
                 }
             };

             for input in &pass.inputs {
                 let use_frame = if input.use_previous_frame {
                     (frame_info.frame_index + (FRAMES_IN_FLIGHT - 1)) % FRAMES_IN_FLIGHT
                 } else {
                     frame_info.frame_index
                 };
                 
                 process_resource(&input.handle, input.stage, input.layout, use_frame, &mut barrier_builder);
             }
             
             for output in &pass.outputs {
                 process_resource(&output.handle, output.stage, output.layout, frame_info.frame_index, &mut barrier_builder);
                 
                 if let Some(resolve_handle) = &output.resolve_target {

                     match output.layout {
                         vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
                             process_resource(
                                 resolve_handle,
                                 vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
                                 vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                 frame_info.frame_index,
                                 &mut barrier_builder,
                             );
                         }

                         _ => {
                             process_resource(
                                 resolve_handle,
                                 vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                                 vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                                 frame_info.frame_index,
                                 &mut barrier_builder
                             );
                         }
                     };
                 }
             }

             barrier_builder.record(&device, command_buffer);
             
             drop(device); // Drop lock before execute
             
             let mut is_dynamic_rendering = false;
             let mut color_attachments = Vec::new();
             let mut depth_attachment: Option<vk::RenderingAttachmentInfo> = None;
             let mut render_extent: Option<vk::Extent2D> = None;
             
             for output in &pass.outputs {
                 if output.layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
                     if let Some(physical) = self.resources.get(&output.handle.0) {
                         let image_guard = match &physical.resource {
                             ResourceType::SwapchainImage(image_arc) => Some(image_arc.lock().unwrap()),
                             ResourceType::Image(images) => Some(images[frame_info.frame_index].lock().unwrap()),
                             ResourceType::Buffer(_) => None,
                         };
                         if let Some(image) = image_guard {

                             let mut clear_value = vk::ClearValue::default();
                             clear_value.color = CLEAR_COLOR;

                             let mut attachment = vk::RenderingAttachmentInfo::default()
                                 .image_view(image.image_view)
                                 .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                                 .load_op(vk::AttachmentLoadOp::CLEAR) 
                                 .store_op(vk::AttachmentStoreOp::STORE)
                                 .clear_value(clear_value);
                             
                             if let Some(resolve_handle) = &output.resolve_target {
                                 if let Some(resolve_physical) = self.resources.get(&resolve_handle.0) {
                                     let resolve_image_guard = match &resolve_physical.resource {
                                         ResourceType::SwapchainImage(image_arc) => Some(image_arc.lock().unwrap()),
                                         ResourceType::Image(images) => Some(images[frame_info.frame_index].lock().unwrap()),
                                         ResourceType::Buffer(_) => None,
                                     };
                                     if let Some(resolve_image) = resolve_image_guard {
                                         attachment = attachment
                                             .resolve_image_view(resolve_image.image_view)
                                             .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                                             .resolve_mode(vk::ResolveModeFlags::AVERAGE);
                                     }
                                 }
                             }
                             
                             color_attachments.push(attachment);
                             is_dynamic_rendering = true;
                             
                             // Use the first attachment's extent as the render area
                             if render_extent.is_none() {
                                 render_extent = Some(vk::Extent2D {
                                     width: image.image_extent.width,
                                     height: image.image_extent.height,
                                 });
                             }
                         }
                     }
                 } else if output.layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
                      if let Some(physical) = self.resources.get(&output.handle.0) {
                         let image_guard = match &physical.resource {
                             ResourceType::SwapchainImage(image_arc) => Some(image_arc.lock().unwrap()),
                             ResourceType::Image(images) => Some(images[frame_info.frame_index].lock().unwrap()),
                             ResourceType::Buffer(_) => None,
                         };
                         if let Some(image) = image_guard {

                             let mut clear_value = vk::ClearValue::default();
                             clear_value.depth_stencil = CLEAR_DEPTH_STENCIL;

                             let mut depth_att = vk::RenderingAttachmentInfo::default()
                                 .image_view(image.image_view)
                                 .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                                 .load_op(vk::AttachmentLoadOp::CLEAR)
                                 .store_op(vk::AttachmentStoreOp::STORE)
                                 .clear_value(clear_value);
                             
                             // Handle depth resolve target with RESOLVE_MODE_MAX
                             if let Some(resolve_handle) = &output.resolve_target {
                                 if let Some(resolve_physical) = self.resources.get(&resolve_handle.0) {
                                     let resolve_image_guard = match &resolve_physical.resource {
                                         ResourceType::SwapchainImage(image_arc) => Some(image_arc.lock().unwrap()),
                                         ResourceType::Image(images) => Some(images[frame_info.frame_index].lock().unwrap()),
                                         ResourceType::Buffer(_) => None,
                                     };
                                     if let Some(resolve_image) = resolve_image_guard {
                                         depth_att = depth_att
                                             .resolve_image_view(resolve_image.image_view)
                                             .resolve_image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                                             .resolve_mode(vk::ResolveModeFlags::MAX);
                                     }
                                 }
                             }
                             
                             depth_attachment = Some(depth_att);
                             
                             is_dynamic_rendering = true;
                             
                             // Use the first attachment's extent as the render area (depth if no color)
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
                .semaphore(self.image_available_semaphores[self.current_frame])
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .value(1)
        ];
        
        // wait transfer operations
        if self.pending_transfers[self.current_frame] {
            wait_semaphores_vec.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.transfer_finished_semaphores[self.current_frame])
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .value(1)
            );
            self.pending_transfers[self.current_frame] = false;
        }
        
        let signal_semaphores = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.render_finished_semaphores[image_index as usize])
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
                device.device.queue_submit2(graphics_queue, &submit_info, self.in_flight_fences[self.current_frame])?;
             }
        }
        
        let wait_semaphores_present = [self.render_finished_semaphores[image_index as usize]];
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
         
         if image_count != self.render_finished_semaphores.len() {
             let device = self.device.lock().unwrap();
             unsafe {
                 device.device.device_wait_idle()?;
                 
                 for sem in &self.render_finished_semaphores {
                     device.device.destroy_semaphore(*sem, None);
                 }
                 
                 self.render_finished_semaphores.clear();
                 
                 let semaphore_info = vk::SemaphoreCreateInfo::default();
                 for _ in 0..image_count {
                     self.render_finished_semaphores.push(device.device.create_semaphore(&semaphore_info, None)?);
                 }
             }
         }
         
         Ok(())
    }

    pub fn submit_transfer_work<F>(&mut self, frame: usize, commands: F) -> Result<()>
    where
        F: FnOnce(vk::CommandBuffer, &BnanDevice),
    {
        let device = self.device.lock().unwrap();
        let cmd_buffer = self.transfer_command_buffers[frame];
        
        unsafe {
            
            // wait transfers
            device.device.wait_for_fences(&[self.transfer_fences[frame]], true, u64::MAX)?;
            device.device.reset_fences(&[self.transfer_fences[frame]])?;
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
                    .semaphore(self.transfer_finished_semaphores[frame])
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
            
            device.device.queue_submit2(device.transfer_queue, &submit_info, self.transfer_fences[frame])?;
        }
        
        self.pending_transfers[frame] = true;
        Ok(())
    }
    
    pub fn has_pending_transfers(&self, frame: usize) -> bool {
        self.pending_transfers[frame]
    }
    
    pub fn get_current_frame(&self) -> usize {
        self.current_frame
    }
    
    pub fn get_queue_indices(&self) -> Result<crate::core::bnan_device::QueueIndices> {
        self.device.lock().unwrap().get_queue_indices()
    }
}
