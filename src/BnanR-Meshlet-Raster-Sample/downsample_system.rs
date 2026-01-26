#[allow(dead_code)]

use std::mem::size_of;

use anyhow::*;
use ash::*;
use cgmath::Vector2;
use BnanR::core::{ArcMut, make_arcmut, RcMut};
use BnanR::core::bnan_device::{BnanBarrierBuilder, BnanDevice, WorkQueue};
use BnanR::core::bnan_image::BnanImage;
use BnanR::core::bnan_pipeline::BnanPipeline;
use BnanR::core::bnan_rendering::{BnanFrameInfo, FRAMES_IN_FLIGHT};
use BnanR::core::bnan_window::WindowObserver;
use BnanR::core::bnan_render_graph::graph::BnanRenderGraph;
use BnanR::core::bnan_render_graph::resource::ResourceHandle;

const DOWNSAMPLE_SHADER_FILEPATH: &str = "./build/BnanR-Sample-Shaders/downsample.comp.spv";

pub trait TemporalObserver<T> {
    fn update(&mut self, data: T);
}

pub struct DownsampleSystem {
    device: ArcMut<BnanDevice>,
    render_graph: ArcMut<BnanRenderGraph>,
    
    // Input resolved depth from main pass
    input_handle: ResourceHandle,
    
    // Hierarchical depth mip chain (one per frame in flight)
    // Each image has multiple mip levels for the depth pyramid
    pub hi_z_images: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT],
    pub hi_z_handle: ResourceHandle,
    pub hi_z_temporal_observers: Vec<RcMut<dyn TemporalObserver<[ArcMut<BnanImage>; FRAMES_IN_FLIGHT]>>>,
    
    // Sampler for reading the previous mip level
    sampler: vk::Sampler,
    
    // Push descriptor set layout and pipeline
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline: BnanPipeline,
    
    // Number of mip levels in the hi-z pyramid
    mip_levels: u32,
}

impl WindowObserver<(i32, i32)> for DownsampleSystem {
    fn update(&mut self, data: (i32, i32)) {
        let (width, height) = data;
        
        unsafe {
            let device_guard = self.device.lock().unwrap();
            device_guard.device.queue_wait_idle(device_guard.graphics_queue).unwrap();
        }
        
        let extent = vk::Extent2D::default()
            .width(width as u32)
            .height(height as u32);

        self.hi_z_images = Self::create_hi_z_images(self.device.clone(), extent).unwrap();
        self.mip_levels = self.hi_z_images[0].lock().unwrap().mip_levels;
        
        // Create mip views for all levels on each image
        for image in &self.hi_z_images {
            let mut img_guard = image.lock().unwrap();
            for mip in 0..self.mip_levels {
                img_guard.create_mip_view(mip).unwrap();
            }
        }
        
        self.render_graph.lock().unwrap().update_render_image(&self.hi_z_handle, self.hi_z_images.clone());

        for observer in &mut self.hi_z_temporal_observers {
            let mut shifted_images = self.hi_z_images.clone();
            shifted_images.rotate_left(1);

            observer.borrow_mut().update(shifted_images);
        }
    }
}

impl Drop for DownsampleSystem {
    fn drop(&mut self) {
        unsafe {
            let device = self.device.lock().unwrap();
            device.device.device_wait_idle().unwrap();
            device.device.destroy_sampler(self.sampler, None);
            device.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

impl DownsampleSystem {
    pub fn new(
        device: ArcMut<BnanDevice>,
        render_graph: ArcMut<BnanRenderGraph>,
        input_handle: ResourceHandle,
    ) -> Result<DownsampleSystem> {
        let swapchain_extent: vk::Extent2D;
        {
            let rendergraph_guard = render_graph.lock().unwrap();
            let swapchain_guard = rendergraph_guard.swapchain.lock().unwrap();
            swapchain_extent = swapchain_guard.extent;
        }
        
        // Create the hi-z mip chain images
        let hi_z_images = Self::create_hi_z_images(device.clone(), swapchain_extent)?;
        let hi_z_handle = Self::create_hi_z_image_handle(render_graph.clone(), hi_z_images.clone());
        let mip_levels = hi_z_images[0].lock().unwrap().mip_levels;
        
        // Create mip views for each level
        for image in &hi_z_images {
            let mut img_guard = image.lock().unwrap();
            for mip in 0..mip_levels {
                img_guard.create_mip_view(mip)?;
            }
        }
        
        // Create sampler for reading previous mip levels
        let sampler = Self::create_sampler(device.clone())?;
        
        // Create push descriptor set layout
        let descriptor_set_layout = Self::create_push_descriptor_set_layout(device.clone())?;
        
        // Create compute pipeline
        let pipeline = Self::create_pipeline(device.clone(), descriptor_set_layout)?;
        
        Ok(DownsampleSystem {
            device,
            render_graph,
            input_handle,
            hi_z_images,
            hi_z_handle,
            hi_z_temporal_observers: Vec::new(),
            sampler,
            descriptor_set_layout,
            pipeline,
            mip_levels,
        })
    }
    
    pub fn register_observer(&mut self, observer: RcMut<dyn TemporalObserver<[ArcMut<BnanImage>; FRAMES_IN_FLIGHT]>>) {
        observer.borrow_mut().update(self.hi_z_images.clone());
        self.hi_z_temporal_observers.push(observer);
    }

    pub fn dispatch(&self, render_graph: &BnanRenderGraph, frame_info: &BnanFrameInfo) {
        let device_guard = self.device.lock().unwrap();
        let push_descriptor = khr::push_descriptor::Device::new(&device_guard.instance, &device_guard.device);
        
        // Get the input resolved depth image for this frame
        let input_images = render_graph
            .get_images(&self.input_handle)
            .expect("Input depth images not found");
        
        let input_image = input_images[frame_info.frame_index].lock().unwrap();
        let hi_z_image = self.hi_z_images[frame_info.frame_index].lock().unwrap();
        
        unsafe {
            device_guard.device.cmd_bind_pipeline(frame_info.main_command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline.pipeline);
        }
        
        // For each mip level, downsample from the previous level
        for mip in 0..self.mip_levels {
            let output_extent = hi_z_image.mip_extent(mip);

            // Determine input: mip 0 reads from resolved depth, others read from previous mip
            let input_view = if mip == 0 {
                input_image.image_view
            } else {
                hi_z_image.mip_views[(mip - 1) as usize]
            };
            
            let output_view = hi_z_image.mip_views[mip as usize];

            // Set up push descriptors
            let input_image_info = vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(input_view)
                .image_layout(vk::ImageLayout::GENERAL);
            
            let output_image_info = vk::DescriptorImageInfo::default()
                .image_view(output_view)
                .image_layout(vk::ImageLayout::GENERAL);
            
            let descriptor_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&input_image_info)),
                vk::WriteDescriptorSet::default()
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&output_image_info)),
            ];

            let push_constants = Vector2::new(1.0 / output_extent.width as f32, 1.0 / output_extent.height as f32);
            
            let push_bytes = unsafe {
                std::slice::from_raw_parts(
                    &push_constants as *const Vector2<f32> as *const u8,
                    size_of::<Vector2<f32>>(),
                )
            };
            
            unsafe {
                // Push descriptors
                push_descriptor.cmd_push_descriptor_set(
                    frame_info.main_command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline.pipeline_layout,
                    0,
                    &descriptor_writes,
                );
                
                // Push constants
                device_guard.device.cmd_push_constants(
                    frame_info.main_command_buffer,
                    self.pipeline.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_bytes,
                );
                
                // Dispatch
                let group_count_x = (output_extent.width + 7) / 8;
                let group_count_y = (output_extent.height + 7) / 8;
                device_guard.device.cmd_dispatch(frame_info.main_command_buffer, group_count_x, group_count_y, 1);
            }
            
            // Image memory barrier for the written mip level to ensure writes complete before next read
            if mip < self.mip_levels {
                let output_subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(mip)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);
                
                let output_barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(hi_z_image.image)
                    .subresource_range(output_subresource_range)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
                
                let dependency_info = vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&output_barrier));
                
                unsafe {
                    device_guard.device.cmd_pipeline_barrier2(frame_info.main_command_buffer, &dependency_info);
                }
            }
        }
    }
    
    fn create_hi_z_images(device: ArcMut<BnanDevice>, extent: vk::Extent2D) -> Result<([ArcMut<BnanImage>; FRAMES_IN_FLIGHT])> {
        let extent_3d = vk::Extent3D::default()
            .width(extent.width)
            .height(extent.height)
            .depth(1);
        
        let mip_levels = BnanImage::calculate_mip_levels(extent_3d);
        
        let images: Result<Vec<_>> = (0..FRAMES_IN_FLIGHT).map(|_| {
            Ok(make_arcmut(BnanImage::new(
                device.clone(),
                vk::Format::R32_SFLOAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                extent_3d,
                vk::SampleCountFlags::TYPE_1,
                Some(mip_levels),
            )?))
        }).collect();
        
        let mut images_array: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT] = images?
            .try_into()
            .map_err(|_| anyhow!("Failed to create hi-z images"))?;
        
        // Transition to TRANSFER_DST, clear with 1.0, then transition to GENERAL
        unsafe {
            let device_guard = device.lock().unwrap();
            let fence = device_guard.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let command_buffer = device_guard.begin_commands(WorkQueue::GRAPHICS, 1)?[0];
            
            // First transition to TRANSFER_DST_OPTIMAL for clearing
            let mut pre_clear_builder = BnanBarrierBuilder::new();
            
            for image in &images_array {
                let img = image.lock().unwrap();
                let subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1);
                
                let barrier = vk::ImageMemoryBarrier2::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(img.image)
                    .subresource_range(subresource_range)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
                
                pre_clear_builder.push_barrier(barrier);
            }
            
            pre_clear_builder.record(&*device_guard, command_buffer);
            
            // Clear all images to 1.0 (far plane) so nothing gets culled on first frame
            let clear_color = vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 0.0] };
            let clear_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(mip_levels)
                .base_array_layer(0)
                .layer_count(1);
            
            for image in &images_array {
                let img = image.lock().unwrap();
                device_guard.device.cmd_clear_color_image(
                    command_buffer,
                    img.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &clear_color,
                    &[clear_range],
                );
            }
            
            let mut post_clear_builder = BnanBarrierBuilder::new();

            for image in &images_array {
                let img = image.lock().unwrap();
                let subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1);
                
                let barrier = vk::ImageMemoryBarrier2::default()
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(img.image)
                    .subresource_range(subresource_range)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
                
                post_clear_builder.push_barrier(barrier);
            }
            
            post_clear_builder.record(&*device_guard, command_buffer);
            
            device_guard.submit_commands(WorkQueue::GRAPHICS, vec![command_buffer], None, Some(fence))?;
            device_guard.device.wait_for_fences(&[fence], true, u64::MAX)?;
            device_guard.device.destroy_fence(fence, None);
        }

        Ok(images_array)
    }

    fn create_hi_z_image_handle(render_graph: ArcMut<BnanRenderGraph>, images: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT]) -> ResourceHandle {
        let mut render_graph_guard = render_graph.lock().unwrap();
        let image_handle = render_graph_guard.import_render_image("HiZPyramid", images.clone());
        let resource =render_graph_guard.resources.get_mut(&image_handle.0).unwrap();

        resource.current_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        resource.current_stage = vk::PipelineStageFlags2::COMPUTE_SHADER;

        image_handle
    }
    
    fn create_sampler(device: ArcMut<BnanDevice>) -> Result<vk::Sampler> {
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_lod(0.0)
            .max_lod(0.0);
        
        unsafe { Ok(device.lock().unwrap().device.create_sampler(&sampler_info, None)?) }
    }
    
    fn create_push_descriptor_set_layout(device: ArcMut<BnanDevice>) -> Result<vk::DescriptorSetLayout> {
        let bindings = [
            // binding 0: input depth (sampled image)
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // binding 1: output depth (storage image)
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(&bindings);
        
        unsafe { Ok(device.lock().unwrap().device.create_descriptor_set_layout(&layout_info, None)?) }
    }
    
    fn create_pipeline(device: ArcMut<BnanDevice>, descriptor_set_layout: vk::DescriptorSetLayout) -> Result<BnanPipeline> {
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(size_of::<Vector2<f32>>() as u32);
        
        BnanPipeline::new_compute_pipeline(
            device,
            DOWNSAMPLE_SHADER_FILEPATH.to_string(),
            vec![descriptor_set_layout],
            Some(vec![push_constant_range]),
        )
    }
}
