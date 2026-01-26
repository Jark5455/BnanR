#[allow(dead_code)]

use anyhow::*;
use ash::*;
use cgmath::*;

use BnanR::core::{ArcMut, make_arcmut, RcMut};
use BnanR::core::bnan_buffer::BnanBuffer;
use BnanR::core::bnan_camera::BnanCamera;
use BnanR::core::bnan_descriptors::*;
use BnanR::core::bnan_device::{BnanBarrierBuilder, BnanDevice, WorkQueue};
use BnanR::core::bnan_image::BnanImage;
use BnanR::core::bnan_mesh::{BnanMeshletDAG, BnanMeshletData, Vertex};
use BnanR::core::bnan_pipeline::{BnanPipeline, GraphicsPipelineConfigInfo};
use BnanR::core::bnan_rendering::{BnanFrameInfo, FRAMES_IN_FLIGHT};
use BnanR::core::bnan_window::WindowObserver;
use BnanR::core::bnan_render_graph::graph::BnanRenderGraph;
use BnanR::core::bnan_render_graph::resource::ResourceHandle;
use BnanR::fs::bpk::BpkArchive;
use BnanR::fs::streaming_buffer::BnanStreamingBuffer;
use crate::downsample_system::TemporalObserver;

const MESHLET_TASK_FILEPATH: &str = "./build/BnanR-Sample-Shaders/simple-raster-mesh.task.spv";
const MESHLET_MESH_FILEPATH: &str = "./build/BnanR-Sample-Shaders/simple-raster-mesh.mesh.spv";
const MESHLET_FRAG_FILEPATH: &str = "./build/BnanR-Sample-Shaders/simple-raster-mesh.frag.spv";

const GLOBAL_BUFFER_SIZE: u64 = 256 * 1024 * 1024;

#[repr(C)]
pub struct GlobalUBO {
    pub projection: Matrix4<f32>,
    pub view: Matrix4<f32>,
    pub inv_view: Matrix4<f32>,
    pub frustum_planes: [Vector4<f32>; 6],
}

#[repr(C)]
pub struct MeshletData {
    pub position_offset: u32,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub index_offset: u32,
    pub index_count: u32,
    pub bound_center: Vector3<f32>,
    pub bound_radius: f32,
}

pub struct MeshletSystem {
    pub device: ArcMut<BnanDevice>,
    pub render_graph: ArcMut<BnanRenderGraph>,
    
    pub streaming_buffer: BnanStreamingBuffer,
    
    pub global_position_buffer: BnanBuffer,
    pub global_vertex_buffer: BnanBuffer,
    pub global_index_buffer: BnanBuffer,
    
    pub meshlet_data_buffer: BnanBuffer,

    // offset trackers for global buffers
    position_offset: u64,
    vertex_offset: u64,
    index_offset: u64,
    
    pub loaded_meshlets: Vec<MeshletData>,
    pub meshlet_dag: Option<BnanMeshletDAG>,
    
    pub depth_handle: ResourceHandle,
    pub color_handle: ResourceHandle,
    pub resolved_depth_handle: ResourceHandle,

    pub depth_images: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT],
    pub color_images: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT],
    pub resolved_depth_images: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT],

    pub descriptor_pool: BnanDescriptorPool,
    pub descriptor_set_layout: BnanDescriptorSetLayout,
    pub descriptor_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],
    pub temporal_descriptor_set_layout: BnanDescriptorSetLayout,
    pub temporal_descriptor_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],
    pub temporal_sampler: vk::Sampler,
    pub ubo_buffers: [BnanBuffer; FRAMES_IN_FLIGHT],
    
    pub pipeline: BnanPipeline,
}

impl WindowObserver<(i32, i32)> for MeshletSystem {
    fn update(&mut self, data: (i32, i32)) {
        let (width, height) = data;

        unsafe {
            let device_guard = self.device.lock().unwrap();
            device_guard.device.queue_wait_idle(device_guard.graphics_queue).unwrap();
        }

        let extent = vk::Extent2D::default()
            .width(width as u32)
            .height(height as u32);

        (self.depth_images, self.color_images, self.resolved_depth_images) = Self::create_framebuffer_images(self.device.clone(), extent).unwrap();

        {
            let mut render_graph_guard = self.render_graph.lock().unwrap();
            render_graph_guard.update_render_image(&self.depth_handle, self.depth_images.clone());
            render_graph_guard.update_render_image(&self.color_handle, self.color_images.clone());
            render_graph_guard.update_render_image(&self.resolved_depth_handle, self.resolved_depth_images.clone());
        }
    }
}

impl TemporalObserver<[ArcMut<BnanImage>; FRAMES_IN_FLIGHT]> for MeshletSystem {
    fn update(&mut self, data: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT]) {

        let image_info = (0..FRAMES_IN_FLIGHT).map(|frame| {
            [vk::DescriptorImageInfo::default()
                .image_view(data[frame].lock().unwrap().image_view)
                .sampler(self.temporal_sampler)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
        }).collect::<Vec<_>>();

        let writes = (0..FRAMES_IN_FLIGHT).map(|frame| {
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .dst_binding(0)
                .image_info(&image_info[frame])
                .dst_set(self.temporal_descriptor_sets[frame])
        }).collect::<Vec<_>>();

        unsafe {
            self.device.lock().unwrap().device.update_descriptor_sets(&writes, &[]);
        }
    }
}

impl Drop for MeshletSystem {
    fn drop(&mut self) {
        unsafe {
            self.device.lock().unwrap().device.destroy_sampler(self.temporal_sampler, None);
        }
    }
}

impl MeshletSystem {
    const STREAMING_BUFFER_SIZE: u64 = 16 * 1024 * 1024;
    const STREAMING_SLOT_SIZE: u64 = ((64 * size_of::<Vector3<f32>>()) + (64 * size_of::<Vertex>()) + (124 * 3 * size_of::<u8>())) as u64;

    pub fn new(device: ArcMut<BnanDevice>, render_graph: ArcMut<BnanRenderGraph>) -> Result<MeshletSystem> {
        let swapchain_extent: vk::Extent2D;
        {
            let rendergraph_guard = render_graph.lock().unwrap();
            let swapchain_guard = rendergraph_guard.swapchain.lock().unwrap();
            swapchain_extent = swapchain_guard.extent;
        }

        let streaming_buffer = BnanStreamingBuffer::new(
            device.clone(),
            Self::STREAMING_BUFFER_SIZE,
            Self::STREAMING_SLOT_SIZE,
        )?;

        let global_position_buffer = Self::create_global_buffer(device.clone(), GLOBAL_BUFFER_SIZE)?;
        let global_vertex_buffer = Self::create_global_buffer(device.clone(), GLOBAL_BUFFER_SIZE)?;
        let global_index_buffer = Self::create_global_buffer(device.clone(), GLOBAL_BUFFER_SIZE)?;

        let meshlet_data_buffer = Self::create_meshlet_data_buffer(device.clone())?;

        let ubo_buffers = Self::create_uniform_buffers(device.clone())?;
        let (depth_images, color_images, resolved_depth_images) = Self::create_framebuffer_images(device.clone(), swapchain_extent)?;

        let (depth_handle, color_handle, resolved_depth_handle) = Self::create_framebuffer_image_handles(render_graph.clone(), depth_images.clone(), color_images.clone(), resolved_depth_images.clone());

        let descriptor_set_layout = Self::create_descriptor_set_layout(device.clone())?;
        let descriptor_pool = Self::create_descriptor_pool(device.clone())?;
        let descriptor_sets = Self::allocate_descriptor_sets(
            device.clone(), 
            &descriptor_set_layout, 
            &descriptor_pool, 
            &ubo_buffers,
            &global_position_buffer,
            &global_vertex_buffer,
            &global_index_buffer,
            &meshlet_data_buffer,
        )?;

        let temporal_descriptor_set_layout = Self::create_temporal_descriptor_set_layout(device.clone())?;
        let temporal_descriptor_sets = Self::allocate_temporal_descriptor_sets(device.clone(), &temporal_descriptor_set_layout, &descriptor_pool)?;
        let temporal_sampler = Self::create_temporal_sampler(device.clone())?;

        let pipeline = Self::create_pipeline(
            device.clone(),
            &descriptor_set_layout,
            &temporal_descriptor_set_layout,
            color_images[0].lock().unwrap().format, 
            depth_images[0].lock().unwrap().format
        )?;

        Ok(MeshletSystem {
            device,
            render_graph,
            streaming_buffer,
            global_position_buffer,
            global_vertex_buffer,
            global_index_buffer,
            meshlet_data_buffer,
            position_offset: 0,
            vertex_offset: 0,
            index_offset: 0,
            loaded_meshlets: Vec::new(),
            meshlet_dag: None,
            depth_handle,
            color_handle,
            resolved_depth_handle,
            depth_images,
            color_images,
            resolved_depth_images,
            descriptor_pool,
            descriptor_set_layout,
            descriptor_sets,
            temporal_descriptor_set_layout,
            temporal_descriptor_sets,
            temporal_sampler,
            ubo_buffers,
            pipeline,
        })
    }

    pub fn load_meshlet_mesh(&mut self, archive_path: &str, mesh_name: &str) -> Result<()> {
        let mut archive = BpkArchive::open(archive_path)?;
        
        let dag = archive.load_meshlet_dag(mesh_name)?;
        
        let mut position_copies: Vec<vk::BufferCopy> = Vec::new();
        let mut vertex_copies: Vec<vk::BufferCopy> = Vec::new();
        let mut index_copies: Vec<vk::BufferCopy> = Vec::new();
        
        const POSITION_SIZE: u64 = size_of::<Vector3<f32>>() as u64; // Vector3<f32> = 3 * 4 bytes
        const VERTEX_SIZE: u64 = size_of::<Vertex>() as u64;   // Vertex struct = normal(12) + tangent(12) + uv(8)
        const TRIANGLE_SIZE: u64 = 3;  // 3 x u8 indices per triangle
        
        for (i, &node_idx) in dag.leaf_indices.iter().enumerate() {
            let node = &dag.nodes[node_idx as usize];
            let meshlet_path = format!("{}/meshlet_{}", mesh_name, node_idx);
            
            let (positions, vertices, triangles) = archive.load_meshlet(&meshlet_path)?;
            
            let pos_size = positions.len() as u64;
            let vert_size = vertices.len() as u64;
            let idx_size = triangles.len() as u64;
            let total_size = pos_size + vert_size + idx_size;

            if self.position_offset + pos_size > GLOBAL_BUFFER_SIZE {
                bail!("Position buffer overflow");
            }
            if self.vertex_offset + vert_size > GLOBAL_BUFFER_SIZE {
                bail!("Vertex buffer overflow");
            }
            if self.index_offset + idx_size > GLOBAL_BUFFER_SIZE {
                bail!("Index buffer overflow");
            }
            
            let alloc = self.streaming_buffer.allocate()
                .ok_or_else(|| anyhow!("Streaming buffer full"))?;

            if total_size > alloc.size {
                let slot_size = alloc.size;
                self.streaming_buffer.free(alloc);
                bail!("Meshlet {} data ({} bytes) exceeds slot size ({} bytes)", 
                      node_idx, total_size, slot_size);
            }
            
            self.streaming_buffer.write_data(&alloc, 0, &positions)?;
            self.streaming_buffer.write_data(&alloc, pos_size, &vertices)?;
            self.streaming_buffer.write_data(&alloc, pos_size + vert_size, &triangles)?;
            
            position_copies.push(vk::BufferCopy {
                src_offset: alloc.offset,
                dst_offset: self.position_offset,
                size: pos_size,
            });
            
            vertex_copies.push(vk::BufferCopy {
                src_offset: alloc.offset + pos_size,
                dst_offset: self.vertex_offset,
                size: vert_size,
            });
            
            index_copies.push(vk::BufferCopy {
                src_offset: alloc.offset + pos_size + vert_size,
                dst_offset: self.index_offset,
                size: idx_size,
            });
            
            let vertex_count = (pos_size / POSITION_SIZE) as u32;
            let index_count = (idx_size / TRIANGLE_SIZE) as u32;

            let bound_center = Vector3::new(node.bounds.x, node.bounds.y, node.bounds.z);
            let bound_radius = node.bounds.w;

            self.loaded_meshlets.push(MeshletData {
                position_offset: self.position_offset as u32,
                vertex_offset: self.vertex_offset as u32,
                index_offset: self.index_offset as u32,
                vertex_count,
                index_count,
                bound_center,
                bound_radius,
            });
            
            self.position_offset += pos_size;
            self.vertex_offset += vert_size;
            self.index_offset += idx_size;
        }
        
        let staging_buffer = self.streaming_buffer.buffer();
        let position_buffer = self.global_position_buffer.buffer;
        let vertex_buffer = self.global_vertex_buffer.buffer;
        let index_buffer = self.global_index_buffer.buffer;
        
        self.render_graph.lock().unwrap().submit_transfer_work(0, |cmd, device| {
            let staging_guard = staging_buffer.lock().unwrap();

            unsafe {
                device.device.cmd_copy_buffer(cmd, staging_guard.buffer, position_buffer, &position_copies);
                device.device.cmd_copy_buffer(cmd, staging_guard.buffer, vertex_buffer, &vertex_copies);
                device.device.cmd_copy_buffer(cmd, staging_guard.buffer, index_buffer, &index_copies);
            }
        })?;
        
        self.meshlet_dag = Some(dag);
        Ok(())
    }

    pub fn update_uniform_buffers(&mut self, frame_info: &BnanFrameInfo, camera: RcMut<BnanCamera>) -> Result<()> {
        let ubo = GlobalUBO {
            projection: camera.borrow().projection_matrix,
            view: camera.borrow().view_matrix,
            inv_view: camera.borrow().inverse_view_matrix,
            frustum_planes: camera.borrow().get_frustum_planes()?
        };

        let data = unsafe { 
            std::slice::from_raw_parts::<u8>(&ubo as *const GlobalUBO as *const u8, size_of::<GlobalUBO>())
        };

        self.ubo_buffers[frame_info.frame_index].write_to_buffer(data, 0)?;
        self.ubo_buffers[frame_info.frame_index].flush(size_of::<GlobalUBO>() as vk::DeviceSize, 0)?;

        Ok(())
    }

    pub fn update_meshlet_data(&mut self) -> Result<()> {
        if self.loaded_meshlets.is_empty() {
            return Ok(());
        }
        
        if self.loaded_meshlets.len() > Self::MAX_MESHLETS {
            bail!("Too many meshlets: {} exceeds max {}", self.loaded_meshlets.len(), Self::MAX_MESHLETS);
        }
        
        let data = unsafe {
            std::slice::from_raw_parts::<u8>(
                self.loaded_meshlets.as_ptr() as *const u8,
                self.loaded_meshlets.len() * size_of::<MeshletData>()
            )
        };
        
        self.meshlet_data_buffer.write_to_buffer(data, 0)?;
        self.meshlet_data_buffer.flush(vk::WHOLE_SIZE, 0)?;
        
        Ok(())
    }

    pub fn draw(&self, frame_info: &BnanFrameInfo) {
        if self.loaded_meshlets.is_empty() {
            return;
        }
        
        unsafe {
            let device_guard = self.device.lock().unwrap();
            let swapchain_image_extent = frame_info.swapchain_image.lock().unwrap().image_extent;

            let image_extent = vk::Extent2D::default()
                .width(swapchain_image_extent.width)
                .height(swapchain_image_extent.height);

            let viewport = [vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(image_extent.width as f32)
                .height(image_extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0)
            ];

            device_guard.device.cmd_set_viewport(frame_info.main_command_buffer, 0, &viewport);

            let scissor = [vk::Rect2D::default()
                .offset(vk::Offset2D { x: 0, y: 0 })
                .extent(image_extent)
            ];

            device_guard.device.cmd_set_scissor(frame_info.main_command_buffer, 0, &scissor);

            device_guard.device.cmd_bind_pipeline(
                frame_info.main_command_buffer, 
                vk::PipelineBindPoint::GRAPHICS, 
                self.pipeline.pipeline
            );
            
            device_guard.device.cmd_bind_descriptor_sets(
                frame_info.main_command_buffer, 
                vk::PipelineBindPoint::GRAPHICS, 
                self.pipeline.pipeline_layout, 
                0, 
                &[self.descriptor_sets[frame_info.frame_index], self.temporal_descriptor_sets[frame_info.frame_index]],
                &[]
            );

            let meshlet_count = self.loaded_meshlets.len() as u32;
            let meshlet_count_bytes = meshlet_count.to_le_bytes();

            device_guard.device.cmd_push_constants(frame_info.main_command_buffer, self.pipeline.pipeline_layout, vk::ShaderStageFlags::TASK_EXT, 0, &meshlet_count_bytes);

            let group_count_x = (meshlet_count + 31) / 32;

            {
                let mesh_shader = ext::mesh_shader::Device::new(&device_guard.instance, &device_guard.device);
                mesh_shader.cmd_draw_mesh_tasks(frame_info.main_command_buffer, group_count_x, 1, 1);
            }
        }
    }
    
    fn create_global_buffer(device: ArcMut<BnanDevice>, size: u64) -> Result<BnanBuffer> {
        BnanBuffer::new(
            device,
            size,
            1,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
    }

    fn create_uniform_buffers(device: ArcMut<BnanDevice>) -> Result<[BnanBuffer; FRAMES_IN_FLIGHT]> {
        let vec: Result<Vec<_>> = (0..FRAMES_IN_FLIGHT).map(|_|
            BnanBuffer::new(
                device.clone(), 
                size_of::<GlobalUBO>() as vk::DeviceSize,
                1, 
                vk::BufferUsageFlags::UNIFORM_BUFFER, 
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            )
        ).collect();

        let mut buffers: [BnanBuffer; FRAMES_IN_FLIGHT] = vec?.try_into()
            .map_err(|_| anyhow!("failed to create uniform buffers"))?;

        for buffer in &mut buffers {
            buffer.map()?;
        }

        Ok(buffers)
    }
    
    const MAX_MESHLETS: usize = 50000;
    
    fn create_meshlet_data_buffer(device: ArcMut<BnanDevice>) -> Result<BnanBuffer> {
        let buffer_size = (size_of::<BnanMeshletData>() * Self::MAX_MESHLETS) as u64;
        
        let mut buffer = BnanBuffer::new(
            device,
            buffer_size,
            1,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        
        buffer.map()?;
        
        Ok(buffer)
    }

    fn create_framebuffer_images(device: ArcMut<BnanDevice>, extent: vk::Extent2D) -> Result<([ArcMut<BnanImage>; FRAMES_IN_FLIGHT], [ArcMut<BnanImage>; FRAMES_IN_FLIGHT], [ArcMut<BnanImage>; FRAMES_IN_FLIGHT])> {
        let depth_format = device.lock().unwrap().find_depth_format()?;
        let color_format = vk::Format::B8G8R8A8_SRGB;

        let sample_count = device.lock().unwrap().msaa_samples;

        let extent = vk::Extent3D::default()
            .width(extent.width)
            .height(extent.height)
            .depth(1);

        // MSAA depth images
        let depth_vec: Result<Vec<_>> = (0..FRAMES_IN_FLIGHT).map(|_|
            Ok(make_arcmut(BnanImage::new(device.clone(), depth_format, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::MemoryPropertyFlags::DEVICE_LOCAL, extent, sample_count, None)?))
        ).collect();

        let depth_images: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT] = depth_vec?.try_into().map_err(|_| anyhow!("something went wrong while creating depth images"))?;

        // MSAA color images
        let color_vec: Result<Vec<_>> = (0..FRAMES_IN_FLIGHT).map(|_|
            Ok(make_arcmut(BnanImage::new(device.clone(), color_format, vk::ImageUsageFlags::COLOR_ATTACHMENT, vk::MemoryPropertyFlags::DEVICE_LOCAL, extent, sample_count, None)?))
        ).collect();

        let color_images: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT] = color_vec?.try_into().map_err(|_| anyhow!("something went wrong while creating color images"))?;
        
        // Single-sample resolved depth images (for sampling in compute/task shaders)
        let resolved_depth_vec: Result<Vec<_>> = (0..FRAMES_IN_FLIGHT).map(|_|
            Ok(make_arcmut(BnanImage::new(device.clone(), depth_format, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED, vk::MemoryPropertyFlags::DEVICE_LOCAL, extent, vk::SampleCountFlags::TYPE_1, None)?))
        ).collect();

        let resolved_depth_images: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT] = resolved_depth_vec?.try_into().map_err(|_| anyhow!("something went wrong while creating resolved depth images"))?;

        unsafe {
            let device_guard = device.lock().unwrap();
            let command_buffer = device_guard.begin_commands(WorkQueue::GRAPHICS, 1)?[0];

            let mut builder = BnanBarrierBuilder::new();

            for image in &depth_images {
                builder.transition_image_layout(image.lock().unwrap().image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL, None, None)?;
            }

            for image in &color_images {
                builder.transition_image_layout(image.lock().unwrap().image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, None, None)?;
            }
            
            // Resolved depth images: transition to SHADER_READ_ONLY_OPTIMAL with explicit DEPTH aspect
            for image in &resolved_depth_images {
                let subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);
                
                let barrier = vk::ImageMemoryBarrier2::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(image.lock().unwrap().image)
                    .subresource_range(subresource_range)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
                
                builder.push_barrier(barrier);
            }

            builder.record(&*device_guard, command_buffer);
            device_guard.submit_commands(WorkQueue::GRAPHICS, vec![command_buffer], None, None)?;
        }

        Ok((depth_images, color_images, resolved_depth_images))
    }

    fn create_framebuffer_image_handles(render_graph: ArcMut<BnanRenderGraph>, depth_images: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT], color_images: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT], resolved_depth_images: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT]) -> (ResourceHandle, ResourceHandle, ResourceHandle) {
        let mut render_graph_guard = render_graph.lock().unwrap();

        let depth_handle = render_graph_guard.import_render_image("MeshletDepthBuffer", depth_images.clone());
        let color_handle = render_graph_guard.import_render_image("MeshletColor", color_images.clone());
        let resolved_depth_handle = render_graph_guard.import_render_image("ResolvedDepth", resolved_depth_images.clone());

        {
            let resource =render_graph_guard.resources.get_mut(&depth_handle.0).unwrap();

            resource.current_layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            resource.current_stage = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS;
        }

        {
            let resource =render_graph_guard.resources.get_mut(&color_handle.0).unwrap();

            resource.current_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
            resource.current_stage = vk::PipelineStageFlags2::FRAGMENT_SHADER;
        }

        {
            let resource =render_graph_guard.resources.get_mut(&resolved_depth_handle.0).unwrap();

            resource.current_layout = vk::ImageLayout::GENERAL;
            resource.current_stage = vk::PipelineStageFlags2::ALL_COMMANDS;
        }

        (depth_handle, color_handle, resolved_depth_handle)
    }

    fn create_descriptor_set_layout(device: ArcMut<BnanDevice>) -> Result<BnanDescriptorSetLayout> {
        BnanDescriptorSetLayoutBuilder::new(vk::DescriptorSetLayoutCreateFlags::empty())
            .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::FRAGMENT)
            .add_binding(1, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::MESH_EXT)
            .add_binding(2, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::MESH_EXT)
            .add_binding(3, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::MESH_EXT)
            .add_binding(4, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::MESH_EXT)
            .build(device)
    }

    fn create_temporal_descriptor_set_layout(device: ArcMut<BnanDevice>) -> Result<BnanDescriptorSetLayout> {
        BnanDescriptorSetLayoutBuilder::new(vk::DescriptorSetLayoutCreateFlags::empty())
            .add_binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::TASK_EXT)
            .build(device)
    }
    
    fn create_descriptor_pool(device: ArcMut<BnanDevice>) -> Result<BnanDescriptorPool> {
        BnanDescriptorPoolBuilder::new(FRAMES_IN_FLIGHT as u32 * 2, vk::DescriptorPoolCreateFlags::empty())
            .add_pool_size(vk::DescriptorType::UNIFORM_BUFFER, FRAMES_IN_FLIGHT as u32)
            .add_pool_size(vk::DescriptorType::STORAGE_BUFFER, FRAMES_IN_FLIGHT as u32 * 4)// 4 SSBOs per frame
            .add_pool_size(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, FRAMES_IN_FLIGHT as u32)
            .build(device)
    }

    fn allocate_descriptor_sets(
        device: ArcMut<BnanDevice>, 
        layout: &BnanDescriptorSetLayout, 
        pool: &BnanDescriptorPool, 
        uniform_buffers: &[BnanBuffer; FRAMES_IN_FLIGHT],
        position_buffer: &BnanBuffer,
        vertex_buffer: &BnanBuffer,
        index_buffer: &BnanBuffer,
        meshlet_data_buffer: &BnanBuffer,
    ) -> Result<[vk::DescriptorSet; FRAMES_IN_FLIGHT]> {
        let sets: Result<Vec<_>> = (0..FRAMES_IN_FLIGHT).map(|frame| {
            let ubo_info = vec![
                vk::DescriptorBufferInfo::default()
                    .buffer(uniform_buffers[frame].buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            ];
            
            let position_info = vec![
                vk::DescriptorBufferInfo::default()
                    .buffer(position_buffer.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            ];
            
            let vertex_info = vec![
                vk::DescriptorBufferInfo::default()
                    .buffer(vertex_buffer.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            ];
            
            let index_info = vec![
                vk::DescriptorBufferInfo::default()
                    .buffer(index_buffer.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            ];
            
            let meshlet_data_info = vec![
                vk::DescriptorBufferInfo::default()
                    .buffer(meshlet_data_buffer.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            ];
            
            BnanDescriptorWriter::new(layout)
                .write_uniform_buffer(0, &ubo_info)
                .write_storage_buffer(1, &position_info)
                .write_storage_buffer(2, &vertex_info)
                .write_storage_buffer(3, &index_info)
                .write_storage_buffer(4, &meshlet_data_info)
                .write(device.clone(), pool)
        }).collect();
        
        let sets_vec = sets?;
        let sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT] = sets_vec.try_into()
            .map_err(|_| anyhow!("failed to create descriptor sets"))?;
        
        Ok(sets)
    }

    fn allocate_temporal_descriptor_sets(
        device: ArcMut<BnanDevice>,
        layout: &BnanDescriptorSetLayout,
        pool: &BnanDescriptorPool,
    ) -> Result<[vk::DescriptorSet; FRAMES_IN_FLIGHT]> {
        let sets: Result<Vec<_>> = (0..FRAMES_IN_FLIGHT).map(|_frame| {BnanDescriptorWriter::new(layout).write(device.clone(), pool)}).collect();

        let sets_vec = sets?;
        let sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT] = sets_vec.try_into()
            .map_err(|_| anyhow!("failed to create descriptor sets"))?;

        Ok(sets)
    }

    fn create_temporal_sampler(device: ArcMut<BnanDevice>) -> Result<vk::Sampler> {
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_lod(0.0)
            .max_lod(0.0);

        unsafe { Ok(device.lock().unwrap().device.create_sampler(&sampler_info, None)?) }
    }

    fn create_pipeline(
        device: ArcMut<BnanDevice>, 
        layout: &BnanDescriptorSetLayout,
        temporal_layout: &BnanDescriptorSetLayout,
        color_format: vk::Format, 
        depth_format: vk::Format
    ) -> Result<BnanPipeline> {

        let push_constant_range = [vk::PushConstantRange::default()
            .size(size_of::<u32>() as u32)
            .stage_flags(vk::ShaderStageFlags::TASK_EXT)];

        let layouts = [layout.layout, temporal_layout.layout];

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&push_constant_range);
            
        let pipeline_layout = unsafe { 
            device.lock().unwrap().device.create_pipeline_layout(&pipeline_layout_info, None)? 
        };
        
        let mut pipeline_config_info = GraphicsPipelineConfigInfo::default();

        pipeline_config_info.attribute_descriptions.clear();
        pipeline_config_info.binding_descriptions.clear();

        let color_formats = [color_format];

        pipeline_config_info.pipeline_rendering_info = pipeline_config_info.pipeline_rendering_info
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(depth_format);

        pipeline_config_info.multisample_info.rasterization_samples = device.lock().unwrap().msaa_samples;
        pipeline_config_info.pipeline_layout = pipeline_layout;

        Ok(BnanPipeline::new_graphics_pipeline(
            device, 
            MESHLET_MESH_FILEPATH.to_owned(), 
            MESHLET_FRAG_FILEPATH.to_owned(),
            Some(MESHLET_TASK_FILEPATH.to_owned()),
            &mut pipeline_config_info
        )?)
    }
}

