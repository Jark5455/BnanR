#[allow(dead_code)]

use anyhow::*;
use ash::*;
use cgmath::*;

use BnanR::core::{ArcMut, make_arcmut};
use BnanR::core::bnan_buffer::BnanBuffer;
use BnanR::core::bnan_camera::BnanCamera;
use BnanR::core::bnan_descriptors::*;
use BnanR::core::bnan_device::BnanDevice;
use BnanR::core::bnan_image::BnanImage;
use BnanR::core::bnan_pipeline::{BnanPipeline, GraphicsPipelineConfigInfo};
use BnanR::core::bnan_rendering::{BnanFrameInfo, FRAMES_IN_FLIGHT};
use BnanR::core::bnan_window::WindowObserver;
use BnanR::core::bnan_render_graph::graph::BnanRenderGraph;
use BnanR::core::bnan_render_graph::resource::ResourceHandle;

const SIMPLE_MESH_FILEPATH: &str = "./build/BnanR-Sample-Shaders/simple-raster-mesh.mesh.spv";
const SIMPLE_FRAG_FILEPATH: &str = "./build/BnanR-Sample-Shaders/simple-raster-mesh.frag.spv";


#[repr(C)]
pub struct SimpleUBO {
    pub projection: Matrix4<f32>,
    pub view: Matrix4<f32>,
}

pub struct SimpleSystem {
    pub device: ArcMut<BnanDevice>,
    pub render_graph: ArcMut<BnanRenderGraph>,
    pub depth_handle: ResourceHandle,
    pub color_handle: ResourceHandle,
    pub descriptor_pool: BnanDescriptorPool,
    pub descriptor_set_layout: BnanDescriptorSetLayout,
    pub ubo_buffers: [BnanBuffer; FRAMES_IN_FLIGHT],
    pub depth_images: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT],
    pub color_images: [ArcMut<BnanImage>; FRAMES_IN_FLIGHT],
    pub descriptor_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],
    pub pipeline: BnanPipeline,
}

impl WindowObserver<(i32, i32)> for SimpleSystem {
    fn update(&mut self, data: (i32, i32)) {
        let (width, height) = data;

        unsafe {
            let device_guard = self.device.lock().unwrap();
            device_guard.device.queue_wait_idle(device_guard.graphics_queue).unwrap();
        }

        let extent = vk::Extent2D::default()
            .width(width as u32)
            .height(height as u32);

        self.depth_images = Self::create_depth_images(self.device.clone(), extent).unwrap();
        self.color_images = Self::create_color_images(self.device.clone(), extent).unwrap();
        
        self.render_graph.lock().unwrap().update_render_image(&self.depth_handle, self.depth_images.clone());
        self.render_graph.lock().unwrap().update_render_image(&self.color_handle, self.color_images.clone());

    }
}

impl SimpleSystem {
    pub fn new(device: ArcMut<BnanDevice>, render_graph: ArcMut<BnanRenderGraph>) -> Result<SimpleSystem> {

        let swapchain_extent: vk::Extent2D;

        {
            let rendergraph_guard = render_graph.lock().unwrap();
            let swapchain_guard = rendergraph_guard.swapchain.lock().unwrap();
            swapchain_extent = swapchain_guard.extent;
        }

        let ubo_buffers = Self::create_uniform_buffers(device.clone())?;
        let depth_images = Self::create_depth_images(device.clone(), swapchain_extent)?;
        let color_images = Self::create_color_images(device.clone(), swapchain_extent)?;
        
        let depth_handle = render_graph.lock().unwrap().import_render_image("DepthBuffer", depth_images.clone());
        let color_handle = render_graph.lock().unwrap().import_render_image("Color", color_images.clone());
        
        let descriptor_set_layout = Self::create_descriptor_set_layout(device.clone())?;
        let descriptor_pool = Self::create_descriptor_pool(device.clone())?;
        let descriptor_sets = Self::allocate_descriptor_sets(device.clone(), &descriptor_set_layout, &descriptor_pool, &ubo_buffers)?;
        
        let pipeline = Self::create_pipeline(device.clone(), &descriptor_set_layout, color_images[0].lock().unwrap().format, depth_images[0].lock().unwrap().format)?;

        Ok(SimpleSystem {
            device,
            render_graph,
            depth_handle,
            color_handle,
            descriptor_pool,
            descriptor_set_layout,
            ubo_buffers,
            depth_images,
            color_images,
            descriptor_sets,
            pipeline,
        })
    }

    pub fn update_uniform_buffers(&mut self, frame_info: &BnanFrameInfo, camera: &BnanCamera) -> Result<()> {

        let ubo = SimpleUBO { projection: camera.projection_matrix, view: camera.view_matrix };
        let data = unsafe { std::slice::from_raw_parts::<u8>(&ubo as *const SimpleUBO as *const u8, size_of::<SimpleUBO>()) };

        self.ubo_buffers[frame_info.frame_index].write_to_buffer(data, 0)?;
        self.ubo_buffers[frame_info.frame_index].flush(vk::WHOLE_SIZE, 0)?;

        Ok(())
    }

    pub fn draw(&self, frame_info: &BnanFrameInfo) {
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

            device_guard.device.cmd_bind_pipeline(frame_info.main_command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline);
            device_guard.device.cmd_bind_descriptor_sets(frame_info.main_command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline_layout, 0, &[self.descriptor_sets[frame_info.frame_index]], &[]);
            
            // Mesh shaders require cmd_draw_mesh_tasks instead of cmd_draw
            let mesh_shader = ext::mesh_shader::Device::new(&device_guard.instance, &device_guard.device);
            mesh_shader.cmd_draw_mesh_tasks(frame_info.main_command_buffer, 1, 1, 1);
        }
    }
    
    fn create_uniform_buffers(device: ArcMut<BnanDevice>) -> Result<[BnanBuffer; FRAMES_IN_FLIGHT]> {
        let vec: Result<Vec<_>> = (0..FRAMES_IN_FLIGHT).map(|_|
            BnanBuffer::new(device.clone(), size_of::<SimpleUBO>() as vk::DeviceSize, 1, vk::BufferUsageFlags::UNIFORM_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
        ).collect();

        let mut buffers: [BnanBuffer; FRAMES_IN_FLIGHT] = vec?.try_into().map_err(|_| anyhow!("something went wrong while creating uniform buffers"))?;

        for buffer in &mut buffers {
            buffer.map()?;
        }

        Ok(buffers)
    }

    fn create_depth_images(device: ArcMut<BnanDevice>, extent: vk::Extent2D) -> Result<[ArcMut<BnanImage>; FRAMES_IN_FLIGHT]> {
        let format = device.lock().unwrap().find_depth_format()?;
        let sample_count = device.lock().unwrap().msaa_samples;

        let extent = vk::Extent3D::default()
            .width(extent.width)
            .height(extent.height)
            .depth(1);

        let vec: Result<Vec<_>> = (0..FRAMES_IN_FLIGHT).map(|_|
            Ok(make_arcmut(BnanImage::new(device.clone(), format, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, extent, sample_count)?))
        ).collect();

        let images = vec?.try_into().map_err(|_| anyhow!("something went wrong while creating depth images"))?;
        Ok(images)
    }

    fn create_color_images(device: ArcMut<BnanDevice>, extent: vk::Extent2D) -> Result<[ArcMut<BnanImage>; FRAMES_IN_FLIGHT]> {
        let format = vk::Format::B8G8R8A8_SRGB;
        let sample_count = device.lock().unwrap().msaa_samples;

        let extent = vk::Extent3D::default()
            .width(extent.width)
            .height(extent.height)
            .depth(1);

        let vec: Result<Vec<_>> = (0..FRAMES_IN_FLIGHT).map(|_|
            Ok(make_arcmut(BnanImage::new(device.clone(), format, vk::ImageUsageFlags::COLOR_ATTACHMENT, extent, sample_count)?))
        ).collect();

        let images = vec?.try_into().map_err(|_| anyhow!("something went wrong while creating depth images"))?;
        Ok(images)
    }

    fn create_descriptor_set_layout(device: ArcMut<BnanDevice>) -> Result<BnanDescriptorSetLayout> {
        BnanDescriptorSetLayoutBuilder::new(vk::DescriptorSetLayoutCreateFlags::empty())
            .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::FRAGMENT)
//            .add_binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
//            .add_binding(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
//            .add_binding(3, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
//            .add_binding(4, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT)
            .build(device)
    }
    
    fn create_descriptor_pool(device: ArcMut<BnanDevice>) -> Result<BnanDescriptorPool> {
        BnanDescriptorPoolBuilder::new(FRAMES_IN_FLIGHT as u32, vk::DescriptorPoolCreateFlags::empty())
            .add_pool_size(vk::DescriptorType::UNIFORM_BUFFER, FRAMES_IN_FLIGHT as u32)
//            .add_pool_size(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, FRAMES_IN_FLIGHT as u32)
            .build(device)
    }

    fn allocate_descriptor_sets(device: ArcMut<BnanDevice>, layout: &BnanDescriptorSetLayout, pool: &BnanDescriptorPool, /* images: Vec<BnanImage>, sampler: vk::Sampler, */ uniform_buffers: &[BnanBuffer; FRAMES_IN_FLIGHT]) -> Result<[vk::DescriptorSet; FRAMES_IN_FLIGHT]> {

        let sets: Result<Vec<_>> = (0..FRAMES_IN_FLIGHT).map(|frame| {

            /*

            let diff_img_info = vec![
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(images[0].image_view)
                    .sampler(sampler)
            ];

            let normal_img_info = vec![
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(images[1].image_view)
                    .sampler(sampler)
            ];

            let metal_img_info = vec![
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(images[2].image_view)
                    .sampler(sampler)
            ];

            let rough_img_info = vec![
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(images[3].image_view)
                    .sampler(sampler)
            ];

             */
            
            let buffer_info = vec![
                vk::DescriptorBufferInfo::default()
                    .buffer(uniform_buffers[frame].buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            ];
            
            BnanDescriptorWriter::new(layout)
                .write_uniform_buffer(0, &buffer_info)
//                .write_combined_image_sampler(1, &diff_img_info)
//                .write_combined_image_sampler(2, &normal_img_info)
//                .write_combined_image_sampler(3, &metal_img_info)
//                .write_combined_image_sampler(4, &rough_img_info)
                .write(device.clone(), pool)
                
        }).collect();
        
        let sets_vec = sets?;
        let sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT] = sets_vec.try_into().map_err(|_| anyhow!("something went wrong while creating descriptor sets"))?;
        
        Ok(sets)
    }
    
    fn create_pipeline(device: ArcMut<BnanDevice>, layout: &BnanDescriptorSetLayout, color_format: vk::Format, depth_format: vk::Format) -> Result<BnanPipeline> {
        let layouts = [layout.layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
            
        let pipeline_layout = unsafe { device.lock().unwrap().device.create_pipeline_layout(&pipeline_layout_info, None)? };
        let mut pipeline_config_info = GraphicsPipelineConfigInfo::default();

        let color_formats = [color_format];

        pipeline_config_info.pipeline_rendering_info = pipeline_config_info.pipeline_rendering_info
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(depth_format);

        pipeline_config_info.multisample_info.rasterization_samples = device.lock().unwrap().msaa_samples;
        pipeline_config_info.pipeline_layout = pipeline_layout;

         Ok(BnanPipeline::new_graphics_pipeline(device, SIMPLE_MESH_FILEPATH.to_owned(), SIMPLE_FRAG_FILEPATH.to_owned(), &mut pipeline_config_info)?)
    }
}