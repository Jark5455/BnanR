#[allow(dead_code)]

use anyhow::*;
use ash::*;
use cgmath::*;

use BnanR::core::ArcMut;
use BnanR::core::bnan_buffer::BnanBuffer;
use BnanR::core::bnan_descriptors::*;
use BnanR::core::bnan_device::BnanDevice;
use BnanR::core::bnan_image::BnanImage;
use BnanR::core::bnan_pipeline::BnanPipeline;
use BnanR::core::bnan_rendering::{BnanFrameInfo, FRAMES_IN_FLIGHT};
use BnanR::core::bnan_window::WindowObserver;

const SIMPLE_COMP_FILEPATH: &str = "./build/BnanR-Sample-Shaders/simple.comp.spv";

#[repr(C, align(16))]
pub struct Vec4(Vector4<f32>);

#[repr(C)]
pub struct SimpleUBO {
    num_spheres: i32,
    delta_time: f32,
    aspect_ratio: f32,
    focal_length: f32,
    viewport_width: f32,
    viewport_height: f32,
    vfov: f32,
    camera_center: Vec4,
    viewport_u: Vec4,
    viewport_v: Vec4,
    pixel_delta_u: Vec4,
    pixel_delta_v: Vec4,
    viewport_upper_left: Vec4,
    pixel00_loc: Vec4,
    look_from: Vec4,
    look_at: Vec4,
    vup: Vec4,
}

#[repr(C)]
pub struct Sphere {
    material: i32,
    radius: f32,
    fuzz: f32,
    ri: f32,
    center: Vec4,
    albedo: Vec4,
}

impl SimpleUBO {
    pub fn new(image_extent: vk::Extent2D, num_spheres: i32, delta_time: f32) -> SimpleUBO {

        let look_from = Vec4(Vector4::new(0.0, 0.0, 0.0, 0.0));
        let look_at = Vec4(Vector4::new(0.0, 0.0, -1.0, 0.0));
        let vup = Vec4(Vector4::new(0.0, 1.0, 0.0, 0.0));

        let aspect_ratio = image_extent.width as f32 / image_extent.height as f32;
        let focal_length = (look_from.0 - look_at.0).magnitude();

        let vfov = 90.0_f32.to_radians();
        let h = (vfov / 2.0).tan();

        let viewport_height = 2.0 * h * focal_length;
        let viewport_width = viewport_height * aspect_ratio;
        let camera_center = Vec4(Vector4::new(0.0, 0.0, 0.0, 0.0));

        let w = (look_from.0 - look_at.0).truncate().normalize();
        let u = vup.0.truncate().cross(w).normalize();
        let v = w.cross(u);
        
        let viewport_u = Vec4(viewport_width * u.extend(0.0));
        let viewport_v = Vec4(viewport_height * -v.extend(0.0));
        
        let pixel_delta_u = Vec4(viewport_u.0 / image_extent.width as f32);
        let pixel_delta_v = Vec4(viewport_v.0 / image_extent.height as f32);
        
        let viewport_upper_left = Vec4(camera_center.0 - (focal_length * w.extend(0.0)) - viewport_u.0/2.0 - viewport_v.0/2.0);
        let pixel00_loc = Vec4(viewport_upper_left.0 + 0.5 * (pixel_delta_u.0 + pixel_delta_v.0));

        SimpleUBO {
            aspect_ratio,
            focal_length,
            viewport_width,
            viewport_height,
            camera_center,
            viewport_u,
            viewport_v,
            vfov,
            pixel_delta_u,
            pixel_delta_v,
            viewport_upper_left,
            pixel00_loc,
            num_spheres,
            delta_time,
            look_from,
            look_at,
            vup
        }
    }
}

pub struct SimpleSystem {
    pub device: ArcMut<BnanDevice>,
    pub descriptor_pool: BnanDescriptorPool,
    pub descriptor_set_layout: BnanDescriptorSetLayout,
    pub draw_images: [BnanImage; FRAMES_IN_FLIGHT],
    pub ubo_buffers: [BnanBuffer; FRAMES_IN_FLIGHT],
    pub storage_buffers: [BnanBuffer; FRAMES_IN_FLIGHT],
    pub descriptor_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],
    pub pipeline: BnanPipeline,
    pub draw_image_sampler: vk::Sampler,
    pub spheres: Vec<Sphere>,
}

impl WindowObserver<(i32, i32)> for SimpleSystem {
    fn update(&mut self, data: (i32, i32)) {

        let queue = self.device.lock().unwrap().graphics_queue;
        unsafe { self.device.lock().unwrap().device.queue_wait_idle(queue).unwrap() };

        let extent = vk::Extent3D::default()
            .width(data.0 as u32)
            .height(data.1 as u32)
            .depth(1);

        let vec: Result<Vec<_>> = (0..FRAMES_IN_FLIGHT).map(|_| BnanImage::new(self.device.clone(), vk::Format::R16G16B16A16_SFLOAT, vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC, extent)).collect();

        let images: [BnanImage; FRAMES_IN_FLIGHT] = match vec.unwrap().try_into() {
            Result::Ok(fences) => fences,
            Err(_) => panic!("something went wrong while creating fences"),
        };

        for image in &images {
            self.device.lock().unwrap().transition_image_layout_async(image.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL, None, None, None).unwrap();
        }

        let queue = self.device.lock().unwrap().compute_queue;
        unsafe { self.device.lock().unwrap().device.queue_wait_idle(queue).unwrap() };

        self.draw_images = images;
        
        for (frame, set) in self.descriptor_sets.iter().enumerate() {
            let image_info = vec![
                vk::DescriptorImageInfo::default()
                    .sampler(self.draw_image_sampler)
                    .image_view(self.draw_images[frame].image_view)
                    .image_layout(vk::ImageLayout::GENERAL)
            ];

            let buffer_info = vec![
                vk::DescriptorBufferInfo::default()
                    .buffer(self.ubo_buffers[frame].buffer)
                    .range(vk::WHOLE_SIZE)
                    .offset(0)
            ];

            let storage_buffer_info = vec![
                vk::DescriptorBufferInfo::default()
                    .buffer(self.storage_buffers[frame].buffer)
                    .range(vk::WHOLE_SIZE)
                    .offset(0)
            ];

            BnanDescriptorWriter::new(&self.descriptor_set_layout)
                .write_storage_image(0, &image_info)
                .write_uniform_buffer(1, &buffer_info)
                .write_storage_buffer(2, &storage_buffer_info)
                .overwrite(self.device.clone(), &self.descriptor_pool, *set);
        }
    }
}

impl Drop for SimpleSystem {
    fn drop(&mut self) {
        unsafe {
            self.device.lock().unwrap().device.destroy_sampler(self.draw_image_sampler, None);
        }
    }
}

impl SimpleSystem {
    pub fn new(device: ArcMut<BnanDevice>, initial_extent: vk::Extent2D) -> Result<SimpleSystem> {

        let extent = vk::Extent3D::default()
            .width(initial_extent.width)
            .height(initial_extent.height)
            .depth(1);

        let spheres = vec![
            
            Sphere {
                material: 0,
                center: Vec4(Vector4::new(0.0, -100.5, -1.0, 0.0)),
                radius: 100.0,
                fuzz: 0.0,
                ri: 0.0,
                albedo: Vec4(Vector4::new(34.0 / 255.0, 34.0 / 255.0, 34.0 / 255.0, 0.0)),
            },
            
            Sphere {
                material: 0,
                center: Vec4(Vector4::new(0.0, 0.0, -1.2, 0.0)),
                radius: 0.5,
                fuzz: 0.0,
                ri: 0.0,
                albedo: Vec4(Vector4::new(0.1, 0.2, 0.5, 0.0)),
            },

            Sphere {
                material: 2,
                center: Vec4(Vector4::new(-1.0, 0.0, -1.0, 0.0)),
                radius: 0.5,
                fuzz: 0.0,
                ri: 1.5,
                albedo: Vec4(Vector4::new(0.8, 0.8, 0.8, 0.0)),
            },

            Sphere {
                material: 1,
                center: Vec4(Vector4::new(1.0, 0.0, -1.0, 0.0)),
                radius: 0.5,
                fuzz: 0.25,
                ri: 0.0,
                albedo: Vec4(Vector4::new(0.8, 0.6, 0.2, 0.0)),
            },
        ];

        let storage_buffers = Self::create_storage_buffers(device.clone(), spheres.len() as u32)?;
        
        let descriptor_pool = Self::create_descriptor_pool(device.clone())?;
        let descriptor_set_layout = Self::create_descriptor_set_layout(device.clone())?;
        let draw_images = Self::create_draw_images(device.clone(), extent)?;
        let draw_image_sampler = Self::create_image_sampler(device.clone())?;
        let ubo_buffers = Self::create_uniform_buffers(device.clone())?;
        let descriptor_sets = Self::allocate_descriptor_sets(device.clone(), &descriptor_set_layout, &descriptor_pool, &draw_images, &ubo_buffers, &storage_buffers, draw_image_sampler)?;
        let pipeline = Self::create_pipeline(device.clone(), &descriptor_set_layout)?;
        
        Ok (
            SimpleSystem {
                device,
                descriptor_pool,
                descriptor_set_layout,
                descriptor_sets,
                draw_images,
                ubo_buffers,
                pipeline,
                draw_image_sampler,
                spheres,
                storage_buffers
            }
        )
    }

    pub fn draw(&self, frame_info: &BnanFrameInfo) {

        let extent = self.draw_images[0].image_extent;

        let x = (extent.width as f32 / 32.0).ceil() as u32;
        let y = (extent.height as f32 / 32.0).ceil() as u32;

        let offsets = [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D { x: extent.width as i32, y: extent.height as i32, z: 1 },
        ];

        let subresources = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_array_layer(0)
            .layer_count(1)
            .mip_level(0);

        let blit = [
            vk::ImageBlit2::default()
                .src_offsets(offsets)
                .dst_offsets(offsets)
                .src_subresource(subresources)
                .dst_subresource(subresources)
        ];

        let blit_info = vk::BlitImageInfo2::default()
            .dst_image(frame_info.swapchain_image)
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_image(self.draw_images[frame_info.frame_index].image)
            .src_image_layout(vk::ImageLayout::GENERAL)
            .filter(vk::Filter::LINEAR)
            .regions(&blit);

        unsafe {
            self.device.lock().unwrap().device.cmd_bind_pipeline(frame_info.main_command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline.pipeline);
            self.device.lock().unwrap().device.cmd_bind_descriptor_sets(frame_info.main_command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline.pipeline_layout, 0, &[self.descriptor_sets[frame_info.frame_index]], &[]);

            self.device.lock().unwrap().device.cmd_dispatch(frame_info.main_command_buffer, x, y, 1);

            self.device.lock().unwrap().flush_writes(frame_info.main_command_buffer, self.draw_images[frame_info.frame_index].image, vk::ImageLayout::GENERAL, vk::AccessFlags2::MEMORY_WRITE, vk::AccessFlags2::TRANSFER_READ, vk::PipelineStageFlags2::ALL_COMMANDS, vk::PipelineStageFlags2::TRANSFER, None, None);
            self.device.lock().unwrap().device.cmd_blit_image2(frame_info.main_command_buffer, &blit_info);
        }
    }

    pub fn update_uniform_buffers(&mut self, frame_info: &BnanFrameInfo) -> Result<()> {
        
        let extent = self.draw_images[0].image_extent;
        let image_extent = vk::Extent2D::default()
            .width(extent.width)
            .height(extent.height);

        let ubo_object = SimpleUBO::new(image_extent, self.spheres.len() as i32, frame_info.frame_time);

        for buffer in &mut self.ubo_buffers {
            let mapped = buffer.map()?;
            unsafe { std::ptr::copy::<SimpleUBO>(&ubo_object, mapped as *mut SimpleUBO, 1); }
            buffer.unmap();
        }
        
        Ok(())
    }
    
    pub fn update_storage_buffers(&mut self) -> Result<()> {
        for buffer in &mut self.storage_buffers {
            let mapped = buffer.map()?;
            unsafe { std::ptr::copy::<Sphere>(self.spheres.as_ptr(), mapped as *mut Sphere, self.spheres.len()); }
            buffer.unmap();
        }

        Ok(())
    }

    fn create_descriptor_pool(device: ArcMut<BnanDevice>) -> Result<BnanDescriptorPool> {
        BnanDescriptorPoolBuilder::new(FRAMES_IN_FLIGHT as u32, vk::DescriptorPoolCreateFlags::empty())
            .add_pool_size(vk::DescriptorType::STORAGE_IMAGE, 1)
            .add_pool_size(vk::DescriptorType::UNIFORM_BUFFER, 1)
            .add_pool_size(vk::DescriptorType::STORAGE_BUFFER, 1)
            .build(device.clone())
    }
    
    fn create_descriptor_set_layout(device: ArcMut<BnanDevice>) -> Result<BnanDescriptorSetLayout> {
        BnanDescriptorSetLayoutBuilder::new(vk::DescriptorSetLayoutCreateFlags::empty())
            .add_binding(0, vk::DescriptorType::STORAGE_IMAGE, vk::ShaderStageFlags::COMPUTE)
            .add_binding(1, vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::COMPUTE)
            .add_binding(2, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE)
            .build(device.clone())
    }

    fn create_draw_images(device: ArcMut<BnanDevice>, extent: vk::Extent3D) -> Result<[BnanImage; FRAMES_IN_FLIGHT]> {
        let vec: Result<Vec<_>> = (0..FRAMES_IN_FLIGHT).map(|_| BnanImage::new(device.clone(), vk::Format::R16G16B16A16_SFLOAT, vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC, extent) ).collect();

        let images: [BnanImage; FRAMES_IN_FLIGHT] = match vec?.try_into() {
            Result::Ok(images) => images,
            Err(_) => bail!("something went wrong while creating draw images"),
        };

        for image in &images {
            device.lock().unwrap().transition_image_layout_async(image.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL, None, None, None)?;
        }

        let queue = device.lock().unwrap().compute_queue;
        unsafe { device.lock().unwrap().device.queue_wait_idle(queue)? };

        Ok(images)
    }

    fn create_uniform_buffers(device: ArcMut<BnanDevice>) -> Result<[BnanBuffer; FRAMES_IN_FLIGHT]> {
        let vec: Result<Vec<_>> = (0..FRAMES_IN_FLIGHT).map(|_| unsafe {
                BnanBuffer::new(device.clone(), size_of::<SimpleUBO>() as vk::DeviceSize, 1, vk::BufferUsageFlags::UNIFORM_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE)
            }
        ).collect();

        let images: [BnanBuffer; FRAMES_IN_FLIGHT] = match vec?.try_into() {
            Result::Ok(buffer) => buffer,
            Err(_) => bail!("something went wrong while creating simple camera UBO buffers"),
        };

        Ok(images)
    }

    fn create_storage_buffers(device: ArcMut<BnanDevice>, count: u32) -> Result<[BnanBuffer; FRAMES_IN_FLIGHT]> {
        let vec: Result<Vec<_>> = (0..FRAMES_IN_FLIGHT).map(|_| unsafe {
            BnanBuffer::new(device.clone(), size_of::<Sphere>() as vk::DeviceSize, count, vk::BufferUsageFlags::STORAGE_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE)
        }
        ).collect();

        let images: [BnanBuffer; FRAMES_IN_FLIGHT] = match vec?.try_into() {
            Result::Ok(buffer) => buffer,
            Err(_) => bail!("something went wrong while creating sphere storage buffers"),
        };

        Ok(images)
    }
    
    fn create_image_sampler(device: ArcMut<BnanDevice>) -> Result<vk::Sampler> {
        let info = vk::SamplerCreateInfo::default();
        unsafe { Ok(device.lock().unwrap().device.create_sampler(&info, None)?) }
    }

    fn allocate_descriptor_sets(device: ArcMut<BnanDevice>, layout: &BnanDescriptorSetLayout, pool: &BnanDescriptorPool, images: &[BnanImage; FRAMES_IN_FLIGHT], uniform_buffers: &[BnanBuffer; FRAMES_IN_FLIGHT], storage_buffers: &[BnanBuffer; FRAMES_IN_FLIGHT], sampler: vk::Sampler) -> Result<[vk::DescriptorSet; FRAMES_IN_FLIGHT]> {

        let sets: Result<Vec<_>> = (0..FRAMES_IN_FLIGHT).map(|frame| {
            let image_info = vec![
                vk::DescriptorImageInfo::default()
                    .sampler(sampler)
                    .image_view(images[frame].image_view)
                    .image_layout(vk::ImageLayout::GENERAL)
            ];

            let uniform_buffer_info = vec![
                vk::DescriptorBufferInfo::default()
                    .buffer(uniform_buffers[frame].buffer)
                    .range(vk::WHOLE_SIZE)
                    .offset(0)
            ];

            let storage_buffer_info = vec![
                vk::DescriptorBufferInfo::default()
                    .buffer(storage_buffers[frame].buffer)
                    .range(vk::WHOLE_SIZE)
                    .offset(0)
            ];

            BnanDescriptorWriter::new(&layout)
                .write_storage_image(0, &image_info)
                .write_uniform_buffer(1, &uniform_buffer_info)
                .write_storage_buffer(2, &storage_buffer_info)
                .write(device.clone(), pool)
        }).collect();

        let sets_vec = sets?;

        let sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT] = match sets_vec.try_into() {
            Result::Ok(sets) => sets,
            Err(_) => bail!("something went wrong while creating descriptor sets"),
        };

        Ok(sets)
    }

    fn create_pipeline(device: ArcMut<BnanDevice>, descriptor_set_layout: &BnanDescriptorSetLayout) -> Result<BnanPipeline> {
        let layouts = vec![descriptor_set_layout.layout];
        Ok(BnanPipeline::new_compute_pipeline(device, SIMPLE_COMP_FILEPATH.to_owned(), layouts)?)
    }
}