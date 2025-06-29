use std::time::Instant;

use anyhow::*;
use ash::*;

use crate::core::ArcMut;
use crate::core::bnan_device::BnanDevice;
use crate::core::bnan_swapchain::BnanSwapchain;
use crate::core::bnan_window::BnanWindow;

#[derive(Copy, Clone)]
#[derive(Debug)]
pub struct BnanFrameInfo {
    pub frame_index: usize,
    pub frame_time: f32,
    pub command_pool: vk::CommandPool,
    pub main_command_buffer: vk::CommandBuffer,
    pub swapchain_image: vk::Image,
}

pub const FRAMES_IN_FLIGHT: usize = 2;

impl BnanFrameInfo {
    pub fn new(command_pool: vk::CommandPool, main_command_buffer: vk::CommandBuffer) -> BnanFrameInfo {
        BnanFrameInfo {
            frame_index: 0,
            frame_time: 0.0,
            command_pool,
            main_command_buffer,
            swapchain_image: vk::Image::null(),
        }
    }
}

pub struct BnanRenderHelper {
    pub window: ArcMut<BnanWindow>,
    pub device: ArcMut<BnanDevice>,
    pub swapchain: ArcMut<BnanSwapchain>,
    pub command_buffers: [vk::CommandBuffer; FRAMES_IN_FLIGHT],
    pub current_image_index: u32,
    pub current_frame_index: usize,
    pub current_frame: u64,
    pub is_frame_started: bool,
    pub frame_data: [BnanFrameInfo; FRAMES_IN_FLIGHT],
    pub acquire_semaphores: [vk::Semaphore; FRAMES_IN_FLIGHT],
    pub render_fences: [vk::Fence; FRAMES_IN_FLIGHT],
    pub render_semaphores: Vec<vk::Semaphore>,
    pub frame_time_reference: Instant,
}

impl Drop for BnanRenderHelper {
    fn drop(&mut self) {
        
        let queue = self.device.lock().unwrap().graphics_queue;
        unsafe { self.device.lock().unwrap().device.queue_wait_idle(queue).unwrap(); }
        
        for i in 0..FRAMES_IN_FLIGHT {
            unsafe {
                self.device.lock().unwrap().device.destroy_semaphore(self.acquire_semaphores[i], None);
                self.device.lock().unwrap().device.destroy_fence(self.render_fences[i], None);
            }
        }
        
        for semaphore in self.render_semaphores.iter() {
            unsafe { self.device.lock().unwrap().device.destroy_semaphore(*semaphore, None) };
        }
    }
}

impl BnanRenderHelper {
    pub fn new(window: ArcMut<BnanWindow>, device: ArcMut<BnanDevice>, swapchain: ArcMut<BnanSwapchain>) -> Result<BnanRenderHelper> {

        let image_count = swapchain.lock().unwrap().images.len() as u32;
        let command_buffers = Self::create_command_buffers(device.clone())?;
        let frame_data = Self::create_frame_data(device.clone(), &command_buffers)?;
        
        let acquire_semaphores = Self::create_semaphores(device.clone())?;
        let render_fences = Self::create_fences(device.clone())?;
        let render_semaphores = Self::create_vec_semaphores(device.clone(), image_count)?;

        Ok (
            BnanRenderHelper {
                window,
                device,
                swapchain,
                command_buffers,
                current_image_index: 0,
                current_frame_index: 0,
                current_frame: 0,
                is_frame_started: false,
                frame_data,
                acquire_semaphores,
                render_fences,
                render_semaphores,
                frame_time_reference: Instant::now(),
            }
        )
    }

    pub fn begin_frame(&mut self) -> Result<vk::CommandBuffer> {
        
        if self.is_frame_started {
            bail!("cannot begin frame if frame is started");
        }
        
        let render_fence = self.render_fences[self.current_frame_index];

        unsafe {
            self.device.lock().unwrap().device.wait_for_fences(&[render_fence], true, u64::MAX)?;
            self.device.lock().unwrap().device.reset_fences(&[render_fence])?;
        }

        let duration = self.frame_time_reference.elapsed().as_nanos() as f32 / 1000.0;
        self.frame_data[self.current_frame_index].frame_time = duration;
        self.frame_time_reference = Instant::now();
        
        let tmp_device = self.device.lock().unwrap();
        let acquire_semaphore = self.acquire_semaphores[self.current_frame_index];
        let swapchain_device = khr::swapchain::Device::new(&tmp_device.instance, &tmp_device.device);

        drop(tmp_device);

        unsafe {
            let result = swapchain_device.acquire_next_image(self.swapchain.lock().unwrap().swapchain, u64::MAX, acquire_semaphore, vk::Fence::null());

            let idx = match result {
                
                Result::Ok((idx, suboptimal)) => match suboptimal {
                    true => {
                        self.recreate_swapchain()?;
                        
                        return Err(Error::from(vk::Result::SUBOPTIMAL_KHR))
                    },
                    
                    false => idx
                },

                Err(e) => {
                    if e == vk::Result::ERROR_OUT_OF_DATE_KHR {
                        self.recreate_swapchain()?;
                        
                        return Err(Error::from(vk::Result::ERROR_OUT_OF_DATE_KHR));
                    } else {
                        bail!(e)
                    }
                }
            };

            self.current_image_index = idx;
        }

        self.is_frame_started = true;

        let command_buffer = self.get_current_frame().main_command_buffer;
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();

        unsafe {
            self.device.lock().unwrap().device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
            self.device.lock().unwrap().device.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;
        }

        let swapchain_image = self.swapchain.lock().unwrap().images[self.current_image_index as usize];
        
        self.frame_data[self.current_frame_index].swapchain_image = swapchain_image;
        self.frame_data[self.current_frame_index].frame_index = self.current_frame_index;
        
        self.device.lock().unwrap().transition_image_layout_sync(command_buffer, swapchain_image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, Some(vk::PipelineStageFlags2::TRANSFER), None, None)?;
        
        Ok(command_buffer)
    }

    pub fn end_frame(&mut self, command_buffer: vk::CommandBuffer) -> Result<()> {
        
        if !self.is_frame_started {
            bail!("cannot end frame if frame isnt started");
        }

        let tmp_device = self.device.lock().unwrap();
        let swapchain_device = khr::swapchain::Device::new(&tmp_device.instance, &tmp_device.device);
        drop(tmp_device);

        let swapchain_image = self.swapchain.lock().unwrap().images[self.current_image_index as usize];

        self.device.lock().unwrap().transition_image_layout_sync(command_buffer, swapchain_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, Some(vk::PipelineStageFlags2::NONE), None, None)?;

        unsafe {
            self.device.lock().unwrap().device.end_command_buffer(command_buffer)?;
        }

        let command_buffer_submit_info = [
            vk::CommandBufferSubmitInfo::default()
                .command_buffer(command_buffer)
        ];

        let wait_info = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.acquire_semaphores[self.current_frame_index])
                .stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .value(1)
        ];

        let signal_info = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.render_semaphores[self.current_image_index as usize])
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .value(1)
        ];

        let submit_info = [
            vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_info)
                .signal_semaphore_infos(&signal_info)
                .command_buffer_infos(&command_buffer_submit_info)
        ];

        unsafe {
            let tmp_device = self.device.lock().unwrap();
            tmp_device.device.queue_submit2(tmp_device.graphics_queue, &submit_info, self.render_fences[self.current_frame_index])?;
        }

        let swapchains = [self.swapchain.lock().unwrap().swapchain];
        let wait_semaphore = [self.render_semaphores[self.current_image_index as usize]];
        let image_index = [self.current_image_index];

        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .wait_semaphores(&wait_semaphore)
            .image_indices(&image_index);

        unsafe {

            let result = swapchain_device.queue_present(self.device.lock().unwrap().graphics_queue, &present_info);

            match result {
                Result::Ok(suboptimal) => if suboptimal {
                    self.recreate_swapchain()?;
                },

                Err(e) => {
                    if e == vk::Result::ERROR_OUT_OF_DATE_KHR {
                        self.recreate_swapchain()?;
                    } else {
                        bail!(e)
                    }
                }
            };
        }

        self.is_frame_started = false;
        self.current_frame += 1;
        self.current_frame_index = (self.current_frame_index + 1) % FRAMES_IN_FLIGHT;
        
        Ok(())
    }

    pub fn get_current_frame(&self) -> BnanFrameInfo {
        self.frame_data[self.current_frame_index]
    }

    fn recreate_swapchain(&self) -> Result<()> {
        let extent = self.window.lock().unwrap().get_window_extent();
        self.swapchain.lock().unwrap().recreate_swapchain(extent)?;
        Ok(())
    }

    fn create_command_buffers(device: ArcMut<BnanDevice>) -> Result<[vk::CommandBuffer; FRAMES_IN_FLIGHT]> {
        let device = device.lock().unwrap();
        let command_pool = device.command_pools[0];

        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .command_buffer_count(FRAMES_IN_FLIGHT as u32)
            .level(vk::CommandBufferLevel::PRIMARY);

        let command_buffers: [vk::CommandBuffer; FRAMES_IN_FLIGHT] = match unsafe { device.device.allocate_command_buffers(&command_buffer_info)?.try_into() } {
            Result::Ok(command_buffers) => command_buffers,
            Err(e) => bail!("something went wrong while creating command buffers"),
        };

        Ok(command_buffers)
    }

    fn create_frame_data(device: ArcMut<BnanDevice>, command_buffers: &[vk::CommandBuffer; FRAMES_IN_FLIGHT]) -> Result<[BnanFrameInfo; FRAMES_IN_FLIGHT]> {
        let command_pool = device.lock().unwrap().command_pools[0];
        let vec: Vec<_>= (0..FRAMES_IN_FLIGHT).map(|i| {BnanFrameInfo::new(command_pool, command_buffers[i])}).collect();

        let frame_data: [BnanFrameInfo; FRAMES_IN_FLIGHT] = match vec.try_into() {
            Result::Ok(frame_data) => frame_data,
            Err(e) => bail!("something went wrong while creating frame data"),
        };

        Ok(frame_data)
    }

    fn create_vec_semaphores(device: ArcMut<BnanDevice>, count: u32) -> Result<Vec<vk::Semaphore>> {
        let device = device.lock().unwrap();
        let semaphore_info = vk::SemaphoreCreateInfo::default();

        unsafe {
            Ok((0..count).map(|_| device.device.create_semaphore(&semaphore_info, None)).collect::<prelude::VkResult<Vec<_>>>()?)
        }
    }

    fn create_semaphores(device: ArcMut<BnanDevice>) -> Result<[vk::Semaphore; FRAMES_IN_FLIGHT]> {
        let vec = Self::create_vec_semaphores(device, FRAMES_IN_FLIGHT as u32)?;

        let semaphores: [vk::Semaphore; FRAMES_IN_FLIGHT] = match vec.try_into() {
            Result::Ok(semaphores) => semaphores,
            Err(e) => bail!("something went wrong while creating semaphores"),
        };

        Ok(semaphores)
    }

    fn create_fences(device: ArcMut<BnanDevice>) -> Result<[vk::Fence; FRAMES_IN_FLIGHT]> {
        let device = device.lock().unwrap();
        let fence_info = vk::FenceCreateInfo::default()
            .flags(vk::FenceCreateFlags::SIGNALED);

        let vec = unsafe { (0..FRAMES_IN_FLIGHT).map(|_| device.device.create_fence(&fence_info, None)).collect::<prelude::VkResult<Vec<_>>>()? };

        let fences: [vk::Fence; FRAMES_IN_FLIGHT] = match vec.try_into() {
            Result::Ok(fences) => fences,
            Err(e) => bail!("something went wrong while creating fences"),
        };

        Ok(fences)
    }
}
