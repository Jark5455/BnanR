use std::collections::VecDeque;
use anyhow::*;
use ash::*;

use crate::core::ArcMut;
use crate::core::bnan_device::BnanDevice;
use crate::core::bnan_swapchain::BnanSwapchain;
use crate::core::bnan_window::BnanWindow;

#[derive(Copy, Clone)]
pub struct BnanFrameInfo {
    pub frame_index: u64,
    pub frame_time: f32,
    pub command_pool: vk::CommandPool,
    pub main_command_buffer: vk::CommandBuffer,
}

const FRAMES_IN_FLIGHT: u32 = 2;

impl BnanFrameInfo {
    pub fn new(command_pool: vk::CommandPool, main_command_buffer: vk::CommandBuffer) -> BnanFrameInfo {
        BnanFrameInfo {
            frame_index: 0,
            frame_time: 0.0,
            command_pool,
            main_command_buffer,
        }
    }
}

pub struct BnanRenderHelper<'a> {
    pub window: ArcMut<BnanWindow<'a>>,
    pub device: ArcMut<BnanDevice>,
    pub swapchain: ArcMut<BnanSwapchain>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub current_image_index: u32,
    // pub current_frame_index: u64,
    pub is_frame_started: bool,
    pub frame_data: Vec<BnanFrameInfo>,
    pub recycled_semaphores: Vec<vk::Semaphore>,
    pub swapchain_acquire_semaphores: Vec<vk::Semaphore>,
    pub swapchain_release_semaphores: Vec<vk::Semaphore>,
    pub queue_submit_fences: Vec<vk::Fence>,
}

impl BnanRenderHelper<'_> {
    pub fn new(window: ArcMut<BnanWindow>, device: ArcMut<BnanDevice>, swapchain: ArcMut<BnanSwapchain>) -> Result<BnanRenderHelper> {

        let image_count = swapchain.lock().unwrap().images.len() as u32;
        let command_buffers = Self::create_command_buffers(device.clone(), image_count)?;
        let frame_data = Self::create_frame_data(device.clone(), &command_buffers, image_count)?;
        
        let swapchain_acquire_semaphores = vec![vk::Semaphore::null(); image_count as usize];
        let swapchain_release_semaphores = Self::create_semaphores(device.clone(), image_count)?;
        let queue_submit_fences = Self::create_fences(device.clone(), image_count)?;

        let recycled_semaphores = Vec::new();
        
        Ok (
            BnanRenderHelper {
                window,
                device,
                swapchain,
                command_buffers,
                current_image_index: 0,
                // current_frame_index: 0,
                is_frame_started: false,
                frame_data,
                recycled_semaphores,
                swapchain_acquire_semaphores,
                swapchain_release_semaphores,
                queue_submit_fences,
            }
        )
    }

    pub fn begin_frame(&mut self) -> Result<vk::CommandBuffer> {
        
        let acquire_semaphore = match self.recycled_semaphores.is_empty() {
            true => unsafe {
                let info = vk::SemaphoreCreateInfo::default();
                self.device.lock().unwrap().device.create_semaphore(&info, None)?
            },
            
            false => self.recycled_semaphores.pop().unwrap(),
        };

        let tmp_device = self.device.lock().unwrap();
        let swapchain_device = khr::swapchain::Device::new(&tmp_device.instance, &tmp_device.device);
        drop(tmp_device);

        unsafe {
            let result = swapchain_device.acquire_next_image(self.swapchain.lock().unwrap().swapchain, 1000000000, acquire_semaphore, vk::Fence::null());
            
            let idx = match result { 
                Result::Ok((idx, suboptimal)) => {
                    
                    if suboptimal {
                        self.recycled_semaphores.push(acquire_semaphore);
                        self.recreate_swapchain()?;
                        return Ok(vk::CommandBuffer::null());
                    }
                    
                    idx
                },
                
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recycled_semaphores.push(acquire_semaphore);
                    self.recreate_swapchain()?;
                    return Ok(vk::CommandBuffer::null());
                },
                
                Err(e) => bail!(e)
            };
            
            self.current_image_index = idx;
        }
        
        let queue_submit_fence = [self.queue_submit_fences[self.current_image_index as usize]];
        let command_pool = self.frame_data[self.current_image_index as usize].command_pool;
        
        unsafe {
            self.device.lock().unwrap().device.wait_for_fences(&queue_submit_fence, true, u64::MAX)?;
            self.device.lock().unwrap().device.reset_fences(&queue_submit_fence)?;
            
            // self.device.lock().unwrap().device.reset_command_pool(command_pool,  vk::CommandPoolResetFlags::empty())?;
        }
        
        let old_semaphore = self.swapchain_acquire_semaphores[self.current_image_index as usize];
        
        if old_semaphore != vk::Semaphore::null() {
            self.recycled_semaphores.push(old_semaphore);
        }
        
        self.swapchain_acquire_semaphores[self.current_image_index as usize] = acquire_semaphore;
        // self.is_frame_started = true;
        
        let command_buffer = self.command_buffers[self.current_image_index as usize];
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device.lock().unwrap().device.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;
        } 
        
        let swapchain_image = self.swapchain.lock().unwrap().images[self.current_image_index as usize];
        self.device.lock().unwrap().transition_image_layout_sync(command_buffer, swapchain_image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, Some(vk::PipelineStageFlags2::TOP_OF_PIPE), None, None)?;
        
        Ok(command_buffer)
        
        /*
        
        if self.is_frame_started {
            bail!("cannot begin frame if frame is started");
        }
        
        let render_fence = self.render_fences[self.current_frame_index as usize % FRAMES_IN_FLIGHT as usize];

        unsafe {
            self.device.lock().unwrap().device.wait_for_fences(&[render_fence], true, 1000000000)?;
            self.device.lock().unwrap().device.reset_fences(&[render_fence])?;
        }

        let tmp_device = self.device.lock().unwrap();
        let acquire_semaphore = self.acquire_semaphores[self.current_frame_index as usize % FRAMES_IN_FLIGHT as usize];
        let swapchain_device = khr::swapchain::Device::new(&tmp_device.instance, &tmp_device.device);

        drop(tmp_device);

        unsafe {
            let result = swapchain_device.acquire_next_image(self.swapchain.lock().unwrap().swapchain, 1000000000, acquire_semaphore, vk::Fence::null());

            let idx = match result {
                Result::Ok((idx, suboptimal)) => idx,

                Err(e) => {
                    if e == vk::Result::ERROR_OUT_OF_DATE_KHR {
                        u32::MAX
                    } else {
                        bail!(e)
                    }
                }
            };

            if idx == u32::MAX {
                self.recreate_swapchain()?;
                return Ok(vk::CommandBuffer::null());
            }

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

        self.device.lock().unwrap().transition_image_layout_sync(command_buffer, swapchain_image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, Some(vk::PipelineStageFlags2::TOP_OF_PIPE), None, None)?;

        Ok(command_buffer)
        
         */
        
    }

    pub fn end_frame(&mut self, command_buffer: vk::CommandBuffer) -> Result<()> {
        
        let swapchain_image = self.swapchain.lock().unwrap().images[self.current_image_index as usize];
        self.device.lock().unwrap().transition_image_layout_sync(command_buffer, swapchain_image, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, Some(vk::PipelineStageFlags2::BOTTOM_OF_PIPE), None, None)?;

        unsafe {
            self.device.lock().unwrap().device.end_command_buffer(command_buffer)?;
        }

        let command_buffer_submit_info = [
            vk::CommandBufferSubmitInfo::default()
                .command_buffer(command_buffer)
        ];

        let wait_info = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.swapchain_acquire_semaphores[self.current_image_index as usize])
                .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .value(1)
        ];

        let signal_info = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.swapchain_release_semaphores[self.current_image_index as usize])
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
            tmp_device.device.queue_submit2(tmp_device.graphics_queue, &submit_info, self.queue_submit_fences[self.current_image_index as usize])?;
        }

        let swapchains = [self.swapchain.lock().unwrap().swapchain];
        let wait_semaphore = [self.swapchain_release_semaphores[self.current_image_index as usize]];
        let image_index = [self.current_image_index];
        
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .wait_semaphores(&wait_semaphore)
            .image_indices(&image_index);

        unsafe {

            let tmp_device = self.device.lock().unwrap();
            let swapchain_device = khr::swapchain::Device::new(&tmp_device.instance, &tmp_device.device);
            drop(tmp_device);
            
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
        
        Ok(())
        
        /*
        
        if !self.is_frame_started {
            bail!("cannot end frame if frame isnt started");
        }

        let tmp_device = self.device.lock().unwrap();
        let swapchain_device = khr::swapchain::Device::new(&tmp_device.instance, &tmp_device.device);
        drop(tmp_device);

        let swapchain_image = self.swapchain.lock().unwrap().images[self.current_image_index as usize];

        self.device.lock().unwrap().transition_image_layout_sync(command_buffer, swapchain_image, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, Some(vk::PipelineStageFlags2::BOTTOM_OF_PIPE), None, None)?;

        unsafe {
            self.device.lock().unwrap().device.end_command_buffer(command_buffer)?;
        }

        let command_buffer_submit_info = [
            vk::CommandBufferSubmitInfo::default()
                .command_buffer(command_buffer)
        ];

        let wait_info = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.acquire_semaphores[self.current_frame_index as usize % FRAMES_IN_FLIGHT as usize])
                .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .value(1)
        ];

        let signal_info = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.image_semaphores[self.current_image_index as usize])
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
            tmp_device.device.queue_submit2(tmp_device.graphics_queue, &submit_info, self.render_fences[self.current_frame_index as usize % FRAMES_IN_FLIGHT as usize])?;
        }

        let swapchains = [self.swapchain.lock().unwrap().swapchain];
        let wait_semaphore = [self.image_semaphores[self.current_image_index as usize]];
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
        self.current_frame_index += 1;

        Ok(())
        
         */
    }

    pub fn get_current_frame(&self) -> BnanFrameInfo {
        // self.frame_data[self.current_frame_index as usize % FRAMES_IN_FLIGHT as usize]
        self.frame_data[self.current_image_index as usize]
    }

    fn recreate_swapchain(&self) -> Result<()> {
        let extent = self.window.lock().unwrap().get_window_extent();
        self.swapchain.lock().unwrap().recreate_swapchain(extent)?;
        Ok(())
    }

    fn create_command_buffers(device: ArcMut<BnanDevice>, count: u32) -> Result<Vec<vk::CommandBuffer>> {
        let device = device.lock().unwrap();
        let command_pool = device.command_pools[0];

        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .command_buffer_count(count)
            .level(vk::CommandBufferLevel::PRIMARY);

        unsafe { Ok(device.device.allocate_command_buffers(&command_buffer_info)?) }
    }

    fn create_frame_data(device: ArcMut<BnanDevice>, command_buffers: &Vec<vk::CommandBuffer>, count: u32) -> Result<Vec<BnanFrameInfo>> {
        let command_pool = device.lock().unwrap().command_pools[0];
        Ok((0..count).map(|i| {BnanFrameInfo::new(command_pool, command_buffers[i as usize])}).collect())
    }

    fn create_semaphores(device: ArcMut<BnanDevice>, count: u32) -> Result<Vec<vk::Semaphore>> {
        let device = device.lock().unwrap();
        let semaphore_info = vk::SemaphoreCreateInfo::default();

        unsafe {
            Ok((0..count).map(|_| device.device.create_semaphore(&semaphore_info, None)).collect::<prelude::VkResult<Vec<_>>>()?)
        }
    }

    fn create_fences(device: ArcMut<BnanDevice>, count: u32) -> Result<Vec<vk::Fence>> {
        let device = device.lock().unwrap();
        let fence_info = vk::FenceCreateInfo::default()
            .flags(vk::FenceCreateFlags::SIGNALED);

        unsafe {
            Ok((0..count).map(|_| device.device.create_fence(&fence_info, None)).collect::<prelude::VkResult<Vec<_>>>()?)
        }
    }
}
