use ash::vk;
use anyhow::*;
use ash::khr;
use crate::core::ArcMut;
use crate::core::bnan_device::BnanDevice;
use crate::core::bnan_rendering::FRAMES_IN_FLIGHT;
use crate::core::bnan_swapchain::BnanSwapchain;

pub struct RenderGraphSync {
    pub device: ArcMut<BnanDevice>,
    
    pub command_buffers: [vk::CommandBuffer; FRAMES_IN_FLIGHT],
    pub image_available_semaphores: [vk::Semaphore; FRAMES_IN_FLIGHT],
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: [vk::Fence; FRAMES_IN_FLIGHT],
    
    // Transfer sync
    pub transfer_command_buffers: [vk::CommandBuffer; FRAMES_IN_FLIGHT],
    pub transfer_finished_semaphores: [vk::Semaphore; FRAMES_IN_FLIGHT],
    pub transfer_fences: [vk::Fence; FRAMES_IN_FLIGHT],
    pub pending_transfer_signal: [bool; FRAMES_IN_FLIGHT],
}

impl Drop for RenderGraphSync {
    fn drop(&mut self) {
        let device = self.device.lock().unwrap();
        unsafe {
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

impl RenderGraphSync {
    pub fn new(device: ArcMut<BnanDevice>, swapchain_image_count: u32) -> Result<Self> {
        let (
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            transfer_command_buffers,
            transfer_finished_semaphores,
            transfer_fences
        ) = Self::create_sync_objects(device.clone(), swapchain_image_count)?;
        
        Ok(Self {
            device,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            transfer_command_buffers,
            transfer_finished_semaphores,
            transfer_fences,
            pending_transfer_signal: [false; FRAMES_IN_FLIGHT],
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
    
    pub fn recreate_semaphores(&mut self, image_count: usize) -> Result<()> {
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
    
    pub fn wait_and_reset_in_flight(&self, current_frame: usize) -> Result<()> {
        let device = self.device.lock().unwrap();
        unsafe {
            device.device.wait_for_fences(&[self.in_flight_fences[current_frame]], true, u64::MAX)?;
            device.device.reset_fences(&[self.in_flight_fences[current_frame]])?;
        }
        Ok(())
    }
    
    pub fn get_command_buffer(&self, current_frame: usize) -> vk::CommandBuffer {
        self.command_buffers[current_frame]
    }
}
