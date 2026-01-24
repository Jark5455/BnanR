
use anyhow::*;
use ash::*;
use vk_mem::*;

use crate::core::ArcMut;
use crate::core::bnan_device::BnanDevice;

pub struct BnanBuffer {
    pub device: ArcMut<BnanDevice>,
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub buffer_size: vk::DeviceSize,
    pub instance_count: u32,
    pub instance_size: vk::DeviceSize,
    pub alignment_size: vk::DeviceSize,

    pub mapped: *mut u8,
}

impl Drop for BnanBuffer {
    fn drop(&mut self) {
        self.unmap();
        unsafe { self.device.lock().unwrap().allocator.destroy_buffer(self.buffer, &mut self.allocation) };
    }
}

impl BnanBuffer {

    pub fn new(device: ArcMut<BnanDevice>, instance_size: vk::DeviceSize, instance_count: u32, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags) -> Result<BnanBuffer> {
        let alignment_size = match usage {
            vk::BufferUsageFlags::STORAGE_BUFFER => {
                let min_offset_alignment = device.lock().unwrap().get_physical_device_properties().properties.limits.min_storage_buffer_offset_alignment;
                Self::get_alignment_size(instance_size, min_offset_alignment)
            },

            vk::BufferUsageFlags::UNIFORM_BUFFER => {
                let min_offset_alignment = device.lock().unwrap().get_physical_device_properties().properties.limits.min_uniform_buffer_offset_alignment;
                Self::get_alignment_size(instance_size, min_offset_alignment)
            },

            _ => {
                Self::get_alignment_size(instance_size, 1)
            }
        };
        
        let buffer_size = alignment_size * instance_count as u64;

        let (buffer, allocation) = Self::create_buffer(device.clone(), buffer_size, usage, properties)?;

        Ok(
            BnanBuffer {
                device,
                buffer,
                allocation,
                buffer_size,
                instance_count,
                instance_size,
                alignment_size,
                mapped: std::ptr::null_mut(),
            }
        )
    }

    pub fn map(&mut self) -> Result<*mut u8> {
        unsafe { self.mapped = self.device.lock().unwrap().allocator.map_memory(&mut self.allocation)? }
        Ok(self.mapped)
    }

    pub fn unmap(&mut self) {
        if self.mapped != std::ptr::null_mut() {
            unsafe { self.device.lock().unwrap().allocator.unmap_memory(&mut self.allocation) }
            self.mapped = std::ptr::null_mut();
        }
    }

    pub fn write_to_buffer(&mut self, data: &[u8], offset: vk::DeviceSize) -> Result<()> {
        if self.mapped.is_null() { bail!("Cannot copy to unmapped buffer") }

        unsafe {
            let dest = self.mapped.add(offset as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dest, data.len());
        }
        
        Ok(())
    }

    pub fn flush(&self, size: vk::DeviceSize, offset: vk::DeviceSize) -> Result<()> {
        self.device.lock().unwrap().allocator.flush_allocation(&self.allocation, offset, size)?;
        Ok(())
    }

    pub fn invalidate(&self, size: vk::DeviceSize, offset: vk::DeviceSize) -> Result<()> {
        self.device.lock().unwrap().allocator.invalidate_allocation(&self.allocation, offset, size)?;
        Ok(())
    }
    
    fn create_buffer(device: ArcMut<BnanDevice>, buffer_size: vk::DeviceSize, usage: vk::BufferUsageFlags, props: vk::MemoryPropertyFlags) -> Result<(vk::Buffer, Allocation)> {

        let info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_info = AllocationCreateInfo {
            usage: MemoryUsage::Auto,
            required_flags: props,
            flags: AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        unsafe { Ok(device.lock().unwrap().allocator.create_buffer(&info, &allocation_info)?) }
    }

    fn get_alignment_size(instance_size: vk::DeviceSize, min_offset_alignment: vk::DeviceSize) -> vk::DeviceSize {
        if min_offset_alignment > 0 {
            return (instance_size + min_offset_alignment - 1) & !(min_offset_alignment - 1);
        }
        
        instance_size
    }
}