use anyhow::{Result, bail};
use ash::vk;
use ash::vk::DeviceSize;
use crate::core::ArcMut;
use crate::core::bnan_buffer::BnanBuffer;
use crate::core::bnan_device::BnanDevice;
use crate::core::make_arcmut;

/// Handle to a sub-allocation within the streaming buffer
#[derive(Debug)]
pub struct StreamingAllocation {
    pub offset: u64,
    pub size: u64,
    slot_index: usize,
}

/// Streaming staging buffer with free list allocator for meshlet streaming
pub struct BnanStreamingBuffer {
    device: ArcMut<BnanDevice>,
    buffer: ArcMut<BnanBuffer>,
    slot_size: u64,
    slot_count: usize,
    free_slots: Vec<usize>,
}

impl Drop for BnanStreamingBuffer {
    fn drop(&mut self) {
        self.buffer.lock().unwrap().unmap();
    }
}

impl BnanStreamingBuffer {
    /// Create a new streaming buffer
    /// 
    /// # Arguments
    /// * `device` - Vulkan device
    /// * `total_size` - Total buffer size in bytes
    /// * `slot_size` - Size of each slot (should fit largest meshlet data)
    pub fn new(device: ArcMut<BnanDevice>, total_size: u64, slot_size: u64) -> Result<Self> {
        if slot_size == 0 {
            bail!("Slot size must be greater than 0");
        }
        
        let slot_count = (total_size / slot_size) as usize;
        if slot_count == 0 {
            bail!("Total size must be at least one slot");
        }
        
        let actual_size = slot_size * slot_count as u64;
        
        // Create HOST_VISIBLE staging buffer for CPU->GPU transfer
        let buffer = BnanBuffer::new(
            device.clone(),
            actual_size,
            1,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        
        // Initialize free list with all slots available
        let free_slots: Vec<usize> = (0..slot_count).rev().collect();
        
        let buffer = make_arcmut(buffer);
        buffer.lock().unwrap().map()?;
        
        Ok(Self {
            device,
            buffer,
            slot_size,
            slot_count,
            free_slots,
        })
    }
    
    /// Allocate a slot from the free list
    /// Returns None if no slots are available
    pub fn allocate(&mut self) -> Option<StreamingAllocation> {
        self.free_slots.pop().map(|slot_index| {
            StreamingAllocation {
                offset: slot_index as u64 * self.slot_size,
                size: self.slot_size,
                slot_index,
            }
        })
    }
    
    /// Write raw data to the streaming buffer at a given offset within an allocation
    /// 
    /// # Arguments
    /// * `alloc` - The allocation to write to
    /// * `offset_in_alloc` - Offset within the allocation
    /// * `data` - Raw bytes to write
    pub fn write_data(
        &mut self,
        alloc: &StreamingAllocation,
        offset_in_alloc: u64,
        data: &[u8],
    ) -> Result<()> {
        let write_offset = alloc.offset + offset_in_alloc;
        
        if offset_in_alloc + data.len() as u64 > alloc.size {
            bail!("Write exceeds allocation bounds: offset {} + size {} > alloc size {}", 
                  offset_in_alloc, data.len(), alloc.size);
        }
        
        let mut buffer = self.buffer.lock().unwrap();
        buffer.write_to_buffer(data, write_offset)?;
        buffer.flush(alloc.size as DeviceSize, write_offset)?;
        Ok(())
    }
    
    /// Copy data from allocation to destination buffer and free the slot
    /// 
    /// # Arguments
    /// * `alloc` - The allocation to move (consumed)
    /// * `dst` - Destination buffer
    /// * `dst_offset` - Offset in destination buffer
    /// * `size` - Number of bytes to copy
    /// * `cmd` - Command buffer to record copy command
    pub fn move_to(
        &mut self,
        alloc: StreamingAllocation,
        dst: &BnanBuffer,
        dst_offset: u64,
        size: u64,
        cmd: vk::CommandBuffer,
    ) -> Result<()> {
        if size > alloc.size {
            bail!("Copy size exceeds allocation size");
        }
        
        let copy_region = vk::BufferCopy {
            src_offset: alloc.offset,
            dst_offset,
            size,
        };
        
        unsafe {
            let device = self.device.lock().unwrap();
            let buffer = self.buffer.lock().unwrap();
            device.device.cmd_copy_buffer(cmd, buffer.buffer, dst.buffer, &[copy_region]);
        }
        
        // Return slot to free list
        self.free(alloc);
        
        Ok(())
    }
    
    /// Free an allocation without copying (return slot to free list)
    pub fn free(&mut self, alloc: StreamingAllocation) {
        self.free_slots.push(alloc.slot_index);
    }
    
    /// Get the underlying buffer
    pub fn buffer(&self) -> ArcMut<BnanBuffer> {
        self.buffer.clone()
    }
    
    /// Get number of free slots
    pub fn free_slot_count(&self) -> usize {
        self.free_slots.len()
    }
    
    /// Get total number of slots
    pub fn total_slot_count(&self) -> usize {
        self.slot_count
    }
    
    /// Get slot size
    pub fn slot_size(&self) -> u64 {
        self.slot_size
    }
}
