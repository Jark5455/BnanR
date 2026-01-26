use std::collections::HashMap;

use anyhow::*;
use ash::*;

use crate::core::ArcMut;
use crate::core::bnan_device::BnanDevice;

pub struct BnanDescriptorSetLayout {
    pub device: ArcMut<BnanDevice>,
    pub layout: vk::DescriptorSetLayout,
}

impl Drop for BnanDescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.lock().unwrap().device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

pub struct BnanDescriptorSetLayoutBuilder<'a> {
    pub bindings: HashMap<u32, vk::DescriptorSetLayoutBinding<'a>>,
    pub flags: vk::DescriptorSetLayoutCreateFlags,
}

impl<'a> BnanDescriptorSetLayoutBuilder<'a> {
    pub fn new(flags: vk::DescriptorSetLayoutCreateFlags) -> BnanDescriptorSetLayoutBuilder<'a> {
        BnanDescriptorSetLayoutBuilder {
            bindings: HashMap::new(),
            flags
        }
    }

    pub fn add_binding(mut self, binding: u32, descriptor_type: vk::DescriptorType, stage_flags: vk::ShaderStageFlags) -> BnanDescriptorSetLayoutBuilder<'a> {
        let layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(descriptor_type)
            .descriptor_count(1)
            .stage_flags(stage_flags);

        self.bindings.insert(binding, layout_binding);
        self
    }

    pub fn build(&self, device: ArcMut<BnanDevice>) -> Result<BnanDescriptorSetLayout> {

        let bindings: Vec<_> = self.bindings.iter().map(|(_, binding)| *binding).collect();

        let info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .flags(self.flags);

        let layout = unsafe { device.lock().unwrap().device.create_descriptor_set_layout(&info, None)? };

        Ok ( BnanDescriptorSetLayout { device, layout } )
    }
}

pub struct BnanDescriptorPool {
    pub device: ArcMut<BnanDevice>,
    pub pool: vk::DescriptorPool,
}

impl Drop for BnanDescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device.lock().unwrap().device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

impl BnanDescriptorPool {
    pub fn allocate_descriptors(&self, descriptor_layouts: Vec<&BnanDescriptorSetLayout>) -> Result<Vec<vk::DescriptorSet>> {
        
        let layouts: Vec<_> = descriptor_layouts.iter().map(|layout| layout.layout).collect();
        
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);
        
        unsafe { Ok ( self.device.lock().unwrap().device.allocate_descriptor_sets(&info)? ) }
    }
    
    pub fn free_descriptor_sets(&self, descriptor_sets: Vec<vk::DescriptorSet>) -> Result<()> {
        unsafe { Ok ( self.device.lock().unwrap().device.free_descriptor_sets(self.pool, &descriptor_sets)? ) }
    }
    
    pub fn reset(&self) -> Result<()> {
        unsafe { Ok (  self.device.lock().unwrap().device.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())? ) }
    }
}

pub struct BnanDescriptorPoolBuilder {
    pub pool_sizes: HashMap<vk::DescriptorType, u32>,
    pub pool_flags: vk::DescriptorPoolCreateFlags,
    pub max_sets: u32,
}

impl BnanDescriptorPoolBuilder {
    pub fn new(max_sets: u32, pool_flags: vk::DescriptorPoolCreateFlags) -> BnanDescriptorPoolBuilder {
        BnanDescriptorPoolBuilder {
            pool_sizes: HashMap::new(),
            pool_flags,
            max_sets
        }
    }

    pub fn add_pool_size(mut self, descriptor_type: vk::DescriptorType, size: u32) -> BnanDescriptorPoolBuilder {
        if let Some(old_pool_size) = self.pool_sizes.get(&descriptor_type) {
            self.pool_sizes.insert(descriptor_type, *old_pool_size + size);
        } else {
            self.pool_sizes.insert(descriptor_type, size);
        }

        self
    }

    pub fn build(&self, device: ArcMut<BnanDevice>) -> Result<BnanDescriptorPool> {
        let pool_sizes: Vec<_> = self.pool_sizes.iter().map(|(descriptor_type, size)| {
            vk::DescriptorPoolSize::default()
                .ty(*descriptor_type)
                .descriptor_count(*size)
        }).collect();

        let info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(self.max_sets)
            .flags(self.pool_flags);

        let pool = unsafe { device.lock().unwrap().device.create_descriptor_pool(&info, None)? };

        Ok ( BnanDescriptorPool { device, pool } )
    }
}

pub struct BnanDescriptorWriter<'a> {
    pub layout: &'a BnanDescriptorSetLayout,
    pub writes: Vec<vk::WriteDescriptorSet<'a>>
}

impl<'a> BnanDescriptorWriter<'a> {
    pub fn new(layout: &'a BnanDescriptorSetLayout) -> BnanDescriptorWriter<'a> {
        BnanDescriptorWriter {
            writes: Vec::new(),
            layout: layout
        }
    }
    
    pub fn write_storage_image(mut self, binding: u32, image_info: &'a [vk::DescriptorImageInfo]) -> BnanDescriptorWriter<'a> {
        self.write_image(binding, vk::DescriptorType::STORAGE_IMAGE, image_info);
        self
    }

    pub fn write_sampled_image(mut self, binding: u32, image_info: &'a [vk::DescriptorImageInfo]) -> BnanDescriptorWriter<'a> {
        self.write_image(binding, vk::DescriptorType::SAMPLED_IMAGE, image_info);
        self
    }

    pub fn write_combined_image_sampler(mut self, binding: u32, image_info: &'a [vk::DescriptorImageInfo]) -> BnanDescriptorWriter<'a> {
        self.write_image(binding, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, image_info);
        self
    }
    
    fn write_image(&mut self, binding: u32, dtype: vk::DescriptorType, image_info: &'a [vk::DescriptorImageInfo]) {
        let write = vk::WriteDescriptorSet::default()
            .descriptor_type(dtype)
            .dst_binding(binding)
            .image_info(image_info);

        self.writes.push(write);
    }

    pub fn write_storage_buffer(mut self, binding: u32, buffer_info: &'a [vk::DescriptorBufferInfo]) -> BnanDescriptorWriter<'a> {
        self.write_buffer(binding, vk::DescriptorType::STORAGE_BUFFER, buffer_info);
        self
    }

    pub fn write_uniform_buffer(mut self, binding: u32, buffer_info: &'a [vk::DescriptorBufferInfo]) -> BnanDescriptorWriter<'a> {
        self.write_buffer(binding, vk::DescriptorType::UNIFORM_BUFFER, buffer_info);
        self
    }
    
    fn write_buffer(&mut self, binding: u32, dtype: vk::DescriptorType, buffer_info: &'a [vk::DescriptorBufferInfo]) {
        let write = vk::WriteDescriptorSet::default()
            .descriptor_type(dtype)
            .dst_binding(binding)
            .buffer_info(buffer_info);
        
        self.writes.push(write);
    }
    
    pub fn write(&self, device: ArcMut<BnanDevice>, pool: &BnanDescriptorPool) -> Result<vk::DescriptorSet> {
        
        let set = pool.allocate_descriptors(vec![self.layout])?[0];
        self.overwrite(device, pool, set);
        Ok(set)
    }
    
    pub fn overwrite(&self, device: ArcMut<BnanDevice>, pool: &BnanDescriptorPool, set: vk::DescriptorSet) {
        let mut writes = self.writes.clone();

        if writes.is_empty() { return; }
        for write in &mut writes {
            write.dst_set = set;
        }

        unsafe { device.lock().unwrap().device.update_descriptor_sets(&writes, &[]) };
    }
}