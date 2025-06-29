use std::fs::File;
use std::io::BufReader;

use anyhow::*;
use ash::*;

use crate::core::ArcMut;
use crate::core::bnan_device::BnanDevice;

pub struct BnanPipeline {
    pub device: ArcMut<BnanDevice>,
    pub shader_modules: Vec<vk::ShaderModule>,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl Drop for BnanPipeline {
    fn drop(&mut self) {
        for shader_module in &self.shader_modules {
            unsafe {
                self.device.lock().unwrap().device.destroy_shader_module(*shader_module, None);
            }
        }
        
        unsafe {
            self.device.lock().unwrap().device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.lock().unwrap().device.destroy_pipeline(self.pipeline, None);
        }
    }
}

impl BnanPipeline {
    
    pub fn new_compute_pipeline(device: ArcMut<BnanDevice>, shader_filepath: String, descriptor_set_layouts: Vec<vk::DescriptorSetLayout>) -> Result<BnanPipeline> {
        let shader_module = Self::load_shader_module(device.clone(), shader_filepath)?;
        let pipeline_layout = Self::create_pipeline_layout(device.clone(), descriptor_set_layouts)?;
        let pipeline = Self::create_compute_pipeline(device.clone(), shader_module, pipeline_layout)?;
        
        Ok(BnanPipeline {
            device,
            shader_modules: vec![shader_module],
            pipeline_layout,
            pipeline
        })
    }
    
    fn load_shader_module(device: ArcMut<BnanDevice>, filepath: String) -> Result<vk::ShaderModule> {

        let mut file = BufReader::new(File::open(&filepath)?);
        let shadercode = util::read_spv(&mut file)?;

        let info = vk::ShaderModuleCreateInfo::default()
            .code(shadercode.as_slice());

        unsafe { Ok(device.lock().unwrap().device.create_shader_module(&info, None)?) }
    }

    fn create_pipeline_layout(device: ArcMut<BnanDevice>, descriptor_set_layouts: Vec<vk::DescriptorSetLayout>) -> Result<vk::PipelineLayout> {
        
        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&descriptor_set_layouts);

        unsafe { Ok(device.lock().unwrap().device.create_pipeline_layout(&info, None)?) }
    }

    fn create_compute_pipeline(device: ArcMut<BnanDevice>, shader_module: vk::ShaderModule, layout: vk::PipelineLayout) -> Result<vk::Pipeline> {
        let shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .name(c"main")
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module);

        let pipeline = vk::ComputePipelineCreateInfo::default()
            .layout(layout)
            .stage(shader_stage);
        
        unsafe { Ok(device.lock().unwrap().device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline], None).unwrap()[0]) }
    }
}