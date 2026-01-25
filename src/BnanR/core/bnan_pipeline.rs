use std::fs::File;
use std::io::BufReader;

use anyhow::*;
use ash::*;
use lazy_static::lazy_static;

use crate::core::ArcMut;
use crate::core::bnan_device::BnanDevice;
use crate::core::bnan_mesh::Vertex;

lazy_static! {
    static ref DEFAULT_COLOR_BLEND_ATTACHMENT_STATE: Vec<vk::PipelineColorBlendAttachmentState> = vec![
        vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A)
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
    ];

    static ref DEFAULT_DYNAMIC_STATES: Vec<vk::DynamicState> = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
}

pub struct BnanPipeline {
    pub device: ArcMut<BnanDevice>,
    pub shader_modules: Vec<vk::ShaderModule>,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

pub struct GraphicsPipelineConfigInfo<'a> {
    pub binding_descriptions: Vec<vk::VertexInputBindingDescription>,
    pub attribute_descriptions: Vec<vk::VertexInputAttributeDescription>,
    pub viewport_info: vk::PipelineViewportStateCreateInfo<'a>,
    pub input_assembly_info: vk::PipelineInputAssemblyStateCreateInfo<'a>,
    pub rasterization_info: vk::PipelineRasterizationStateCreateInfo<'a>,
    pub multisample_info: vk::PipelineMultisampleStateCreateInfo<'a>,
    pub color_blend_info: vk::PipelineColorBlendStateCreateInfo<'a>,
    pub depth_stencil_info: vk::PipelineDepthStencilStateCreateInfo<'a>,
    pub dynamic_state_info: vk::PipelineDynamicStateCreateInfo<'a>,
    pub pipeline_rendering_info: vk::PipelineRenderingCreateInfo<'a>,
    pub pipeline_layout: vk::PipelineLayout,
    pub render_pass: vk::RenderPass,
    pub subpass: u32,
}

impl<'a> Default for GraphicsPipelineConfigInfo<'a> {
    fn default() -> GraphicsPipelineConfigInfo<'a> {
        let mut info = GraphicsPipelineConfigInfo {
            binding_descriptions: Vertex::binding_descriptions(),
            attribute_descriptions: Vertex::attribute_descriptions(),
            viewport_info: Default::default(),
            input_assembly_info: Default::default(),
            rasterization_info: Default::default(),
            multisample_info: Default::default(),
            color_blend_info: Default::default(),
            depth_stencil_info: Default::default(),
            dynamic_state_info: Default::default(),
            pipeline_rendering_info: Default::default(),
            pipeline_layout: Default::default(),
            render_pass: Default::default(),
            subpass: 0,
        };

        info.viewport_info = info.viewport_info
            .viewport_count(1)
            .scissor_count(1);

        info.input_assembly_info = info.input_assembly_info
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        info.rasterization_info = info.rasterization_info
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0);

        info.multisample_info = info.multisample_info
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);

        info.color_blend_info = info.color_blend_info
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&DEFAULT_COLOR_BLEND_ATTACHMENT_STATE)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        info.depth_stencil_info = info.depth_stencil_info
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false);

        info.dynamic_state_info = info.dynamic_state_info
            .dynamic_states(&DEFAULT_DYNAMIC_STATES);

        info
    }
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

    pub fn new_graphics_pipeline(device: ArcMut<BnanDevice>, mesh_shader: String, fragment_shader: String, task_shader: Option<String>, pipeline_config_info: &mut GraphicsPipelineConfigInfo) -> Result<BnanPipeline> {
        let mesh_shader_module = Self::load_shader_module(device.clone(), mesh_shader)?;
        let fragment_shader_module = Self::load_shader_module(device.clone(), fragment_shader)?;

        let pipeline: vk::Pipeline;

        if let Some(task_shader) = task_shader {
            let task_shader_module = Self::load_shader_module(device.clone(), task_shader)?;
            pipeline = Self::create_graphics_pipeline(device.clone(), mesh_shader_module, fragment_shader_module, Some(task_shader_module), pipeline_config_info)?;
        } else {
            pipeline = Self::create_graphics_pipeline(device.clone(), mesh_shader_module, fragment_shader_module, None, pipeline_config_info)?;
        }

        Ok(BnanPipeline {
            device,
            shader_modules: vec![mesh_shader_module, fragment_shader_module],
            pipeline_layout: pipeline_config_info.pipeline_layout,
            pipeline
        })
    }

    pub fn new_traditional_graphics_pipeline(device: ArcMut<BnanDevice>, vertex_shader: String, fragment_shader: String, pipeline_config_info: &mut GraphicsPipelineConfigInfo) -> Result<BnanPipeline> {
        let vertex_shader_module = Self::load_shader_module(device.clone(), vertex_shader)?;
        let fragment_shader_module = Self::load_shader_module(device.clone(), fragment_shader)?;
        let pipeline = Self::create_traditional_graphics_pipeline(device.clone(), vertex_shader_module, fragment_shader_module, pipeline_config_info)?;

        Ok(BnanPipeline {
            device,
            shader_modules: vec![vertex_shader_module, fragment_shader_module],
            pipeline_layout: pipeline_config_info.pipeline_layout,
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

    fn create_graphics_pipeline(device: ArcMut<BnanDevice>, mesh_shader_module: vk::ShaderModule, fragment_shader_module: vk::ShaderModule, task_shader: Option<vk::ShaderModule>, info: &mut GraphicsPipelineConfigInfo) -> Result<vk::Pipeline> {
        let mesh_shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .name(c"main")
            .stage(vk::ShaderStageFlags::MESH_EXT)
            .module(mesh_shader_module);

        let fragment_shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .name(c"main")
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module);

        let mut stages = vec![mesh_shader_stage, fragment_shader_stage];

        if let Some(task_shader) = task_shader {
            let task_shader_stage = vk::PipelineShaderStageCreateInfo::default()
                .name(c"main")
                .stage(vk::ShaderStageFlags::TASK_EXT)
                .module(task_shader);

            stages.push(task_shader_stage);
        }

        let vertex_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&info.attribute_descriptions)
            .vertex_binding_descriptions(&info.binding_descriptions);

        let pipeline = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_info)
            .viewport_state(&info.viewport_info)
            .input_assembly_state(&info.input_assembly_info)
            .rasterization_state(&info.rasterization_info)
            .multisample_state(&info.multisample_info)
            .color_blend_state(&info.color_blend_info)
            .depth_stencil_state(&info.depth_stencil_info)
            .dynamic_state(&info.dynamic_state_info)
            .layout(info.pipeline_layout)
            .render_pass(info.render_pass)
            .subpass(info.subpass)

            .base_pipeline_index(-1)
            .base_pipeline_handle(vk::Pipeline::null())
            .push_next(&mut info.pipeline_rendering_info);

        unsafe {
            Ok(device.lock().unwrap().device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline], None).unwrap()[0])
        }
    }

    fn create_traditional_graphics_pipeline(device: ArcMut<BnanDevice>, vertex_shader_module: vk::ShaderModule, fragment_shader_module: vk::ShaderModule, info: &mut GraphicsPipelineConfigInfo) -> Result<vk::Pipeline> {
        let vertex_shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .name(c"main")
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module);

        let fragment_shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .name(c"main")
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module);

        let stages = [vertex_shader_stage, fragment_shader_stage];

        let vertex_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&info.attribute_descriptions)
            .vertex_binding_descriptions(&info.binding_descriptions);

        let pipeline = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_info)
            .viewport_state(&info.viewport_info)
            .input_assembly_state(&info.input_assembly_info)
            .rasterization_state(&info.rasterization_info)
            .multisample_state(&info.multisample_info)
            .color_blend_state(&info.color_blend_info)
            .depth_stencil_state(&info.depth_stencil_info)
            .dynamic_state(&info.dynamic_state_info)
            .layout(info.pipeline_layout)
            .render_pass(info.render_pass)
            .subpass(info.subpass)

            .base_pipeline_index(-1)
            .base_pipeline_handle(vk::Pipeline::null())
            .push_next(&mut info.pipeline_rendering_info);

        unsafe {
            Ok(device.lock().unwrap().device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline], None).unwrap()[0])
        }
    }
}