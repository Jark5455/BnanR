use std::collections::HashSet;
use std::ffi::{c_char, c_void, CStr};
use std::ptr::slice_from_raw_parts;

use anyhow::*;
use ash::*;
use lazy_static::*;
use rayon::*;
use sdl3_sys::everything::*;
use vk_mem::*;

use crate::core::ArcMut;
use crate::core::bnan_window::BnanWindow;

lazy_static! {
    static ref ENTRY: Entry = unsafe { Entry::load().unwrap() };
    static ref DEVICE_EXTENSIONS: Vec<&'static CStr> = vec![c"VK_KHR_swapchain", c"VK_KHR_dynamic_rendering", c"VK_EXT_descriptor_indexing", c"VK_EXT_mesh_shader"];
    static ref VALIDATION_LAYERS: Vec<&'static CStr> = vec![c"VK_LAYER_KHRONOS_validation"];
}

#[allow(unused_macros)]

#[cfg(debug_assertions)]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

pub struct BnanDevice {
    pub instance: Instance,
    pub thread_pool: ThreadPool,
    pub command_pools: Vec<vk::CommandPool>,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub msaa_samples: vk::SampleCountFlags,
    pub graphics_queue: vk::Queue,
    pub compute_queue: vk::Queue,
    pub transfer_queue: vk::Queue,
    pub device: Device,
    pub allocator: Allocator,
}

#[derive(Copy, Clone)]
pub struct QueueIndices {
    pub graphics_family: Option<u32>,
    pub graphics_queue_index: u32,
    pub compute_family: Option<u32>,
    pub compute_queue_index: u32,
    pub transfer_family: Option<u32>,
    pub transfer_queue_index: u32,
}

impl QueueIndices {

    fn new() -> QueueIndices {
        QueueIndices {
            graphics_family: None,
            graphics_queue_index: 0,
            compute_family: None,
            compute_queue_index: 0,
            transfer_family: None,
            transfer_queue_index: 0,
        }
    }

    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.compute_family.is_some() && self.transfer_family.is_some()
    }
}

#[derive(Clone)]
pub struct SwapChainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapChainSupportDetails {
    fn is_adequate(&self) -> bool {
        !self.formats.is_empty() && !self.present_modes.is_empty()
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn debug_callback(debug_utils_message_severity_flags_ext: vk::DebugUtilsMessageSeverityFlagsEXT, message_type: vk::DebugUtilsMessageTypeFlagsEXT, callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT, user_data: *mut c_void) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*callback_data).p_message) };

    let stype = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "GENERAL: ",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "VALIDATION: ",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "PERFORMANCE: ",
        _ => "OTHER",
    };

    let (color_begin, color_end) = match debug_utils_message_severity_flags_ext {
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => { ("\x1b[33mWARNING ", "\x1b[0m") },
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => { ("\x1b[31mERROR ", "\x1b[0m") },
        _ => { ("", "") }
    };

    println!("{}{}{}{}", color_begin, stype, message.to_str().unwrap(), color_end);

    vk::FALSE
}

pub struct BnanBarrierBuilder {
    image_barriers: Vec<vk::ImageMemoryBarrier2<'static>>,
}

impl BnanBarrierBuilder {
    pub fn new() -> Self {
        Self {
            image_barriers: Vec::new(),
        }
    }
    
    pub fn transition_image_layout(
        &mut self,
        device: &BnanDevice,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        undefined_exec_stage: Option<vk::PipelineStageFlags2>,
        levels: Option<u32>,
        layers: Option<u32>,
    ) -> Result<&mut Self> {
        let barrier = device.build_image_transition_barrier(
            image, old_layout, new_layout, undefined_exec_stage, levels, layers
        )?;
        self.image_barriers.push(barrier);
        Ok(self)
    }
    
    pub fn transition_image_layout_raw(
        &mut self,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
        levels: Option<u32>,
        layers: Option<u32>,
    ) -> &mut Self {
        let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(levels.unwrap_or(1))
            .base_array_layer(0)
            .layer_count(layers.unwrap_or(1));
        
        let barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(image)
            .subresource_range(subresource_range)
            .src_access_mask(src_access)
            .src_stage_mask(src_stage)
            .dst_access_mask(dst_access)
            .dst_stage_mask(dst_stage)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        
        self.image_barriers.push(barrier);
        self
    }
    
    pub fn flush_writes(
        &mut self,
        image: vk::Image,
        layout: vk::ImageLayout,
        src_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
        levels: Option<u32>,
        layers: Option<u32>,
    ) -> &mut Self {
        let aspect_mask = if layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(levels.unwrap_or(1))
            .base_array_layer(0)
            .layer_count(layers.unwrap_or(1));
        
        let barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(layout)
            .new_layout(layout)  // Same layout - no transition
            .image(image)
            .subresource_range(subresource_range)
            .src_access_mask(src_access)
            .src_stage_mask(src_stage)
            .dst_access_mask(dst_access)
            .dst_stage_mask(dst_stage)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        
        self.image_barriers.push(barrier);
        self
    }
    
    pub fn is_empty(&self) -> bool {
        self.image_barriers.is_empty()
    }
    
    pub fn record_sync(&mut self, device: &BnanDevice, command_buffer: vk::CommandBuffer) {
        if self.image_barriers.is_empty() {
            return;
        }
        
        device.record_image_barriers(command_buffer, &self.image_barriers);
        self.image_barriers.clear();
    }
    
    pub fn record_async(&mut self, device: &BnanDevice) -> Result<()> {
        if self.image_barriers.is_empty() {
            return Ok(());
        }
        
        let command_pool = device.command_pools[BnanDevice::COMPUTE_COMMAND_POOL];
        
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY);
        
        let command_buffer = unsafe { device.device.allocate_command_buffers(&alloc_info)?[0] };
        let begin_info = vk::CommandBufferBeginInfo::default();
        
        unsafe {
            device.device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
            device.device.begin_command_buffer(command_buffer, &begin_info)?;
        }
        
        device.record_image_barriers(command_buffer, &self.image_barriers);
        
        let command_buffers = [command_buffer];
        let submit_info = [
            vk::SubmitInfo::default()
                .command_buffers(&command_buffers)
        ];
        
        unsafe {
            device.device.end_command_buffer(command_buffer)?;
            device.device.queue_submit(device.compute_queue, &submit_info, vk::Fence::null())?;
        }
        
        self.image_barriers.clear();
        Ok(())
    }
}

impl Default for BnanBarrierBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for BnanDevice {
    fn drop(&mut self) {
        for pool in &self.command_pools {
            unsafe { self.device.destroy_command_pool(*pool, None) };
        }
        
        if ENABLE_VALIDATION_LAYERS {
            let debug_utils_instance = ext::debug_utils::Instance::new(&*ENTRY, &self.instance);
            unsafe { debug_utils_instance.destroy_debug_utils_messenger(self.debug_messenger, None); }
        }

        let surface_instance = khr::surface::Instance::new(&*ENTRY, &self.instance);
        unsafe { surface_instance.destroy_surface(self.surface, None) };
    }
}

impl BnanDevice {
    pub fn new(window: ArcMut<BnanWindow>) -> Result<BnanDevice> {
        let mut instance = Self::create_instance()?;
        let thread_pool = Self::create_thread_pool()?;

        let debug_messenger = match ENABLE_VALIDATION_LAYERS {
            true => Self::setup_debug_messenger(&instance)?,
            false => vk::DebugUtilsMessengerEXT::null()
        };

        let mut pwindow = window.lock().expect("failed to unlock window mutex");

        let surface = Self::create_surface(&mut *pwindow, &mut instance)?;
        let physical_device = Self::pick_physical_device(&instance, surface)?;

        let indices = Self::find_queue_families(&instance, surface, physical_device)?;
        let msaa_samples = Self::get_msaa_sample_count(&instance, physical_device)?;

        let device = Self::create_logical_device(&instance, &indices, physical_device)?;
        let allocator = Self::create_allocator(&instance, &device, physical_device)?;

        let graphics_queue = Self::get_graphics_queue(&device, &indices)?;
        let compute_queue = Self::get_compute_queue(&device, &indices)?;
        let transfer_queue = Self::get_transfer_queue(&device, &indices)?;
        
        let command_pools = Self::create_command_pools(&device, &indices)?;

        Ok(
            BnanDevice {
                instance,
                thread_pool,
                command_pools,
                debug_messenger,
                surface,
                physical_device,
                msaa_samples,
                graphics_queue,
                compute_queue,
                transfer_queue,
                device,
                allocator
            }
        )
    }

    pub fn get_physical_device_properties(&self) -> vk::PhysicalDeviceProperties2<'_> {
        let mut properties = vk::PhysicalDeviceProperties2::default();
        unsafe { self.instance.get_physical_device_properties2(self.physical_device, &mut properties) };
        properties
    }
    
    pub fn flush_writes(&self, command_buffer: vk::CommandBuffer, image: vk::Image, layout: vk::ImageLayout, src_access_mask: vk::AccessFlags2, dst_access_mask: vk::AccessFlags2, src_stage_mask: vk::PipelineStageFlags2, dst_stage_mask: vk::PipelineStageFlags2, levels: Option<u32>, layers: Option<u32>) {
        let aspect_mask = match layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
            true => {vk::ImageAspectFlags::DEPTH}
            false => {vk::ImageAspectFlags::COLOR}
        };

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(levels.unwrap_or(1))
            .base_array_layer(0)
            .layer_count(layers.unwrap_or(1));

        let barrier = [
            vk::ImageMemoryBarrier2::default()
                .old_layout(layout)
                .new_layout(layout)
                .image(image)
                .subresource_range(subresource_range)
                .src_access_mask(src_access_mask)
                .src_stage_mask(src_stage_mask)
                .dst_access_mask(dst_access_mask)
                .dst_stage_mask(dst_stage_mask)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        ];

        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(&barrier);

        unsafe {
            self.device.cmd_pipeline_barrier2(command_buffer, &dependency_info);
        }
    }
    
    pub fn transition_image_layout_async(&self, image: vk::Image, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout, undefined_exec_stage: Option<vk::PipelineStageFlags2>, levels: Option<u32>, layers: Option<u32>) -> Result<()> {
        let command_pool = self.command_pools[Self::COMPUTE_COMMAND_POOL];

        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY);

        let command_buffer= unsafe { self.device.allocate_command_buffers(&command_buffer_info)?[0] };
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();

        unsafe {
            self.device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;
        }

        self.transition_image_layout_sync(command_buffer, image, old_layout, new_layout, undefined_exec_stage, levels, layers)?;

        let command_buffers = [command_buffer];
        let submit_info = [
            vk::SubmitInfo::default()
            .command_buffers(&command_buffers)
        ];

        unsafe {
            self.device.end_command_buffer(command_buffer)?;
            self.device.queue_submit(self.compute_queue, &submit_info, vk::Fence::null())?;
        }
        
        Ok(())
    }
    
    pub fn transition_image_layout_sync(&self, command_buffer: vk::CommandBuffer, image: vk::Image, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout, undefined_exec_stage: Option<vk::PipelineStageFlags2>, levels: Option<u32>, layers: Option<u32>) -> Result<()> {
        let barrier = self.build_image_transition_barrier(image, old_layout, new_layout, undefined_exec_stage, levels, layers)?;
        self.record_image_barriers(command_buffer, &[barrier]);
        Ok(())
    }
    
    pub fn build_image_transition_barrier(&self, image: vk::Image, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout, undefined_exec_stage: Option<vk::PipelineStageFlags2>, levels: Option<u32>, layers: Option<u32>) -> Result<vk::ImageMemoryBarrier2<'static>> {

        let aspect_mask = match new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
            true => {vk::ImageAspectFlags::DEPTH}
            false => {vk::ImageAspectFlags::COLOR}
        };

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(levels.unwrap_or(1))
            .base_array_layer(0)
            .layer_count(layers.unwrap_or(1));

        let (src_access_mask, src_stage_mask) = match old_layout {
            vk::ImageLayout::UNDEFINED => (vk::AccessFlags2::NONE, undefined_exec_stage.unwrap_or(vk::PipelineStageFlags2::NONE)),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (vk::AccessFlags2::COLOR_ATTACHMENT_WRITE, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE, vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (vk::AccessFlags2::TRANSFER_READ, vk::PipelineStageFlags2::TRANSFER),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => (vk::AccessFlags2::TRANSFER_WRITE, vk::PipelineStageFlags2::TRANSFER),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (vk::AccessFlags2::SHADER_READ, vk::PipelineStageFlags2::FRAGMENT_SHADER),
            _ => bail!("unsupported layout transition"),
        };

        let (dst_access_mask, dst_stage_mask) = match new_layout {
            vk::ImageLayout::GENERAL => (vk::AccessFlags2::MEMORY_WRITE, vk::PipelineStageFlags2::ALL_COMMANDS),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (vk::AccessFlags2::COLOR_ATTACHMENT_WRITE, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE, vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (vk::AccessFlags2::TRANSFER_READ, vk::PipelineStageFlags2::TRANSFER),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => (vk::AccessFlags2::TRANSFER_WRITE, vk::PipelineStageFlags2::TRANSFER),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (vk::AccessFlags2::SHADER_READ, vk::PipelineStageFlags2::FRAGMENT_SHADER),
            vk::ImageLayout::PRESENT_SRC_KHR => (vk::AccessFlags2::NONE, undefined_exec_stage.unwrap_or(vk::PipelineStageFlags2::NONE)),
            _ => bail!("unsupported layout transition"),
        };

        Ok(vk::ImageMemoryBarrier2::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(image)
            .subresource_range(subresource_range)
            .src_access_mask(src_access_mask)
            .src_stage_mask(src_stage_mask)
            .dst_access_mask(dst_access_mask)
            .dst_stage_mask(dst_stage_mask)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED))
    }
    
    pub fn record_image_barriers(&self, command_buffer: vk::CommandBuffer, barriers: &[vk::ImageMemoryBarrier2]) {
        if barriers.is_empty() {
            return;
        }
        
        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(barriers);

        unsafe {
            self.device.cmd_pipeline_barrier2(command_buffer, &dependency_info);
        }
    }

    pub fn get_swapchain_support(&self) -> Result<SwapChainSupportDetails> {
        Self::query_swapchain_support(&self.instance, self.physical_device, self.surface)
    }

    pub fn get_queue_indices(&self) -> Result<QueueIndices> {
        Self::find_queue_families(&self.instance, self.surface, self.physical_device)
    }

    fn create_instance() -> Result<Instance> {

        unsafe {

            let appinfo = vk::ApplicationInfo::default()
                .application_name(c"BnanR Engine App")
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(c"BnanR Engine")
                .engine_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vk::API_VERSION_1_3);

            let extensions = Self::get_required_instance_extensions()?;

            let mut instance_info = vk::InstanceCreateInfo::default()
                .application_info(&appinfo)
                .enabled_extension_names(extensions.as_slice());

            let layers: Vec<*const c_char>;

            if ENABLE_VALIDATION_LAYERS {
                layers = VALIDATION_LAYERS.iter().map(|layer| layer.as_ptr()).collect::<Vec<_>>();
                instance_info = instance_info.enabled_layer_names(&layers);
            }

            let instance = ENTRY.create_instance(&instance_info, None)?;

            Ok(instance)
        }
    }

    fn get_required_instance_extensions() -> Result<Vec<*const c_char>> {
        unsafe {
            let mut count = 0u32;
            SDL_Vulkan_GetInstanceExtensions(&mut count);
            let mut extensions = Vec::new();
            let sdl_extensions = slice_from_raw_parts(SDL_Vulkan_GetInstanceExtensions(&mut count), count as usize).as_ref().unwrap();
            extensions.extend_from_slice(sdl_extensions);

            if ENABLE_VALIDATION_LAYERS {
                extensions.push(c"VK_EXT_debug_utils".as_ptr());
            }

            Ok(extensions)
        }
    }
    
    fn create_thread_pool() -> Result<ThreadPool> {
        Ok(ThreadPoolBuilder::new().build()?)
    }

    fn setup_debug_messenger(instance: &Instance) -> Result<vk::DebugUtilsMessengerEXT> {

        unsafe {
            let debug_utils_instance = ext::debug_utils::Instance::new(&*ENTRY, &instance);

            let debug_messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE)
                .pfn_user_callback(Some(debug_callback));

            Ok(debug_utils_instance.create_debug_utils_messenger(&debug_messenger_info, None)?)
        }
    }

    fn create_surface(window: &mut BnanWindow, instance: &mut Instance) -> Result<vk::SurfaceKHR> {
        let mut surface = vk::SurfaceKHR::null();
        window.create_window_surface(instance, &mut surface)?;
        Ok(surface)
    }

    fn find_queue_families(instance: &Instance, surface: vk::SurfaceKHR, device: vk::PhysicalDevice) -> Result<QueueIndices> {

        let mut indices = QueueIndices::new();

        let queue_families: Vec<vk::QueueFamilyProperties>;

        unsafe {
            queue_families = instance.get_physical_device_queue_family_properties(device);
        }

        let surface_instance = khr::surface::Instance::new(&*ENTRY, instance);

        // 1. Find graphics queue family (must support presentation)
        let graphics_candidate = queue_families.iter().enumerate().find(|(i, props)| {
            let present_support = unsafe { surface_instance.get_physical_device_surface_support(device, *i as u32, surface) };
            props.queue_flags.contains(vk::QueueFlags::GRAPHICS) && present_support.unwrap_or(false)
        });

        if let Some((g_index, _)) = graphics_candidate {
            indices.graphics_family = Some(g_index as u32);
            indices.graphics_queue_index = 0;
        } else {
            bail!("No compatible graphics queue found");
        }

        let dedicated_compute_candidate = queue_families.iter().enumerate().find(|(i, props)| {
            props.queue_flags.contains(vk::QueueFlags::COMPUTE) && (*i as u32 != indices.graphics_family.unwrap())
        });

        if let Some((c_index, _)) = dedicated_compute_candidate {
            indices.compute_family = Some(c_index as u32);
            indices.compute_queue_index = 0;
        } else {
            let graphics_family_indice = indices.graphics_family.unwrap() as usize;
            if queue_families[graphics_family_indice].queue_flags.contains(vk::QueueFlags::COMPUTE) {
                indices.compute_family = Some(graphics_family_indice as u32);
                if queue_families[graphics_family_indice].queue_count > 1 {
                    indices.compute_queue_index = 1;
                } else {
                    indices.compute_queue_index = 0;
                }
            } else {
                bail!("No compatible compute queue found");
            }
        }
        
        let dedicated_transfer_candidate = queue_families.iter().enumerate().find(|(i, props)| {
            props.queue_flags.contains(vk::QueueFlags::TRANSFER) 
            && !props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && *i as u32 != indices.graphics_family.unwrap()
            && *i as u32 != indices.compute_family.unwrap()
        });

        if let Some((t_index, _)) = dedicated_transfer_candidate {
            indices.transfer_family = Some(t_index as u32);
            indices.transfer_queue_index = 0;
        } else {
            let graphics_family_indice = indices.graphics_family.unwrap() as usize;
            indices.transfer_family = Some(graphics_family_indice as u32);
            
            let used_indices = [indices.graphics_queue_index, indices.compute_queue_index];
            let queue_count = queue_families[graphics_family_indice].queue_count;
            indices.transfer_queue_index = (0..queue_count)
                .find(|i| !used_indices.contains(i))
                .unwrap_or(0);
        }

        Ok(indices)
    }

    fn check_device_extension_support(instance: &Instance, device: vk::PhysicalDevice) -> Result<bool> {

        let available_extensions: Vec<vk::ExtensionProperties>;

        unsafe {
            available_extensions = instance.enumerate_device_extension_properties(device)?;
        }

        let mut required_extensions_set: HashSet<&CStr> = HashSet::from_iter(DEVICE_EXTENSIONS.clone());

        for extension in available_extensions.iter() {
            let extension_name = extension.extension_name_as_c_str()?;
            required_extensions_set.remove(&extension_name);
        }

        Ok(required_extensions_set.is_empty())
    }

    fn query_swapchain_support(instance: &Instance, device: vk::PhysicalDevice, surface: vk::SurfaceKHR) -> Result<SwapChainSupportDetails> {

        let surface_instance = khr::surface::Instance::new(&*ENTRY, instance);

        unsafe {
            let details = SwapChainSupportDetails {
                capabilities: surface_instance.get_physical_device_surface_capabilities(device, surface)?,
                formats: surface_instance.get_physical_device_surface_formats(device, surface)?,
                present_modes: surface_instance.get_physical_device_surface_present_modes(device, surface)?,
            };

            Ok(details)
        }
    }

    fn find_supported_format(instance: &Instance, device: vk::PhysicalDevice, candidates: Vec<vk::Format>, tiling: vk::ImageTiling, features: vk::FormatFeatureFlags) -> Result<vk::Format> {
        for format in candidates.iter() {
            unsafe {
                let props = instance.get_physical_device_format_properties(device, *format);

                match tiling {
                    vk::ImageTiling::LINEAR => {
                        if props.linear_tiling_features & features == features {
                            return Ok(*format)
                        }
                    },

                    vk::ImageTiling::OPTIMAL => {
                        if props.optimal_tiling_features & features == features {
                            return Ok(*format)
                        }
                    }

                    _ => {}
                }
            }
        }

        bail!("failed to find suitable format")
    }

    fn is_device_suitable(instance: &Instance, surface: vk::SurfaceKHR, device: vk::PhysicalDevice) -> Result<bool> {

        let indices = Self::find_queue_families(instance, surface, device)?;

        if !indices.is_complete() {
            return Ok(false);
        }

        if !Self::check_device_extension_support(instance, device)? {
            return Ok(false);
        }

        let details = Self::query_swapchain_support(instance, device, surface)?;
        if !details.is_adequate() {
            return Ok(false);
        }

        let mut features2 = vk::PhysicalDeviceFeatures2::default();
        let mut descriptor_indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
        let mut synchronization2_features = vk::PhysicalDeviceSynchronization2Features::default();
        let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::default();
        let mut mesh_shader_features = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();

        unsafe {
            features2.p_next = std::mem::transmute(&mut descriptor_indexing_features);
            descriptor_indexing_features.p_next = std::mem::transmute(&mut synchronization2_features);
            synchronization2_features.p_next = std::mem::transmute(&mut dynamic_rendering_features);
            dynamic_rendering_features.p_next = std::mem::transmute(&mut mesh_shader_features);

            instance.get_physical_device_features2(device, &mut features2);
        }

        Ok(features2.features.sampler_anisotropy == vk::TRUE &&
            descriptor_indexing_features.descriptor_binding_partially_bound == vk::TRUE &&
            descriptor_indexing_features.runtime_descriptor_array == vk::TRUE &&
            descriptor_indexing_features.descriptor_binding_variable_descriptor_count == vk::TRUE &&
            synchronization2_features.synchronization2 == vk::TRUE &&
            dynamic_rendering_features.dynamic_rendering == vk::TRUE &&
            mesh_shader_features.mesh_shader == vk::TRUE &&
            mesh_shader_features.task_shader == vk::TRUE
        )
    }

    fn pick_physical_device(instance: &Instance, surface: vk::SurfaceKHR) -> Result<vk::PhysicalDevice> {
        let mut physical_device = vk::PhysicalDevice::null();

        unsafe {
            for device in instance.enumerate_physical_devices()? {
                if Self::is_device_suitable(instance, surface, device)? {
                    physical_device = device;
                    break;
                }
            }
        }

        if physical_device == vk::PhysicalDevice::null() {
           bail!("failed to find suitable GPU!");
        }

        unsafe {
            let props = instance.get_physical_device_properties(physical_device);
            println!("Physical device: {}", CStr::from_ptr(props.device_name.as_ptr()).to_str()?);
        }

        Ok(physical_device)
    }

    pub fn find_depth_format(&self) -> Result<vk::Format> {
        let candidates = vec![vk::Format::D32_SFLOAT, vk::Format::D32_SFLOAT_S8_UINT, vk::Format::D24_UNORM_S8_UINT];
        Self::find_supported_format(&self.instance, self.physical_device, candidates, vk::ImageTiling::OPTIMAL, vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
    }

    fn get_msaa_sample_count(instance: &Instance, device: vk::PhysicalDevice) -> Result<vk::SampleCountFlags> {

        let props: vk::PhysicalDeviceProperties;

        unsafe {
            props = instance.get_physical_device_properties(device);
        }

        let counts = props.limits.framebuffer_color_sample_counts & props.limits.framebuffer_depth_sample_counts;
        if counts & vk::SampleCountFlags::TYPE_64 == vk::SampleCountFlags::TYPE_64 { return Ok(vk::SampleCountFlags::TYPE_64); }
        if counts & vk::SampleCountFlags::TYPE_32 == vk::SampleCountFlags::TYPE_32 { return Ok(vk::SampleCountFlags::TYPE_32); }
        if counts & vk::SampleCountFlags::TYPE_16 == vk::SampleCountFlags::TYPE_16 { return Ok(vk::SampleCountFlags::TYPE_16); }
        if counts & vk::SampleCountFlags::TYPE_8 == vk::SampleCountFlags::TYPE_8 { return Ok(vk::SampleCountFlags::TYPE_8); }
        if counts & vk::SampleCountFlags::TYPE_4 == vk::SampleCountFlags::TYPE_4 { return Ok(vk::SampleCountFlags::TYPE_4); }
        if counts & vk::SampleCountFlags::TYPE_2 == vk::SampleCountFlags::TYPE_2 { return Ok(vk::SampleCountFlags::TYPE_2); }

        Ok(vk::SampleCountFlags::TYPE_1)
    }

    fn create_logical_device(instance: &Instance, indices: &QueueIndices, device: vk::PhysicalDevice) -> Result<Device> {

        let mut family_queue_counts = std::collections::HashMap::<u32, u32>::new();
        
        let gfx_family = indices.graphics_family.unwrap();
        let compute_family = indices.compute_family.unwrap();
        let transfer_family = indices.transfer_family.unwrap();
        
        *family_queue_counts.entry(gfx_family).or_insert(0) = 
            (*family_queue_counts.get(&gfx_family).unwrap_or(&0)).max(indices.graphics_queue_index + 1);
        *family_queue_counts.entry(compute_family).or_insert(0) = 
            (*family_queue_counts.get(&compute_family).unwrap_or(&0)).max(indices.compute_queue_index + 1);
        *family_queue_counts.entry(transfer_family).or_insert(0) = 
            (*family_queue_counts.get(&transfer_family).unwrap_or(&0)).max(indices.transfer_queue_index + 1);
        
        let priorities_storage: Vec<Vec<f32>> = family_queue_counts.values().map(|&count| vec![1.0; count as usize]).collect();
        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = family_queue_counts.keys().zip(priorities_storage.iter()).map(|(&family, priorities)| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(priorities)
        }).collect();

        let mut device_features = vk::PhysicalDeviceFeatures2::default();
        device_features.features.sampler_anisotropy = vk::TRUE;

        let mut synchronization2_features = vk::PhysicalDeviceSynchronization2Features::default()
            .synchronization2(true);

        let mut descriptor_indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true);

        let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::default()
            .dynamic_rendering(true);

        let mut mesh_shading_features = vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
            .mesh_shader(true)
            .task_shader(true);

        let extensions: Vec<_> = DEVICE_EXTENSIONS.iter().map(|s| s.as_ptr()).collect();

        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extensions)
            .push_next(&mut device_features)
            .push_next(&mut synchronization2_features)
            .push_next(&mut descriptor_indexing_features)
            .push_next(&mut dynamic_rendering_features)
            .push_next(&mut mesh_shading_features);

        unsafe {
            Ok(instance.create_device(device, &info, None)?)
        }
    }

    fn create_allocator(instance: &Instance, device: &Device, physical_device: vk::PhysicalDevice) -> Result<Allocator> {
        let mut info = AllocatorCreateInfo::new(instance, device, physical_device);
        info.vulkan_api_version = vk::API_VERSION_1_3;

        unsafe { Ok(Allocator::new(info)?) }
    }

    /// Command pool indices
    pub const GRAPHICS_COMMAND_POOL: usize = 0;
    pub const COMPUTE_COMMAND_POOL: usize = 1;
    pub const TRANSFER_COMMAND_POOL: usize = 2;

    fn create_command_pools(device: &Device, indices: &QueueIndices) -> Result<Vec<vk::CommandPool>> {
        let mut pools = Vec::with_capacity(3);

        // Index 0: Graphics command pool
        let graphics_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(indices.graphics_family.unwrap())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        
        unsafe {
            pools.push(device.create_command_pool(&graphics_pool_info, None)?);
        }

        // Index 1: Compute command pool
        let compute_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(indices.compute_family.unwrap())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        unsafe {
            pools.push(device.create_command_pool(&compute_pool_info, None)?);
        }

        // Index 2: Transfer command pool
        let transfer_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(indices.transfer_family.unwrap())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        unsafe {
            pools.push(device.create_command_pool(&transfer_pool_info, None)?);
        }

        Ok(pools)
    }

    fn get_graphics_queue(device: &Device, indices: &QueueIndices) -> Result<vk::Queue> {
        unsafe { Ok(device.get_device_queue(indices.graphics_family.unwrap(), indices.graphics_queue_index)) }
    }

    fn get_compute_queue(device: &Device, indices: &QueueIndices) -> Result<vk::Queue> {
        unsafe { Ok(device.get_device_queue(indices.compute_family.unwrap(), indices.compute_queue_index)) }
    }

    fn get_transfer_queue(device: &Device, indices: &QueueIndices) -> Result<vk::Queue> {
        unsafe { Ok(device.get_device_queue(indices.transfer_family.unwrap(), indices.transfer_queue_index)) }
    }
}