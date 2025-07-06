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
    static ref DEVICE_EXTENSIONS: Vec<&'static CStr> = vec![c"VK_KHR_swapchain", c"VK_KHR_dynamic_rendering", c"VK_EXT_descriptor_indexing"];
}

#[allow(unused_macros)]

#[cfg(debug_assertions)]
const ENABLE_VALIDATION_LAYERS: bool = true;
const VALIDATION_LAYERS: &[*const c_char] = &[c"VK_LAYER_KHRONOS_validation".as_ptr()];
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
    pub device: Device,
    pub allocator: Allocator,
}

#[derive(Copy, Clone)]
pub struct QueueIndices {
    pub graphics_family: Option<u32>,
    pub graphics_queue_index: u32,
    pub compute_family: Option<u32>,
    pub compute_queue_index: u32,
}

impl QueueIndices {

    fn new() -> QueueIndices {
        QueueIndices {
            graphics_family: None,
            graphics_queue_index: 0,
            compute_family: None,
            compute_queue_index: 0,
        }
    }

    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.compute_family.is_some()
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
                device,
                allocator
            }
        )
    }

    pub fn get_physical_device_properties(&self) -> vk::PhysicalDeviceProperties2 {
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
        let command_pool = self.command_pools[2];

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

        let barrier = [
            vk::ImageMemoryBarrier2::default()
                .old_layout(old_layout)
                .new_layout(new_layout)
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

        Ok(())
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

            if  ENABLE_VALIDATION_LAYERS {
                instance_info = instance_info.enabled_layer_names(VALIDATION_LAYERS);
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


        // Attempt to find a dedicated compute family
        let dedicated_compute_candidate = queue_families.iter().enumerate().find(|(i, props)| {
            props.queue_flags.contains(vk::QueueFlags::COMPUTE) && (*i as u32 != indices.graphics_family.unwrap())
        });

        if let Some((c_index, _)) = dedicated_compute_candidate {
            indices.compute_family = Some(c_index as u32);
            indices.compute_queue_index = 0;

            return Ok(indices);
        }

        // Attempt to use same family as graphics family
        let graphics_family_indice = indices.graphics_family.unwrap() as usize;
        if queue_families[graphics_family_indice].queue_flags.contains(vk::QueueFlags::COMPUTE) {
            indices.compute_family = Some(graphics_family_indice as u32);



            // if multiple queues exist, dedicate one to compute, otherwise share one queue
            if queue_families[graphics_family_indice].queue_count > 1 {
                indices.compute_queue_index = 1;
            } else {
                indices.compute_queue_index = 0;
            }

            return Ok(indices);
        }

        // Otherwise, no compute family exists
        bail!("No compatible graphics queue found");
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

        unsafe {
            features2.p_next = std::mem::transmute(&mut descriptor_indexing_features);
            descriptor_indexing_features.p_next = std::mem::transmute(&mut synchronization2_features);
            instance.get_physical_device_features2(device, &mut features2);
        }

        Ok(features2.features.sampler_anisotropy == vk::TRUE && descriptor_indexing_features.descriptor_binding_partially_bound == vk::TRUE && descriptor_indexing_features.runtime_descriptor_array == vk::TRUE && descriptor_indexing_features.descriptor_binding_variable_descriptor_count == vk::TRUE && synchronization2_features.synchronization2 == vk::TRUE)
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

        let queue_create_infos = match indices.graphics_family.unwrap() == indices.compute_family.unwrap() {
            true => match indices.graphics_queue_index == indices.compute_queue_index {
                true => vec![
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(indices.graphics_family.unwrap())
                        .queue_priorities(&[1.0]),
                ],

                false => vec![
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(indices.graphics_family.unwrap())
                        .queue_priorities(&[1.0, 1.0]),
                ],
            }

            false => vec![
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(indices.graphics_family.unwrap())
                    .queue_priorities(&[1.0]),

                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(indices.compute_family.unwrap())
                    .queue_priorities(&[1.0])
            ]
        };

        let mut synchronization2_features = vk::PhysicalDeviceSynchronization2Features::default()
            .synchronization2(true);

        let mut descriptor_indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true);

        let mut device_features = vk::PhysicalDeviceFeatures2::default();
        device_features.features.sampler_anisotropy = vk::TRUE;

        unsafe {
            device_features.p_next = std::mem::transmute(&mut descriptor_indexing_features);
            descriptor_indexing_features.p_next = std::mem::transmute(&mut synchronization2_features);
        }

        let extensions: Vec<_> = DEVICE_EXTENSIONS.iter().map(|s| s.as_ptr()).collect();

        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(queue_create_infos.as_slice())
            .enabled_extension_names(extensions.as_slice())
            .push_next(&mut device_features);

        unsafe {
            Ok(instance.create_device(device, &info, None)?)
        }
    }

    fn create_allocator(instance: &Instance, device: &Device, physical_device: vk::PhysicalDevice) -> Result<Allocator> {
        let mut info = AllocatorCreateInfo::new(instance, device, physical_device);
        info.vulkan_api_version = vk::API_VERSION_1_3;

        unsafe { Ok(Allocator::new(info)?) }
    }

    // constant number of command pools for now, will expand later
    // must be multiple of 2

    // in a perfect world below this would work
    // let num_pools = std::thread::available_parallelism()?.get();
    
    const NUM_POOLS: u32 = 4;

    fn create_command_pools(device: &Device, indices: &QueueIndices) -> Result<Vec<vk::CommandPool>> {

        let num_pools = Self::NUM_POOLS;
        let mut pools = Vec::with_capacity(num_pools as usize);

        // dedicate half of the pools to graphics
        for _ in 0..num_pools / 2 {
            let pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(indices.graphics_family.unwrap())
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            
            unsafe {
                pools.push(device.create_command_pool(&pool_info, None)?);
            }
        }

        // and the other half to compute
        for _ in num_pools / 2..num_pools {
            let pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(indices.compute_family.unwrap())
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            unsafe {
                pools.push(device.create_command_pool(&pool_info, None)?);
            }
        }

        Ok(pools)
    }

    fn get_graphics_queue(device: &Device, indices: &QueueIndices) -> Result<vk::Queue> {
        unsafe { Ok(device.get_device_queue(indices.graphics_family.unwrap(), indices.graphics_queue_index)) }
    }

    fn get_compute_queue(device: &Device, indices: &QueueIndices) -> Result<vk::Queue> {
        unsafe { Ok(device.get_device_queue(indices.compute_family.unwrap(), indices.compute_queue_index)) }
    }
}