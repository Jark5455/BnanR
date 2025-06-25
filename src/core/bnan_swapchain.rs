use anyhow::*;
use ash::*;

use crate::core::ArcMut;
use crate::core::bnan_device::{BnanDevice, SwapChainSupportDetails};

pub struct BnanSwapchain {
    pub device: ArcMut<BnanDevice>,
    pub swapchain: vk::SwapchainKHR,
    pub present_mode: vk::PresentModeKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub image_count: u32,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
}

impl BnanSwapchain {
    pub fn new(device: ArcMut<BnanDevice>) -> Result<BnanSwapchain> {

        let swapchain_support = device.lock().unwrap().get_swapchain_support()?;
        let extent = swapchain_support.capabilities.current_extent;

        if extent.width == u32::MAX {
            bail!("Something went wrong in swapchain creation");
        }

        let surface_format = Self::choose_swap_surface_format(swapchain_support.formats.clone());
        let present_mode = Self::choose_swap_present_mode(swapchain_support.present_modes.clone());
        let image_count = Self::choose_swap_image_count(&swapchain_support);
        
        let swapchain = Self::create_swapchain(device.clone(), extent, surface_format, image_count, present_mode, None)?;
        let images = Self::get_swapchain_images(device.clone(), swapchain)?;
        let image_views = Self::create_swapchain_image_views(device.clone(), images.clone(), surface_format.format)?;

        Ok (
            BnanSwapchain {
                device,
                swapchain,
                present_mode,
                surface_format,
                image_count,
                images,
                image_views,
            }
        )
    }

    pub fn recreate_swapchain(&mut self, extent: vk::Extent2D) -> Result<()> {
        
        {
            let tmp_device = self.device.lock().unwrap();
            unsafe { tmp_device.device.queue_wait_idle(tmp_device.graphics_queue)? };
        }
        
        let old_swapchain = self.swapchain.clone();
        self.swapchain = Self::create_swapchain(self.device.clone(), extent, self.surface_format, self.image_count, self.present_mode, Some(old_swapchain))?;
        
        Ok(())
    }
    
    fn choose_swap_image_count(swapchain_support: &SwapChainSupportDetails) -> u32 {
        let mut image_count = swapchain_support.capabilities.min_image_count + 1;
        
        if swapchain_support.capabilities.max_image_count > 0 && image_count > swapchain_support.capabilities.max_image_count {
            image_count = swapchain_support.capabilities.max_image_count;
        }
        
        image_count
    }

    fn choose_swap_present_mode(available_modes: Vec<vk::PresentModeKHR>) -> vk::PresentModeKHR {
        for mode in &available_modes {
            if *mode == vk::PresentModeKHR::MAILBOX {
                println!("Present Mode: Mailbox");
                return vk::PresentModeKHR::MAILBOX;
            }
        }

        vk::PresentModeKHR::FIFO_RELAXED
    }

    fn choose_swap_surface_format(available_formats: Vec<vk::SurfaceFormatKHR>) -> vk::SurfaceFormatKHR {
        for format in &available_formats {
            if format.format == vk::Format::B8G8R8A8_SRGB && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
                return *format;
            }
        }

        println!("WARNING: No suitable surface format found!, falling back to format at index 0");

        available_formats[0]
    }

    fn create_swapchain(device: ArcMut<BnanDevice>, extent: vk::Extent2D, surface_format: vk::SurfaceFormatKHR, image_count: u32, present_mode: vk::PresentModeKHR, old_swapchain: Option<vk::SwapchainKHR>) -> Result<vk::SwapchainKHR> {

        let device = device.lock().unwrap();

        let swapchain_support = device.get_swapchain_support()?;
        let swapchain_device = khr::swapchain::Device::new(&device.instance, &device.device);

        let mut swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(device.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(swapchain_support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        if old_swapchain.is_some() {
            swapchain_info.old_swapchain = old_swapchain.unwrap();
        }

        unsafe {
            Ok(swapchain_device.create_swapchain(&swapchain_info, None)?)
        }
    }

    fn get_swapchain_images(device: ArcMut<BnanDevice>, swapchain: vk::SwapchainKHR) -> Result<Vec<vk::Image>> {
        let device = device.lock().unwrap();
        let swapchain_device = khr::swapchain::Device::new(&device.instance, &device.device);

        unsafe { Ok(swapchain_device.get_swapchain_images(swapchain)?) }
    }

    fn create_swapchain_image_views(device: ArcMut<BnanDevice>, images: Vec<vk::Image>, format: vk::Format) -> Result<Vec<vk::ImageView>> {
        let device = device.lock().unwrap();

        let image_views: prelude::VkResult<Vec<_>> = images.iter().map(|image| {

            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let imageview_info = vk::ImageViewCreateInfo::default()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(subresource_range);

            unsafe { device.device.create_image_view(&imageview_info, None) }

        }).collect();
        
        Ok(image_views?)
    }
}