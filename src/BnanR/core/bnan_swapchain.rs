use anyhow::*;
use ash::*;

use crate::core::ArcMut;
use crate::core::bnan_device::{BnanDevice, SwapChainSupportDetails};
use crate::core::bnan_window::{WindowObserver};

pub struct BnanSwapchain {
    pub device: ArcMut<BnanDevice>,
    pub loader: khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub present_mode: vk::PresentModeKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub image_count: u32,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub extent: vk::Extent2D,
}

impl Drop for BnanSwapchain {
    fn drop(&mut self) {

        let device = self.device.lock().unwrap();
        
        for image_view in &self.image_views {
            unsafe { device.device.destroy_image_view(*image_view, None); }
        }
        
        unsafe {
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}

impl WindowObserver<(i32, i32)> for BnanSwapchain {
    fn update(&mut self, data: (i32, i32)) {
        
        let extent = vk::Extent2D::default()
            .width(data.0 as u32)
            .height(data.1 as u32);

        unsafe {
            let device = self.device.lock().unwrap();
            device.device.queue_wait_idle(device.graphics_queue).expect("waited too long on resize");
        }

        self.recreate_swapchain(extent).unwrap();
    }
}

impl BnanSwapchain {
    pub fn new(device: ArcMut<BnanDevice>, extent: Option<vk::Extent2D>, old_swapchain: Option<vk::SwapchainKHR>) -> Result<BnanSwapchain> {

        let swapchain_support = device.lock().unwrap().get_swapchain_support()?;
        
        let default_extent = vk::Extent2D::default()
            .height(800)
            .width(600);
        
        let extent = extent.unwrap_or(default_extent);
        
        let loader = Self::create_loader(device.clone());
        let surface_format = Self::choose_swap_surface_format(swapchain_support.formats.clone());
        let present_mode = Self::choose_swap_present_mode(swapchain_support.present_modes.clone());
        let image_count = Self::choose_swap_image_count(&swapchain_support);
        
        let swapchain = Self::create_swapchain(device.clone(), &loader, extent, surface_format, image_count, present_mode, old_swapchain)?;
        let images = Self::get_swapchain_images(device.clone(), &loader, swapchain)?;
        let image_views = Self::create_swapchain_image_views(device.clone(), images.clone(), surface_format.format)?;

        Ok (
            BnanSwapchain {
                device,
                loader,
                swapchain,
                present_mode,
                surface_format,
                image_count,
                images,
                image_views,
                extent,
            }
        )
    }

    pub fn recreate_swapchain(&mut self, extent: vk::Extent2D) -> Result<()> {
        let old_swapchain = self.swapchain.clone();
        let new_swapchain = BnanSwapchain::new(self.device.clone(), Some(extent), Some(old_swapchain))?;
        let _ = std::mem::replace(self, new_swapchain);
        Ok(())
    }
    
    fn create_loader(device: ArcMut<BnanDevice>) -> khr::swapchain::Device {
        let device = device.lock().unwrap();
        khr::swapchain::Device::new(&device.instance, &device.device)
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

    fn create_swapchain(device: ArcMut<BnanDevice>, loader: &khr::swapchain::Device, extent: vk::Extent2D, surface_format: vk::SurfaceFormatKHR, image_count: u32, present_mode: vk::PresentModeKHR, old_swapchain: Option<vk::SwapchainKHR>) -> Result<vk::SwapchainKHR> {
        
        let swapchain_support = device.lock().unwrap().get_swapchain_support()?;

        let mut swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(device.lock().unwrap().surface)
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
            Ok(loader.create_swapchain(&swapchain_info, None)?)
        }
    }

    fn get_swapchain_images(device: ArcMut<BnanDevice>, loader: &khr::swapchain::Device, swapchain: vk::SwapchainKHR) -> Result<Vec<vk::Image>> {
        unsafe { Ok(loader.get_swapchain_images(swapchain)?) }
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