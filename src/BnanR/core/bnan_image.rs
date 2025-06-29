
use anyhow::*;
use ash::*;
use vk_mem::*;

use crate::core::ArcMut;
use crate::core::bnan_device::BnanDevice;

pub struct BnanImage {
    pub device: ArcMut<BnanDevice>,
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub image_allocation: Allocation,
    pub image_extent: vk::Extent3D,
    pub format: vk::Format,
}

impl Drop for BnanImage {
    fn drop(&mut self) {
        unsafe {
            self.device.lock().unwrap().device.destroy_image_view(self.image_view, None);
            self.device.lock().unwrap().allocator.destroy_image(self.image, &mut self.image_allocation);
        }
    }
}

impl BnanImage {
    pub fn new(device: ArcMut<BnanDevice>, format: vk::Format, usage: vk::ImageUsageFlags, image_extent: vk::Extent3D) -> Result<BnanImage> {
        let (image, image_allocation) = Self::create_image(device.clone(), format, usage, image_extent)?;
        let image_view = Self::create_image_view(device.clone(), image, format, vk::ImageAspectFlags::COLOR)?;
        
        Ok (BnanImage {
            device,
            image,
            image_view,
            image_allocation,
            image_extent,
            format
        })
    }

    fn create_image(device: ArcMut<BnanDevice>, format: vk::Format, usage: vk::ImageUsageFlags, extent: vk::Extent3D) -> Result<(vk::Image, Allocation)> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage);

        let allocation_info = AllocationCreateInfo {
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            usage: MemoryUsage::Auto,
            ..Default::default()
        };
        
        unsafe { Ok(device.lock().unwrap().allocator.create_image(&info, &allocation_info)?) }
    }

    fn create_image_view(device: ArcMut<BnanDevice>, image: vk::Image, format: vk::Format, aspect: vk::ImageAspectFlags) -> Result<vk::ImageView> {
        
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect)
            .base_array_layer(0)
            .layer_count(1)
            .base_mip_level(0)
            .level_count(1);
        
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .format(format)
            .subresource_range(subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);
        
        unsafe { Ok(device.lock().unwrap().device.create_image_view(&info, None)?) }
    }
}