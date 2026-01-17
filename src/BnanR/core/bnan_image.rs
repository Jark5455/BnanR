
use anyhow::*;
use ash::*;
use vk_mem::*;

use crate::core::ArcMut;
use crate::core::bnan_device::BnanDevice;

pub struct BnanImage {
    pub device: ArcMut<BnanDevice>,
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub image_allocation: Option<Allocation>,
    pub image_extent: vk::Extent3D,
    pub format: vk::Format,
    pub owned: bool,
}

impl Drop for BnanImage {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                self.device.lock().unwrap().device.destroy_image_view(self.image_view, None);
                if let Some(mut alloc) = self.image_allocation.take() {
                    self.device.lock().unwrap().allocator.destroy_image(self.image, &mut alloc);
                }
            }
        }
    }
}

impl BnanImage {
    pub fn new(device: ArcMut<BnanDevice>, format: vk::Format, usage: vk::ImageUsageFlags, image_extent: vk::Extent3D, sample_count: vk::SampleCountFlags) -> Result<BnanImage> {
        let (image, image_allocation) = Self::create_image(device.clone(), format, usage, image_extent, sample_count)?;

        let image_aspect = match format {
            vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
            vk::Format::D32_SFLOAT_S8_UINT => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            vk::Format::D24_UNORM_S8_UINT => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,

            _ => vk::ImageAspectFlags::COLOR,
        };

        let image_view = Self::create_image_view(device.clone(), image, format, image_aspect)?;
        
        Ok (BnanImage {
            device,
            image,
            image_view,
            image_allocation: Some(image_allocation),
            image_extent,
            format,
            owned: true,
        })
    }

    pub fn from_image(device: ArcMut<BnanDevice>, image: vk::Image, image_view: vk::ImageView, format: vk::Format, image_extent: vk::Extent3D) -> BnanImage {
        BnanImage {
            device,
            image,
            image_view,
            image_allocation: None,
            image_extent,
            format,
            owned: false
        }
    }

    fn create_image(device: ArcMut<BnanDevice>, format: vk::Format, usage: vk::ImageUsageFlags, extent: vk::Extent3D, sample_count: vk::SampleCountFlags) -> Result<(vk::Image, Allocation)> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(sample_count)
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