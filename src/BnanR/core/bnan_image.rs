
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
    pub mip_levels: u32,
    pub mip_views: Vec<vk::ImageView>,
    pub owned: bool,
}

impl Drop for BnanImage {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                let device = self.device.lock().unwrap();
                
                // Destroy per-mip views
                for view in &self.mip_views {
                    device.device.destroy_image_view(*view, None);
                }
                
                // Destroy main image view
                device.device.destroy_image_view(self.image_view, None);
                
                // Destroy image and allocation
                if let Some(mut alloc) = self.image_allocation.take() {
                    drop(device); // Release lock before reallocating
                    self.device.lock().unwrap().allocator.destroy_image(self.image, &mut alloc);
                }
            }
        }
    }
}

impl BnanImage {
    pub fn new(
        device: ArcMut<BnanDevice>, 
        format: vk::Format, 
        usage: vk::ImageUsageFlags, 
        properties: vk::MemoryPropertyFlags, 
        image_extent: vk::Extent3D, 
        sample_count: vk::SampleCountFlags,
        mip_levels: Option<u32>,
    ) -> Result<BnanImage> {
        let mip_count = mip_levels.unwrap_or(1);
        let (image, image_allocation) = Self::create_image(device.clone(), format, usage, properties, image_extent, sample_count, mip_count)?;

        let image_aspect = match format {
            vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
            vk::Format::D32_SFLOAT_S8_UINT => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            vk::Format::D24_UNORM_S8_UINT => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,

            _ => vk::ImageAspectFlags::COLOR,
        };

        let image_view = Self::create_image_view(device.clone(), image, format, image_aspect, mip_count)?;
        
        Ok (BnanImage {
            device,
            image,
            image_view,
            image_allocation: Some(image_allocation),
            image_extent,
            format,
            mip_levels: mip_count,
            mip_views: Vec::new(),
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
            mip_levels: 1,
            mip_views: Vec::new(),
            owned: false
        }
    }
    
    /// Calculate the number of mip levels for a given extent
    pub fn calculate_mip_levels(extent: vk::Extent3D) -> u32 {
        let max_dim = extent.width.max(extent.height) as f32;
        (max_dim.log2().floor() as u32) + 1
    }
    
    /// Get the extent at a specific mip level
    pub fn mip_extent(&self, mip_level: u32) -> vk::Extent3D {
        vk::Extent3D {
            width: (self.image_extent.width >> mip_level).max(1),
            height: (self.image_extent.height >> mip_level).max(1),
            depth: 1,
        }
    }
    
    /// Create an image view for a specific mip level and store it internally.
    /// Returns the index of the created view in the mip_views vector.
    pub fn create_mip_view(&mut self, mip_level: u32) -> Result<usize> {
        let image_aspect = match self.format {
            vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
            vk::Format::D32_SFLOAT_S8_UINT => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            vk::Format::D24_UNORM_S8_UINT => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            _ => vk::ImageAspectFlags::COLOR,
        };
        
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(image_aspect)
            .base_array_layer(0)
            .layer_count(1)
            .base_mip_level(mip_level)
            .level_count(1);
        
        let info = vk::ImageViewCreateInfo::default()
            .image(self.image)
            .format(self.format)
            .subresource_range(subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);
        
        let view = unsafe { self.device.lock().unwrap().device.create_image_view(&info, None)? };
        
        let index = self.mip_views.len();
        self.mip_views.push(view);
        Ok(index)
    }

    fn create_image(device: ArcMut<BnanDevice>, format: vk::Format, usage: vk::ImageUsageFlags, properties: vk::MemoryPropertyFlags, extent: vk::Extent3D, sample_count: vk::SampleCountFlags, mip_levels: u32) -> Result<(vk::Image, Allocation)> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(sample_count)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage);

        let allocation_info = AllocationCreateInfo {
            required_flags: properties,
            usage: MemoryUsage::Auto,
            ..Default::default()
        };

        unsafe { Ok(device.lock().unwrap().allocator.create_image(&info, &allocation_info)?) }
    }

    fn create_image_view(device: ArcMut<BnanDevice>, image: vk::Image, format: vk::Format, aspect: vk::ImageAspectFlags, mip_levels: u32) -> Result<vk::ImageView> {
        
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect)
            .base_array_layer(0)
            .layer_count(1)
            .base_mip_level(0)
            .level_count(mip_levels);
        
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .format(format)
            .subresource_range(subresource_range)
            .view_type(vk::ImageViewType::TYPE_2D);
        
        unsafe { Ok(device.lock().unwrap().device.create_image_view(&info, None)?) }
    }
}