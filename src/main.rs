use ash::*;

use crate::core::{ArcMut, RcMut};
use crate::core::bnan_device::BnanDevice;
use crate::core::bnan_swapchain::BnanSwapchain;
use crate::core::bnan_window::BnanWindow;
use crate::core::bnan_rendering::{BnanRenderHelper};

mod core;
mod stl;

fn main() {
    let b_quit: RcMut<bool>;
    let swapchain: ArcMut<BnanSwapchain>;

    let mut quit_callback: Box<dyn FnMut()>;
    let mut resize_callback: Box<dyn FnMut(i32, i32)>;

    let window = ArcMut::new(BnanWindow::new(800, 600));
    let device = ArcMut::new(BnanDevice::new(window.clone()).unwrap());
    
    swapchain = ArcMut::new(BnanSwapchain::new(device.clone()).unwrap());
    
    let render_helper = ArcMut::new(BnanRenderHelper::new(window.clone(), device.clone(), swapchain.clone()).unwrap());

    b_quit = RcMut::new(false);
    quit_callback = Box::new(|| *b_quit.borrow_mut() = true);
    resize_callback = Box::new(|width, height| swapchain.lock().unwrap().recreate_swapchain(vk::Extent2D::default().width(width as u32).height(height as u32)).unwrap());

    window.lock().unwrap().register_quit_callback(quit_callback.as_mut());
    window.lock().unwrap().register_resize_callback(resize_callback.as_mut());
    
    let mut i = 0;
    
    while !*b_quit.borrow() {
        window.lock().unwrap().process_events();
        
        let command_buffer = render_helper.lock().unwrap().begin_frame().unwrap();
        
        if command_buffer != vk::CommandBuffer::null() {
            render_helper.lock().unwrap().end_frame(command_buffer).unwrap();
        }
        
        i += 1;
        
        if i == 10 {
            panic!("e");
        }
    }
}
