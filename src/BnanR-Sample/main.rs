mod simple_system;

use ash::*;

use BnanR::core::{make_arcmut, make_rcmut};
use BnanR::core::bnan_device::BnanDevice;
use BnanR::core::bnan_swapchain::BnanSwapchain;
use BnanR::core::bnan_window::{BnanWindow, WindowObserver};
use BnanR::core::bnan_rendering::BnanRenderHelper;

use crate::simple_system::SimpleSystem;

struct Quit {
    pub quit: bool,
}

impl WindowObserver<()> for Quit {
    fn update(&mut self, _data: ()) {
        self.quit = true
    }
}

fn main() {

    let window = make_arcmut(BnanWindow::new(800, 600).unwrap());
    
    let quit = make_rcmut(Quit {quit: false});
    let initial_window_extent = window.lock().unwrap().get_window_extent();
    
    let device = make_arcmut(BnanDevice::new(window.clone()).unwrap());
    let swapchain = make_arcmut(BnanSwapchain::new(device.clone(), Some(initial_window_extent), None).unwrap());
    let render_helper = make_arcmut(BnanRenderHelper::new(window.clone(), device.clone(), swapchain.clone()).unwrap());

    let simple_system = make_arcmut(SimpleSystem::new(device.clone(), initial_window_extent).unwrap());
    simple_system.lock().unwrap().update_storage_buffers().unwrap();

    window.lock().unwrap().register_quit_observer(quit.clone());
    window.lock().unwrap().register_atomic_resize_observer(swapchain.clone());
    window.lock().unwrap().register_atomic_resize_observer(simple_system.clone());

    while !quit.borrow().quit {
        window.lock().unwrap().process_events();

        let command_buffer = render_helper.lock().unwrap().begin_frame().unwrap();
        let frame_info = render_helper.lock().unwrap().get_current_frame();

        simple_system.lock().unwrap().update_uniform_buffers(&frame_info).unwrap();
        simple_system.lock().unwrap().draw(&frame_info);
        
        if command_buffer != vk::CommandBuffer::null() {
            render_helper.lock().unwrap().end_frame(command_buffer).unwrap();
        }
    }
    
    let q = device.lock().unwrap().graphics_queue;
    unsafe { device.lock().unwrap().device.queue_wait_idle(q).unwrap() };
    window.lock().unwrap().clear_observers();
}
