mod simple_system;

use ash::*;

use BnanR::core::{make_arcmut, make_rcmut};
use BnanR::core::bnan_device::BnanDevice;
use BnanR::core::bnan_swapchain::BnanSwapchain;
use BnanR::core::bnan_window::{BnanWindow, WindowObserver};
use BnanR::core::bnan_render_graph::graph::BnanRenderGraph;
use BnanR::core::bnan_render_graph::pass::{RenderPass, RenderPassResource};

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
    
    let mut render_graph = BnanRenderGraph::new(window.clone(), device.clone(), swapchain.clone()).unwrap();
    
    let mut simple_system = make_rcmut(SimpleSystem::new(device.clone(), initial_window_extent).unwrap());
    simple_system.borrow_mut().update_storage_buffers().unwrap();

    window.lock().unwrap().register_quit_observer(quit.clone());
    window.lock().unwrap().register_atomic_resize_observer(swapchain.clone());
    window.lock().unwrap().register_resize_observer(simple_system.clone());
    
    let backbuffer_handle = render_graph.get_backbuffer_handle();
    
    let pass = RenderPass::new(
        "Raytrace Pass".to_string(),
        vec![],
        vec![
            RenderPassResource {
                handle: backbuffer_handle,
                stage: vk::PipelineStageFlags2::TRANSFER,
                layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                resolve_target: None,
            }
        ],
        Box::new(move |_cmd, frame_info| {
            simple_system.borrow_mut().update_uniform_buffers(frame_info).unwrap();
            simple_system.borrow().draw(frame_info);
        })
    );
    
    render_graph.add_pass(pass);

    while !quit.borrow().quit {
        window.lock().unwrap().process_events();

        match render_graph.execute() {
            Ok(_) => {},
            Err(e) => {
                 if let Some(vk_res) = e.downcast_ref::<vk::Result>() {
                      if *vk_res == vk::Result::ERROR_OUT_OF_DATE_KHR {
                      render_graph.recreate_swapchain().unwrap();
                      }
                 }
                 println!("Render Graph Error: {:?}", e);
            }
        }
    }
    
    let q = device.lock().unwrap().graphics_queue;
    unsafe { device.lock().unwrap().device.queue_wait_idle(q).unwrap() };
    window.lock().unwrap().clear_observers();
}
