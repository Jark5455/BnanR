mod simple_system;

use ash::*;

use BnanR::core::{make_arcmut, make_rcmut};
use BnanR::core::bnan_camera::BnanCamera;
use BnanR::core::bnan_device::BnanDevice;
use BnanR::core::bnan_swapchain::BnanSwapchain;
use BnanR::core::bnan_window::{BnanWindow, WindowObserver};
use BnanR::core::bnan_render_graph::graph::BnanRenderGraph;
use BnanR::core::bnan_render_graph::pass::RenderPass;
use BnanR::core::bnan_render_graph::pass::RenderPassResource;

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

    let render_graph = make_arcmut(BnanRenderGraph::new(window.clone(), device.clone(), swapchain.clone()).unwrap());

    let camera = BnanCamera::new();
    let simple_system = make_rcmut(SimpleSystem::new(device.clone(), render_graph.clone()).unwrap());

    {
        let mut window_guard = window.lock().unwrap();

        window_guard.register_quit_observer(quit.clone());
        window_guard.register_atomic_resize_observer(swapchain.clone());
        window_guard.register_resize_observer(simple_system.clone());
    }

    let backbuffer = render_graph.lock().unwrap().get_backbuffer_handle();

    let depth_handle = simple_system.borrow().depth_handle.clone();
    let color_handle = simple_system.borrow().color_handle.clone();

    let pass = RenderPass::new(
        "Main Render Pass".to_string(),
        vec![],
        vec![
            RenderPassResource {
                handle: depth_handle,
                access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                resolve_target: None,
            },
            RenderPassResource {
                handle: color_handle,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                resolve_target: Some(backbuffer),
            },
        ],

        Box::new(move |_cmd, frame_info| {
            simple_system.borrow_mut().update_uniform_buffers(frame_info, &camera).unwrap();
            simple_system.borrow().draw(frame_info);
        })
    );

    render_graph.lock().unwrap().add_pass(pass);

    while !quit.borrow().quit {
        window.lock().unwrap().process_events();

        match render_graph.lock().unwrap().execute() {
            Ok(_) => {},
            Err(e) => {
                if let Some(vk_res) = e.downcast_ref::<vk::Result>() {
                    if *vk_res == vk::Result::ERROR_OUT_OF_DATE_KHR {
                    render_graph.lock().unwrap().recreate_swapchain().expect("ERROR_OUT_OF_DATE");
                    }
                }
                println!("Render Graph Error: {:?}", e);
            }
        }
    }

    unsafe { device.lock().unwrap().device.device_wait_idle().unwrap() };
    window.lock().unwrap().clear_observers();
}