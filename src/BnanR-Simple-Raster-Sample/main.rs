mod simple_system;

use ash::*;
use cgmath::num_traits::FloatConst;
use cgmath::Vector3;
use BnanR::core::{make_arcmut, make_rcmut};
use BnanR::core::bnan_camera::BnanCamera;
use BnanR::core::bnan_device::BnanDevice;
use BnanR::core::bnan_swapchain::BnanSwapchain;
use BnanR::core::bnan_window::{BnanWindow, WindowObserver};
use BnanR::core::bnan_render_graph::graph::BnanRenderGraph;
use BnanR::core::bnan_render_graph::pass::{RenderPass, RenderPassResource};
use BnanR::core::bnan_render_graph::resource::ResourceUsage;
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

    let camera = make_rcmut(BnanCamera::new());

    let aspect = window.lock().unwrap().get_aspect_ratio();
    camera.borrow_mut().set_perspective_projection(f32::PI() / 2.0, aspect, 0.1, 10.0);
    camera.borrow_mut().set_view(Vector3 {x: 0.0, y: 0.0, z: -2.0}, Vector3 {x: 0.0, y: 0.0, z: 0.0});

    let system = make_rcmut(SimpleSystem::new(device.clone(), render_graph.clone()).unwrap());

    {
        let mut window_guard = window.lock().unwrap();

        window_guard.register_quit_observer(quit.clone());
        window_guard.register_atomic_resize_observer(swapchain.clone());
        window_guard.register_resize_observer(system.clone());
        window_guard.register_mouse_observer(camera.clone());
        window_guard.register_keyboard_observer(camera.clone());
        window_guard.register_resize_observer(camera.clone());
    }

    let backbuffer = render_graph.lock().unwrap().get_backbuffer_handle();

    let depth_handle = system.borrow().depth_handle.clone();
    let color_handle = system.borrow().color_handle.clone();

    let pass = RenderPass::new(
        "Main Render Pass".to_string(),
        vec![],
        vec![
            RenderPassResource::new(depth_handle, ResourceUsage::DepthStencilAttachment),
            RenderPassResource::with_resolve(color_handle, ResourceUsage::ColorAttachment, backbuffer),
        ],

        Box::new(move |_graph, frame_info| {
            camera.borrow_mut().move_in_xz(frame_info.frame_time / 1000.0);
            system.borrow_mut().update_uniform_buffers(frame_info, camera.clone()).unwrap();
            system.borrow().draw(frame_info);
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