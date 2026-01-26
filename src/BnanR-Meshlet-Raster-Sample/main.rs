mod meshlet_system;
mod downsample_system;

use ash::*;
use cgmath::num_traits::FloatConst;
use cgmath::Vector3;

use BnanR::core::{make_arcmut, make_rcmut};
use BnanR::core::bnan_camera::BnanCamera;
use BnanR::core::bnan_device::BnanDevice;
use BnanR::core::bnan_swapchain::BnanSwapchain;
use BnanR::core::bnan_window::{BnanWindow, WindowObserver};
use BnanR::core::bnan_render_graph::graph::BnanRenderGraph;
use BnanR::core::bnan_render_graph::pass::RenderPass;
use BnanR::core::bnan_render_graph::pass::RenderPassResource;
use crate::downsample_system::DownsampleSystem;
use crate::meshlet_system::MeshletSystem;

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

    let meshlet_system = make_rcmut(MeshletSystem::new(device.clone(), render_graph.clone()).unwrap());
    meshlet_system.borrow_mut().load_meshlet_mesh("./build/assets2.bpk", "assets/ceramic_vase_01_4k.blend").unwrap();
    meshlet_system.borrow_mut().update_meshlet_data().unwrap();

    let backbuffer = render_graph.lock().unwrap().get_backbuffer_handle();

    let depth_handle = meshlet_system.borrow().depth_handle.clone();
    let color_handle = meshlet_system.borrow().color_handle.clone();
    let resolved_depth_handle = meshlet_system.borrow().resolved_depth_handle.clone();

    let downsample_system = make_rcmut(DownsampleSystem::new(device.clone(), render_graph.clone(), resolved_depth_handle.clone()).unwrap());
    let hiz_handle = downsample_system.borrow().hi_z_handle.clone();
    downsample_system.borrow_mut().register_observer(meshlet_system.clone());

    {
        let mut window_guard = window.lock().unwrap();

        window_guard.register_quit_observer(quit.clone());
        window_guard.register_atomic_resize_observer(swapchain.clone());
        window_guard.register_resize_observer(meshlet_system.clone());
        window_guard.register_mouse_observer(camera.clone());
        window_guard.register_keyboard_observer(camera.clone());
        window_guard.register_resize_observer(camera.clone());
    }

    let main_pass = RenderPass::new(
        "Main Render Pass".to_string(),
        vec![
            RenderPassResource {
                handle: hiz_handle.clone(),
                stage: vk::PipelineStageFlags2::TASK_SHADER_EXT,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                resolve_target: None,
                use_previous_frame: true,
            },
        ],
        vec![
            RenderPassResource {
                handle: depth_handle,
                stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                resolve_target: Some(resolved_depth_handle.clone()),
                use_previous_frame: false,
            },
            RenderPassResource {
                handle: color_handle,
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                resolve_target: Some(backbuffer),
                use_previous_frame: false,
            },
        ],
        
        Box::new(move |_graph, frame_info| {
            camera.borrow_mut().move_in_xz(frame_info.frame_time / 1000.0);
            meshlet_system.borrow_mut().update_uniform_buffers(frame_info, camera.clone()).unwrap();
            meshlet_system.borrow().draw(frame_info);
        })
    );



    let downsample_pass = RenderPass::new(
        "Downsampling Compute Pass".to_string(),
        vec![
            RenderPassResource {
                handle: resolved_depth_handle,
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                layout: vk::ImageLayout::GENERAL,
                resolve_target: None,
                use_previous_frame: false,
            }
        ],

        vec![
            RenderPassResource {
                handle: hiz_handle,
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                layout: vk::ImageLayout::GENERAL,
                resolve_target: None,
                use_previous_frame: false,
            }
        ],

        Box::new(move |graph, frame_info| {
            downsample_system.borrow().dispatch(graph, frame_info);
        })
    );

    {
        let mut render_graph_guard = render_graph.lock().unwrap();
        render_graph_guard.add_pass(main_pass);
        render_graph_guard.add_pass(downsample_pass);
    }

    let mut count = 0;

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