use std::ffi::CStr;
use std::ptr;

use anyhow::*;
use ash::*;
use sdl3_sys::everything::*;

type WindowQuitCallback<'a> = &'a mut dyn FnMut();
type WindowResizeCallback<'a> = &'a mut dyn FnMut(i32, i32);

pub struct BnanWindow<'a> {
    pub width: i32,
    pub height: i32,

    pub window: Box<SDL_Window>,

    pub quit_callbacks: Vec<WindowQuitCallback<'a>>,
    pub resize_callbacks: Vec<WindowResizeCallback<'a>>,
}

impl Drop for BnanWindow<'_> {
    fn drop(&mut self) {
        unsafe {
            SDL_DestroyWindow(self.window.as_mut());
            SDL_Quit();
        }
    }
}

impl<'a> BnanWindow<'a> {
    pub fn new(w: i32, h: i32) -> BnanWindow<'a> {
    
        let window: Box<SDL_Window>;
        
        unsafe {
            SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
            window = Box::from_raw(SDL_CreateWindow(c"BnanR".as_ptr(), w, h, SDL_WINDOW_VULKAN | SDL_WINDOW_HIGH_PIXEL_DENSITY));
        }
        
        BnanWindow {
            width: w,
            height: h,
            window,
            quit_callbacks: Vec::new(),
            resize_callbacks: Vec::new(),
        }
    }

    pub fn create_window_surface(&mut self, instance: &mut Instance, surface: &mut vk::SurfaceKHR) -> Result<()> {
        unsafe {
            if SDL_Vulkan_CreateSurface(self.window.as_mut(), std::mem::transmute(instance.handle()), ptr::null(), std::mem::transmute(surface as *mut vk::SurfaceKHR)) != true {
                let error = CStr::from_ptr(SDL_GetError());
                bail!(format!("failed to create window surface!: {}", error.to_str()?));
            }
        }

        Ok(())
    }

    pub fn get_window_extent(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width as u32,
            height: self.height as u32,
        }
    }
    
    pub fn register_quit_callback(&mut self, callback: WindowQuitCallback<'a>) {
        self.quit_callbacks.push(callback);
    }
    
    pub fn register_resize_callback(&mut self, callback: WindowResizeCallback<'a>) {
        self.resize_callbacks.push(callback);
    }
    
    pub fn process_events(&mut self) {

        let mut e = SDL_Event::default();

        unsafe {

            SDL_PollEvent(&mut e);

            match SDL_EventType(e.r#type) {
                SDL_EVENT_QUIT => {
                    for callback in &mut self.quit_callbacks {
                        callback();
                    }
                },

                SDL_EVENT_WINDOW_RESIZED => {

                    SDL_GetWindowSize(self.window.as_ref() as *const SDL_Window as *mut SDL_Window, &mut self.width as *mut i32, &mut self.height as *mut i32);

                    for callback in &mut self.resize_callbacks {
                        callback(self.width, self.height);
                    }
                },

                SDL_EVENT_WINDOW_MINIMIZED => {

                },

                SDL_EVENT_WINDOW_RESTORED => {

                }

                _ => {},
            }
        }
    }
}