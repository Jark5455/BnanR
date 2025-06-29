use std::ffi::CStr;
use std::ptr;

use anyhow::*;
use ash::*;
use sdl3_sys::everything::*;

use crate::core::{ArcMut, RcMut};

pub trait WindowObserver<T> {
    fn update(&mut self, data: T);
}

pub struct BnanWindow {
    pub width: i32,
    pub height: i32,
    pub window: Box<SDL_Window>,
    
    pub quit_observers: Vec<RcMut<dyn WindowObserver<()>>>,
    pub resize_observers: Vec<RcMut<dyn WindowObserver<(i32, i32)>>>,
    
    pub atomic_quit_observers: Vec<ArcMut<dyn WindowObserver<()>>>,
    pub atomic_resize_observers: Vec<ArcMut<dyn WindowObserver<(i32, i32)>>>,
}

impl Drop for BnanWindow {
    fn drop(&mut self) {
        unsafe {
            SDL_DestroyWindow(self.window.as_mut());
            SDL_Quit();
        }
    }
}

impl BnanWindow {
    pub fn new(w: i32, h: i32) -> Result<BnanWindow> {
    

        let window = unsafe {
            SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
            
            let pwindow = SDL_CreateWindow(c"BnanR".as_ptr(), w, h, SDL_WINDOW_VULKAN | SDL_WINDOW_HIGH_PIXEL_DENSITY);

            if pwindow.is_null() {
                let error = CStr::from_ptr(SDL_GetError()).to_str().unwrap();
                bail!(error);
            }
            
            SDL_SetWindowResizable(pwindow, true);
            SDL_SetWindowFocusable(pwindow, true);
            
            Box::from_raw(pwindow)
        };
        
        Ok(BnanWindow {
            width: w,
            height: h,
            window,
            
            quit_observers: Vec::new(),
            resize_observers: Vec::new(),
            
            atomic_quit_observers: Vec::new(),
            atomic_resize_observers: Vec::new(),
        })
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
    
    pub fn register_quit_observer(&mut self, observer: RcMut<dyn WindowObserver<()>>) {
        self.quit_observers.push(observer);
    }
    
    pub fn register_resize_observer(&mut self, observer: RcMut<dyn WindowObserver<(i32, i32)>>) {
        self.resize_observers.push(observer);
    }
    
    pub fn register_atomic_quit_observer(&mut self, observer: ArcMut<dyn WindowObserver<()>>) {
        self.atomic_quit_observers.push(observer);
    }
    
    pub fn register_atomic_resize_observer(&mut self, observer: ArcMut<dyn WindowObserver<(i32, i32)>>) {
        self.atomic_resize_observers.push(observer);
    }
    
    pub fn clear_observers(&mut self) {
        self.atomic_quit_observers.clear();
        self.atomic_resize_observers.clear();
        self.quit_observers.clear();
        self.resize_observers.clear();
    }
    
    pub fn process_events(&mut self) {

        let mut e = SDL_Event::default();

        unsafe {

            SDL_PollEvent(&mut e);

            match SDL_EventType(e.r#type) {
                SDL_EVENT_QUIT => {
                    for observer in &self.quit_observers {
                        observer.borrow_mut().update(());
                    }

                    for observer in &self.atomic_quit_observers {
                        observer.lock().unwrap().update(());
                    }
                },

                SDL_EVENT_WINDOW_RESIZED => {

                    SDL_GetWindowSize(self.window.as_ref() as *const SDL_Window as *mut SDL_Window, &mut self.width as *mut i32, &mut self.height as *mut i32);

                    for observer in &self.resize_observers {
                        observer.borrow_mut().update((self.width, self.height));
                    }

                    for observer in &self.atomic_resize_observers {
                        observer.lock().unwrap().update((self.width, self.height));
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