#![allow(dead_code)]
#![allow(unused_variables)]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

pub mod bnan_device;
pub mod bnan_window;
pub mod bnan_swapchain;
pub mod bnan_rendering;
pub mod bnan_image;
pub mod bnan_descriptors;
pub mod bnan_pipeline;
pub mod bnan_buffer;

pub type RcMut<T> = Rc<RefCell<T>>;
pub type ArcMut<T> = Arc<Mutex<T>>;

pub fn make_rcmut<T>(data: T) -> RcMut<T> {
    Rc::new(RefCell::new(data))
}

pub fn make_arcmut<T>(data: T) -> ArcMut<T> {
    Arc::new(Mutex::new(data))
}