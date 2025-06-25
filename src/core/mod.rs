#![allow(dead_code)]
#![allow(unused_variables)]

use std::cell::RefCell;
use std::ops::{Deref};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

pub(crate) mod bnan_device;
pub(crate) mod bnan_window;
pub(crate) mod bnan_swapchain;
pub(crate) mod bnan_rendering;

pub(crate) struct RcMut<T>(pub Rc<RefCell<T>>);

impl<T> RcMut<T> {
    pub fn new(data: T) -> Self {
        RcMut(Rc::new(RefCell::new(data)))
    }
}

impl<T> Deref for RcMut<T> {
    type Target = Rc<RefCell<T>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> Clone for RcMut<T> {
    fn clone(&self) -> Self {
        RcMut(self.0.clone())
    }
}

pub(crate) struct ArcMut<T>(pub Arc<Mutex<T>>);
impl<T> ArcMut<T> {
    pub fn new(data: T) -> ArcMut<T> {
        ArcMut(Arc::new(Mutex::new(data)))
    }
}

impl<T> Deref for ArcMut<T> {
    type Target = Arc<Mutex<T>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> Clone for ArcMut<T> {
    fn clone(&self) -> Self {
        ArcMut(self.0.clone())
    }
}