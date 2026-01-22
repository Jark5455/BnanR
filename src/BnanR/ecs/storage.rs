use std::any::Any;

use crate::ecs::component::Component;
use crate::stl::sparse_set::SparseSet;

pub trait AnyStorage: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn remove(&mut self, index: usize);
}

pub struct ComponentStorage<T: Component> {
    data: SparseSet<T>,
}

impl<T: Component> ComponentStorage<T> {
    pub fn new() -> Self {
        Self {
            data: SparseSet::new(),
        }
    }

    pub fn insert(&mut self, index: usize, component: T) -> Option<T> {
        self.data.insert(index, component)
    }

    pub fn remove(&mut self, index: usize) -> Option<T> {
        self.data.remove(index)
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if self.data.contains(index) {
            Some(&self.data[index])
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if self.data.contains(index) {
            Some(&mut self.data[index])
        } else {
            None
        }
    }

    pub fn contains(&self, index: usize) -> bool {
        self.data.contains(index)
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }
}

impl<T: Component> Default for ComponentStorage<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Component + 'static> AnyStorage for ComponentStorage<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn remove(&mut self, index: usize) {
        self.data.remove(index);
    }
}
