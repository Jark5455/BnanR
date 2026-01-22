use std::any::TypeId;
use std::collections::HashMap;

use crate::ecs::component::Component;
use crate::ecs::entity::{Entity, EntityAllocator};
use crate::ecs::storage::{AnyStorage, ComponentStorage};

pub struct World {
    entities: EntityAllocator,
    storages: HashMap<TypeId, Box<dyn AnyStorage>>,
}

impl World {
    pub fn new() -> Self {
        Self {
            entities: EntityAllocator::new(),
            storages: HashMap::new(),
        }
    }

    pub fn spawn(&mut self) -> Entity {
        self.entities.allocate()
    }

    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }

        let index = entity.index();
        for storage in self.storages.values_mut() {
            storage.remove(index);
        }

        self.entities.deallocate(entity)
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) -> Option<T> {
        if !self.entities.is_alive(entity) {
            return None;
        }

        let storage = self.get_or_create_storage::<T>();
        storage.insert(entity.index(), component)
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        if !self.entities.is_alive(entity) {
            return None;
        }

        self.get_storage_mut::<T>()
            .and_then(|storage| storage.remove(entity.index()))
    }

    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        if !self.entities.is_alive(entity) {
            return None;
        }

        self.get_storage::<T>()
            .and_then(|storage| storage.get(entity.index()))
    }

    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        if !self.entities.is_alive(entity) {
            return None;
        }

        self.get_storage_mut::<T>()
            .and_then(|storage| storage.get_mut(entity.index()))
    }

    pub fn has<T: Component>(&self, entity: Entity) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }

        self.get_storage::<T>()
            .map(|storage| storage.contains(entity.index()))
            .unwrap_or(false)
    }

    fn get_or_create_storage<T: Component>(&mut self) -> &mut ComponentStorage<T> {
        let type_id = TypeId::of::<T>();

        if !self.storages.contains_key(&type_id) {
            self.storages
                .insert(type_id, Box::new(ComponentStorage::<T>::new()));
        }

        self.storages
            .get_mut(&type_id)
            .unwrap()
            .as_any_mut()
            .downcast_mut::<ComponentStorage<T>>()
            .unwrap()
    }

    fn get_storage<T: Component>(&self) -> Option<&ComponentStorage<T>> {
        self.storages
            .get(&TypeId::of::<T>())
            .and_then(|storage| storage.as_any().downcast_ref::<ComponentStorage<T>>())
    }

    fn get_storage_mut<T: Component>(&mut self) -> Option<&mut ComponentStorage<T>> {
        self.storages
            .get_mut(&TypeId::of::<T>())
            .and_then(|storage| storage.as_any_mut().downcast_mut::<ComponentStorage<T>>())
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}
