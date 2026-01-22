#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Entity {
    pub id: u32,
    pub generation: u32,
}

impl Entity {
    pub fn new(id: u32, generation: u32) -> Self {
        Self { id, generation }
    }

    pub fn index(&self) -> usize {
        self.id as usize
    }
}

pub struct EntityAllocator {
    generations: Vec<u32>,
    free_ids: Vec<u32>,
    next_id: u32,
}

impl EntityAllocator {
    pub fn new() -> Self {
        Self {
            generations: Vec::new(),
            free_ids: Vec::new(),
            next_id: 0,
        }
    }

    pub fn allocate(&mut self) -> Entity {
        if let Some(id) = self.free_ids.pop() {
            let generation = self.generations[id as usize];
            Entity::new(id, generation)
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.generations.push(0);
            Entity::new(id, 0)
        }
    }

    pub fn deallocate(&mut self, entity: Entity) -> bool {
        let idx = entity.id as usize;
        if idx < self.generations.len() && self.generations[idx] == entity.generation {
            self.generations[idx] += 1;
            self.free_ids.push(entity.id);
            true
        } else {
            false
        }
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        let idx = entity.id as usize;
        idx < self.generations.len() && self.generations[idx] == entity.generation
    }
}

impl Default for EntityAllocator {
    fn default() -> Self {
        Self::new()
    }
}
