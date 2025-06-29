use intrusive_collections::intrusive_adapter;
use intrusive_collections::{LinkedList, LinkedListLink};
use intrusive_collections::linked_list::{CursorMut};

pub struct Node<T> {
    pub link: LinkedListLink,
    pub value: T,
}

impl<T> Node<T> {
    pub fn new(value: T) -> Self {
        Node {
            link: LinkedListLink::new(),
            value,
        }
    }
}

intrusive_adapter!(pub NodeAdapter<T> = Box<Node<T>>: Node<T> { link: LinkedListLink });

#[derive(Default)]
pub struct CircularList<T> {
    list: LinkedList<NodeAdapter<T>>,
}

impl<T> CircularList<T> {
    pub fn new() -> Self {
        CircularList {
            list: LinkedList::new(NodeAdapter::new()),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    pub fn push_front(&mut self, value: T) {
        self.list.push_front(Box::new(Node::new(value)));
    }

    pub fn push_back(&mut self, value: T) {
        self.list.push_back(Box::new(Node::new(value)));
    }
    
    pub fn cursor_mut(&mut self) -> CircularListCursor<T> {
        let mut cursor = self.list.cursor_mut();
        
        if cursor.is_null() {
            cursor.move_next();
        }

        CircularListCursor { cursor }
    }
}

pub struct CircularListCursor<'a, T> {
    cursor: CursorMut<'a, NodeAdapter<T>>,
}

impl<'a, T> CircularListCursor<'a, T> {
    pub fn current(&self) -> Option<&T> {
        self.cursor.get().map(|node| &node.value)
    }

    pub fn rotate_next(&mut self) {
        if self.cursor.get().is_some() {
            self.cursor.move_next();
            if self.cursor.is_null() {
                self.cursor.move_next();
            }
        }
    }

    pub fn rotate_prev(&mut self) {
        if self.cursor.get().is_some() {
            self.cursor.move_prev();
            if self.cursor.is_null() {
                self.cursor.move_prev();
            }
        }
    }
    
    pub fn remove_current(&mut self) -> Option<T> {
        if self.cursor.get().is_some() {
            let removed_node = self.cursor.remove().unwrap();
            
            if self.cursor.is_null() {
                self.cursor.move_next();
            }
            
            Some(removed_node.value)
        } else {
            None
        }
    }
    
    pub fn insert_after(&mut self, value: T) {
        self.cursor.insert_after(Box::new(Node::new(value)));
    }
    
    pub fn insert_before(&mut self, value: T) {
        self.cursor.insert_before(Box::new(Node::new(value)));
    }
}


#[cfg(test)]
mod circular_list_tests {
    use super::*;
    
    #[test]
    fn test_circular_list() {
        let mut list = CircularList::new();

        list.push_front(1);
        list.push_back(2);
        list.push_back(3); // List is now [1, 2, 3]
        
        {
            // Create a mutable cursor to interact with the list
            let mut cursor = list.cursor_mut();
            
            // The cursor starts at the head
            assert_eq!(cursor.current(), Some(&1));

            cursor.rotate_next();
            assert_eq!(cursor.current(), Some(&2));

            cursor.rotate_next();
            assert_eq!(cursor.current(), Some(&3));

            cursor.rotate_next(); // Wraps around
            assert_eq!(cursor.current(), Some(&1));

            cursor.rotate_prev(); // Wraps around the other way
            assert_eq!(cursor.current(), Some(&3));
        }
        
        print!("Current list state: ");
        
        let len = 4;
        
        let mut cursor = list.cursor_mut();
            
        for _ in 0..len {
            print!("{} ", cursor.current().unwrap());
            cursor.rotate_next();
        }
    }
}
