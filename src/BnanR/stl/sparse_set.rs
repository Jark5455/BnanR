use std::fmt::Debug;
use std::ops::{Index, IndexMut};

#[derive(Clone, Debug)]
pub struct SparseSet<T> {
    sparse: Vec<usize>,
    dense: Vec<T>,
    dense_idx: Vec<usize>,
}

impl<T> Index<usize> for SparseSet<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dense[self.sparse[index]]
    }
}

impl<T> IndexMut<usize> for SparseSet<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.dense[self.sparse[index]]
    }
}

pub struct Iter<'a, T> {
    inner: std::slice::Iter<'a, T>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<T> SparseSet<T> {
    pub fn new() -> Self {
        Self {
            sparse: Vec::new(),
            dense: Vec::new(),
            dense_idx: Vec::new(),
        }
    }

    pub fn contains(&self, idx: usize) -> bool {
        self.dense_idx[self.sparse[idx]] == idx
    }

    pub fn insert(&mut self, pos: usize, elem: T) -> Option<T> {
        if pos >= self.sparse.len() {
            self.sparse.resize(pos + 1usize, usize::MAX);
        }

        if self.sparse[pos] != usize::MAX {
            let dense_index = self.sparse[pos];
            let old_value = std::mem::replace(&mut self.dense[dense_index], elem);
            return Some(old_value);
        }

        let dense_index = self.dense.len();

        self.dense.push(elem);
        self.dense_idx.push(pos);

        self.sparse[pos] = dense_index;
        None
    }

    pub fn remove(&mut self, pos: usize) -> Option<T> {
        if pos >= self.sparse.len() {
            return None;
        }

        let dense_index = self.sparse[pos];

        if dense_index == usize::MAX {
            return None;
        }

        let last_index = self.dense.len() - 1;
        let last_idx = self.dense_idx[dense_index];

        self.dense.swap(dense_index, last_index);
        self.dense_idx.swap(dense_index, last_index);

        self.sparse[last_idx] = last_index;
        self.sparse[pos] = usize::MAX;

        self.dense_idx.pop();
        self.dense.pop()
    }

    pub fn iter(&self) -> Iter<'_, T> {
        Iter { inner: self.dense.iter() }
    }
}



