//! Typed per-asset-kind storage with generational indices.

use crate::assets::AssetId;

/// Typed storage for one kind of GPU handle (mesh, texture, material).
pub(crate) struct AssetRegistry<T: Copy> {
    /// Generational slots: (generation, Option<handle>).
    slots: Vec<(u8, Option<T>)>,
    /// Free slot indices for reuse.
    free_list: Vec<u32>,
}

impl<T: Copy> AssetRegistry<T> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
        }
    }

    /// Allocate a slot, returning an AssetId. The handle is initially None (loading).
    pub fn allocate(&mut self) -> AssetId {
        if let Some(index) = self.free_list.pop() {
            let (cur_gen, handle) = &mut self.slots[index as usize];
            *cur_gen = cur_gen.wrapping_add(1);
            *handle = None;
            AssetId::new(index, *cur_gen)
        } else {
            let index = self.slots.len() as u32;
            assert!(index < 0x00FF_FFFF, "too many assets");
            self.slots.push((0, None));
            AssetId::new(index, 0)
        }
    }

    /// Insert a pre-loaded handle, returning an AssetId.
    pub fn insert(&mut self, value: T) -> AssetId {
        let id = self.allocate();
        let slot = &mut self.slots[id.index() as usize];
        slot.1 = Some(value);
        id
    }

    /// Set the GPU handle for an allocated slot (after background loading completes).
    #[allow(dead_code)]
    pub fn set(&mut self, id: AssetId, value: T) -> bool {
        if let Some((cur_gen, handle)) = self.slots.get_mut(id.index() as usize) {
            if *cur_gen == id.generation() {
                *handle = Some(value);
                return true;
            }
        }
        false
    }

    /// Get the GPU handle, or None if still loading or freed.
    pub fn get(&self, id: AssetId) -> Option<T> {
        self.slots.get(id.index() as usize).and_then(|(cur_gen, h)| {
            if *cur_gen == id.generation() {
                *h
            } else {
                None
            }
        })
    }

    /// Free a slot for reuse.
    #[allow(dead_code)]
    pub fn remove(&mut self, id: AssetId) -> bool {
        if let Some((cur_gen, handle)) = self.slots.get_mut(id.index() as usize) {
            if *cur_gen == id.generation() {
                *handle = None;
                self.free_list.push(id.index());
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_set() {
        let mut reg = AssetRegistry::<u32>::new();
        let id = reg.allocate();
        assert!(reg.get(id).is_none());
        reg.set(id, 42);
        assert_eq!(reg.get(id), Some(42));
    }

    #[test]
    fn insert_preloaded() {
        let mut reg = AssetRegistry::<u32>::new();
        let id = reg.insert(99);
        assert_eq!(reg.get(id), Some(99));
    }

    #[test]
    fn generation_invalidates_old_handle() {
        let mut reg = AssetRegistry::<u32>::new();
        let id1 = reg.insert(1);
        reg.remove(id1);
        let id2 = reg.allocate();
        // Old handle should not resolve.
        assert!(reg.get(id1).is_none());
        // New handle should work.
        reg.set(id2, 2);
        assert_eq!(reg.get(id2), Some(2));
    }
}
