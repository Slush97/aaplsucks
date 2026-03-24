//! Inventory system — items, stacks, and slot-based inventories.

use std::collections::HashMap;

use serde::Deserialize;

/// Unique item identifier (interned string key).
pub type ItemId = u16;

/// Definition of an item type loaded from data.
#[derive(Debug, Clone, Deserialize)]
pub struct ItemDef {
    pub id: String,
    pub name: String,
    pub stack_size: u32,
}

/// Registry of all item definitions, mapping string ids to numeric ItemIds.
pub struct ItemRegistry {
    defs: Vec<ItemDef>,
    name_to_id: HashMap<String, ItemId>,
}

impl ItemRegistry {
    /// Load item definitions from a RON file.
    pub fn load_from_ron(data: &str) -> Self {
        let defs: Vec<ItemDef> = ron::from_str(data).expect("failed to parse items.ron");
        let mut name_to_id = HashMap::new();
        for (i, def) in defs.iter().enumerate() {
            name_to_id.insert(def.id.clone(), i as ItemId);
        }
        Self { defs, name_to_id }
    }

    /// Look up numeric id from string id.
    pub fn id_of(&self, name: &str) -> Option<ItemId> {
        self.name_to_id.get(name).copied()
    }

    /// Get the item definition for a numeric id.
    pub fn get(&self, id: ItemId) -> &ItemDef {
        &self.defs[id as usize]
    }

    /// Maximum stack size for an item.
    pub fn stack_size(&self, id: ItemId) -> u32 {
        self.defs[id as usize].stack_size
    }

    /// Display name for an item.
    pub fn name(&self, id: ItemId) -> &str {
        &self.defs[id as usize].name
    }

    /// Number of registered items.
    pub fn count(&self) -> usize {
        self.defs.len()
    }
}

/// A stack of items in a single slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ItemStack {
    pub item: ItemId,
    pub count: u32,
}

impl ItemStack {
    pub fn new(item: ItemId, count: u32) -> Self {
        Self { item, count }
    }
}

/// Slot-based inventory (hecs component).
pub struct Inventory {
    pub slots: Vec<Option<ItemStack>>,
}

impl Inventory {
    /// Create an empty inventory with the given number of slots.
    pub fn new(num_slots: usize) -> Self {
        Self {
            slots: vec![None; num_slots],
        }
    }

    /// Try to insert an item stack, merging into existing stacks first.
    /// Returns the leftover count that couldn't be inserted (0 = full success).
    pub fn insert(&mut self, item: ItemId, mut count: u32, registry: &ItemRegistry) -> u32 {
        let max_stack = registry.stack_size(item);

        // First pass: merge into existing stacks of the same item.
        for slot in &mut self.slots {
            if count == 0 {
                break;
            }
            if let Some(stack) = slot {
                if stack.item == item && stack.count < max_stack {
                    let space = max_stack - stack.count;
                    let add = count.min(space);
                    stack.count += add;
                    count -= add;
                }
            }
        }

        // Second pass: fill empty slots.
        for slot in &mut self.slots {
            if count == 0 {
                break;
            }
            if slot.is_none() {
                let add = count.min(max_stack);
                *slot = Some(ItemStack::new(item, add));
                count -= add;
            }
        }

        count
    }

    /// Try to remove `count` of `item`. Returns how many were actually removed.
    pub fn remove(&mut self, item: ItemId, mut count: u32) -> u32 {
        let mut removed = 0u32;
        for slot in &mut self.slots {
            if count == 0 {
                break;
            }
            if let Some(stack) = slot {
                if stack.item == item {
                    let take = count.min(stack.count);
                    stack.count -= take;
                    count -= take;
                    removed += take;
                    if stack.count == 0 {
                        *slot = None;
                    }
                }
            }
        }
        removed
    }

    /// Count total of a specific item across all slots.
    pub fn count_item(&self, item: ItemId) -> u32 {
        self.slots
            .iter()
            .filter_map(|s| s.as_ref())
            .filter(|s| s.item == item)
            .map(|s| s.count)
            .sum()
    }

    /// Check if the inventory has at least `count` of `item`.
    pub fn has(&self, item: ItemId, count: u32) -> bool {
        self.count_item(item) >= count
    }

    /// Whether the inventory is completely empty.
    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(|s| s.is_none())
    }

    /// Whether the inventory has no free slots and all stacks are full.
    pub fn is_full(&self, registry: &ItemRegistry) -> bool {
        self.slots.iter().all(|s| match s {
            None => false,
            Some(stack) => stack.count >= registry.stack_size(stack.item),
        })
    }

    /// Get the first non-empty slot's item id (for inserters to decide what to grab).
    pub fn first_item(&self) -> Option<ItemId> {
        self.slots
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|s| s.item)
            .next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_registry() -> ItemRegistry {
        ItemRegistry::load_from_ron(
            r#"[
                (id: "iron-ore", name: "Iron Ore", stack_size: 50),
                (id: "iron-plate", name: "Iron Plate", stack_size: 100),
            ]"#,
        )
    }

    #[test]
    fn insert_and_count() {
        let reg = test_registry();
        let mut inv = Inventory::new(4);
        let iron = reg.id_of("iron-ore").unwrap();
        let leftover = inv.insert(iron, 30, &reg);
        assert_eq!(leftover, 0);
        assert_eq!(inv.count_item(iron), 30);
    }

    #[test]
    fn insert_merges_stacks() {
        let reg = test_registry();
        let mut inv = Inventory::new(1);
        let iron = reg.id_of("iron-ore").unwrap();
        inv.insert(iron, 20, &reg);
        inv.insert(iron, 20, &reg);
        assert_eq!(inv.count_item(iron), 40);
        // Should be one stack of 40
        assert_eq!(inv.slots[0].unwrap().count, 40);
    }

    #[test]
    fn insert_overflow() {
        let reg = test_registry();
        let mut inv = Inventory::new(1);
        let iron = reg.id_of("iron-ore").unwrap();
        let leftover = inv.insert(iron, 60, &reg);
        assert_eq!(leftover, 10); // stack_size is 50
        assert_eq!(inv.count_item(iron), 50);
    }

    #[test]
    fn remove_items() {
        let reg = test_registry();
        let mut inv = Inventory::new(4);
        let iron = reg.id_of("iron-ore").unwrap();
        inv.insert(iron, 30, &reg);
        let removed = inv.remove(iron, 10);
        assert_eq!(removed, 10);
        assert_eq!(inv.count_item(iron), 20);
    }

    #[test]
    fn remove_clears_empty_slots() {
        let reg = test_registry();
        let mut inv = Inventory::new(4);
        let iron = reg.id_of("iron-ore").unwrap();
        inv.insert(iron, 10, &reg);
        inv.remove(iron, 10);
        assert!(inv.is_empty());
    }

    #[test]
    fn has_check() {
        let reg = test_registry();
        let mut inv = Inventory::new(4);
        let iron = reg.id_of("iron-ore").unwrap();
        inv.insert(iron, 30, &reg);
        assert!(inv.has(iron, 30));
        assert!(!inv.has(iron, 31));
    }
}
