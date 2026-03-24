//! Resource nodes and mining — ore patches and miner buildings.

use esox_engine::hecs;

use crate::belt::GridPos;
use crate::inventory::{ItemId, ItemRegistry};
use crate::power::PowerConsumer;
use crate::recipe::OutputInventory;

/// An ore patch on the world grid.
pub struct ResourceNode {
    /// What item this node yields.
    pub item: ItemId,
    /// Remaining units before depletion.
    pub remaining: u32,
    /// Grid position.
    pub grid_pos: GridPos,
}

impl ResourceNode {
    pub fn new(item: ItemId, amount: u32, grid_pos: GridPos) -> Self {
        Self {
            item,
            remaining: amount,
            grid_pos,
        }
    }

    pub fn is_depleted(&self) -> bool {
        self.remaining == 0
    }
}

/// A mining drill that extracts items from a resource node.
pub struct Miner {
    /// Grid position of the miner.
    pub grid_pos: GridPos,
    /// Ticks per extraction cycle.
    pub cycle_ticks: u32,
    /// Current progress in ticks.
    pub progress: u32,
    /// Whether the miner is actively extracting.
    pub active: bool,
}

/// Default mining speed: one ore every 2 seconds at 60Hz.
pub const MINER_CYCLE_TICKS: u32 = 120;

impl Miner {
    pub fn new(grid_pos: GridPos) -> Self {
        Self {
            grid_pos,
            cycle_ticks: MINER_CYCLE_TICKS,
            progress: 0,
            active: false,
        }
    }
}

/// Run the mining system for one tick.
///
/// Each miner looks for a resource node at its grid position.
/// When the extraction cycle completes, the miner's output inventory gains one item.
pub fn mining_tick_system(world: &mut hecs::World, items: &ItemRegistry) {
    // Collect resource nodes for lookup.
    let nodes: Vec<(hecs::Entity, GridPos, ItemId, u32)> = world
        .query_mut::<&ResourceNode>()
        .into_iter()
        .map(|(e, n)| (e, n.grid_pos, n.item, n.remaining))
        .collect();

    // Build grid_pos -> (entity, item, remaining) lookup.
    let mut node_lookup: std::collections::HashMap<GridPos, (hecs::Entity, ItemId, u32)> =
        std::collections::HashMap::new();
    for (entity, pos, item, remaining) in &nodes {
        if *remaining > 0 {
            node_lookup.insert(*pos, (*entity, *item, *remaining));
        }
    }

    // Collect miners.
    let miners: Vec<(hecs::Entity, GridPos, u32, u32)> = world
        .query_mut::<&Miner>()
        .into_iter()
        .map(|(e, m)| (e, m.grid_pos, m.progress, m.cycle_ticks))
        .collect();

    for (miner_entity, grid_pos, progress, cycle_ticks) in miners {
        let Some(&(node_entity, item, _remaining)) = node_lookup.get(&grid_pos) else {
            // No resource node here.
            if let Ok(mut miner) = world.get::<&mut Miner>(miner_entity) {
                miner.active = false;
                miner.progress = 0;
            }
            continue;
        };

        // Check power. Entities without a PowerConsumer are always powered (e.g. in tests).
        let powered = world
            .get::<&PowerConsumer>(miner_entity)
            .map(|c| c.is_powered())
            .unwrap_or(true);
        if !powered {
            if let Ok(mut miner) = world.get::<&mut Miner>(miner_entity) {
                miner.active = false;
            }
            continue;
        }

        // Check if output inventory is full.
        let output_full = world
            .get::<&OutputInventory>(miner_entity)
            .map(|o| o.0.is_full(items))
            .unwrap_or(true);

        if output_full {
            if let Ok(mut miner) = world.get::<&mut Miner>(miner_entity) {
                miner.active = false;
            }
            continue;
        }

        let new_progress = progress + 1;
        if new_progress >= cycle_ticks {
            // Extract one unit.
            if let Ok(mut node) = world.get::<&mut ResourceNode>(node_entity) {
                if node.remaining > 0 {
                    node.remaining -= 1;
                }
            }
            // Add to miner's output.
            if let Ok(mut output) = world.get::<&mut OutputInventory>(miner_entity) {
                output.0.insert(item, 1, items);
            }
            let mut miner = world.get::<&mut Miner>(miner_entity).unwrap();
            miner.progress = 0;
            miner.active = true;
        } else {
            let mut miner = world.get::<&mut Miner>(miner_entity).unwrap();
            miner.progress = new_progress;
            miner.active = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::inventory::Inventory;

    fn test_registry() -> ItemRegistry {
        ItemRegistry::load_from_ron(
            r#"[(id: "iron-ore", name: "Iron Ore", stack_size: 50)]"#,
        )
    }

    #[test]
    fn miner_extracts_from_node() {
        let items = test_registry();
        let ore = items.id_of("iron-ore").unwrap();

        let mut world = hecs::World::new();

        // Resource node at (0, 0) with 100 ore.
        world.spawn((ResourceNode::new(ore, 100, GridPos::new(0, 0)),));

        // Miner at (0, 0).
        let mut miner = Miner::new(GridPos::new(0, 0));
        miner.cycle_ticks = 5; // fast for testing
        let miner_entity = world.spawn((miner, OutputInventory(Inventory::new(4))));

        // Run 5 ticks to complete one cycle.
        for _ in 0..5 {
            mining_tick_system(&mut world, &items);
        }

        {
            let output = world.get::<&OutputInventory>(miner_entity).unwrap();
            assert_eq!(output.0.count_item(ore), 1);
        }

        // Resource should have decremented.
        for (_, node) in world.query_mut::<&ResourceNode>() {
            assert_eq!(node.remaining, 99);
        }
    }

    #[test]
    fn miner_stops_when_depleted() {
        let items = test_registry();
        let ore = items.id_of("iron-ore").unwrap();

        let mut world = hecs::World::new();
        world.spawn((ResourceNode::new(ore, 1, GridPos::new(0, 0)),));

        let mut miner = Miner::new(GridPos::new(0, 0));
        miner.cycle_ticks = 3;
        let miner_entity = world.spawn((miner, OutputInventory(Inventory::new(4))));

        // Extract the one unit.
        for _ in 0..3 {
            mining_tick_system(&mut world, &items);
        }

        {
            let output = world.get::<&OutputInventory>(miner_entity).unwrap();
            assert_eq!(output.0.count_item(ore), 1);
        }

        // Run more ticks — should not extract more.
        for _ in 0..10 {
            mining_tick_system(&mut world, &items);
        }

        {
            let output = world.get::<&OutputInventory>(miner_entity).unwrap();
            assert_eq!(output.0.count_item(ore), 1);
        }
    }
}
