//! Inserter system — robotic arms that transfer items between inventories.
//!
//! Each inserter is an entity with an `Inserter` component that specifies
//! a pickup grid position and a dropoff grid position. Each tick, the inserter
//! tries to grab an item from the pickup source and place it at the dropoff.

use esox_engine::hecs;

use crate::belt::{BeltSegment, GridPos};
use crate::inventory::{Inventory, ItemId, ItemRegistry};

/// How many ticks an inserter takes to complete one pickup-and-drop cycle.
pub const INSERTER_CYCLE_TICKS: u32 = 30; // 0.5 seconds at 60Hz

/// State of the inserter's arm during its cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InserterState {
    /// Waiting at the pickup position, ready to grab.
    Idle,
    /// Swinging from pickup to dropoff, carrying an item.
    Delivering,
    /// Swinging back from dropoff to pickup (empty-handed).
    Returning,
}

/// Inserter component — transfers items between adjacent grid positions.
pub struct Inserter {
    /// Grid position this inserter picks up from.
    pub pickup_pos: GridPos,
    /// Grid position this inserter drops off to.
    pub dropoff_pos: GridPos,
    /// Current arm state.
    pub state: InserterState,
    /// Currently held item (while delivering).
    pub held_item: Option<ItemId>,
    /// Progress through the current swing (0..cycle_ticks).
    pub progress: u32,
    /// Ticks per pickup-to-dropoff swing.
    pub cycle_ticks: u32,
    /// Optional item filter. If `Some`, only picks up this item type.
    pub filter: Option<ItemId>,
}

impl Inserter {
    pub fn new(pickup_pos: GridPos, dropoff_pos: GridPos) -> Self {
        Self {
            pickup_pos,
            dropoff_pos,
            state: InserterState::Idle,
            held_item: None,
            progress: 0,
            cycle_ticks: INSERTER_CYCLE_TICKS,
            filter: None,
        }
    }

    /// Normalized animation progress [0.0, 1.0] for rendering the arm swing.
    pub fn animation_t(&self) -> f32 {
        if self.cycle_ticks == 0 {
            return 0.0;
        }
        self.progress as f32 / self.cycle_ticks as f32
    }
}

/// Source or destination for an inserter: either an inventory entity or a belt entity.
enum InsertTarget {
    Inventory(hecs::Entity),
    Belt(hecs::Entity),
}

/// Find the entity at a grid position that can be a source/target.
fn find_target_at(world: &hecs::World, pos: GridPos) -> Option<InsertTarget> {
    // Check belts first.
    for (entity, belt) in world.query::<&BeltSegment>().iter() {
        if belt.grid_pos == pos {
            return Some(InsertTarget::Belt(entity));
        }
    }
    // Check entities with Inventory + GridPos tag.
    for (entity, (_, grid)) in world.query::<(&Inventory, &GridPos)>().iter() {
        if *grid == pos {
            return Some(InsertTarget::Inventory(entity));
        }
    }
    None
}

/// Try to grab an item from a source.
fn try_pickup(
    world: &mut hecs::World,
    target: &InsertTarget,
    filter: Option<ItemId>,
) -> Option<ItemId> {
    match target {
        InsertTarget::Belt(entity) => {
            let mut belt = world.get::<&mut BeltSegment>(*entity).ok()?;
            // Take from the front slot (exit of belt).
            let item = belt.peek_front()?;
            if let Some(f) = filter {
                if item != f {
                    return None;
                }
            }
            belt.take_front()
        }
        InsertTarget::Inventory(entity) => {
            let mut inv = world.get::<&mut Inventory>(*entity).ok()?;
            // Pick first available item (respecting filter).
            let item = if let Some(f) = filter {
                if inv.has(f, 1) { Some(f) } else { None }
            } else {
                inv.first_item()
            }?;
            let removed = inv.remove(item, 1);
            if removed > 0 {
                Some(item)
            } else {
                None
            }
        }
    }
}

/// Try to drop an item into a target.
fn try_dropoff(
    world: &mut hecs::World,
    target: &InsertTarget,
    item: ItemId,
    registry: &ItemRegistry,
) -> bool {
    match target {
        InsertTarget::Belt(entity) => {
            if let Ok(mut belt) = world.get::<&mut BeltSegment>(*entity) {
                belt.push_back(item)
            } else {
                false
            }
        }
        InsertTarget::Inventory(entity) => {
            if let Ok(mut inv) = world.get::<&mut Inventory>(*entity) {
                inv.insert(item, 1, registry) == 0
            } else {
                false
            }
        }
    }
}

/// Run the inserter system for one tick.
pub fn inserter_tick_system(world: &mut hecs::World, registry: &ItemRegistry) {
    // Collect inserter state to avoid borrow conflicts.
    let inserters: Vec<(hecs::Entity, GridPos, GridPos, InserterState, Option<ItemId>, u32, u32, Option<ItemId>)> =
        world
            .query_mut::<&Inserter>()
            .into_iter()
            .map(|(e, ins)| {
                (e, ins.pickup_pos, ins.dropoff_pos, ins.state, ins.held_item, ins.progress, ins.cycle_ticks, ins.filter)
            })
            .collect();

    for (entity, pickup_pos, dropoff_pos, state, held_item, progress, cycle_ticks, filter) in inserters {
        match state {
            InserterState::Idle => {
                // Try to pick up an item.
                let target = find_target_at(world, pickup_pos);
                if let Some(target) = target {
                    if let Some(item) = try_pickup(world, &target, filter) {
                        let mut ins = world.get::<&mut Inserter>(entity).unwrap();
                        ins.held_item = Some(item);
                        ins.state = InserterState::Delivering;
                        ins.progress = 0;
                    }
                }
            }
            InserterState::Delivering => {
                let new_progress = progress + 1;
                if new_progress >= cycle_ticks {
                    // Try to drop off.
                    let target = find_target_at(world, dropoff_pos);
                    let dropped = if let Some(target) = target {
                        if let Some(item) = held_item {
                            try_dropoff(world, &target, item, registry)
                        } else {
                            true // nothing to drop
                        }
                    } else {
                        false
                    };

                    let mut ins = world.get::<&mut Inserter>(entity).unwrap();
                    if dropped {
                        ins.held_item = None;
                        ins.state = InserterState::Returning;
                        ins.progress = 0;
                    }
                    // If not dropped, stay at end of deliver (blocked).
                } else {
                    let mut ins = world.get::<&mut Inserter>(entity).unwrap();
                    ins.progress = new_progress;
                }
            }
            InserterState::Returning => {
                let new_progress = progress + 1;
                let mut ins = world.get::<&mut Inserter>(entity).unwrap();
                if new_progress >= cycle_ticks {
                    ins.state = InserterState::Idle;
                    ins.progress = 0;
                } else {
                    ins.progress = new_progress;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belt::{Dir4, SLOTS_PER_BELT};

    fn test_registry() -> ItemRegistry {
        ItemRegistry::load_from_ron(
            r#"[(id: "ore", name: "Ore", stack_size: 50)]"#,
        )
    }

    #[test]
    fn inserter_picks_from_belt_drops_to_inventory() {
        let reg = test_registry();
        let ore = reg.id_of("ore").unwrap();
        let mut world = hecs::World::new();

        // Source belt at (0,0) with an item at front.
        let mut belt = BeltSegment::new(GridPos::new(0, 0), Dir4::East);
        belt.items[SLOTS_PER_BELT - 1] = Some(ore);
        world.spawn((belt,));

        // Destination chest at (0,1) with inventory.
        world.spawn((Inventory::new(4), GridPos::new(0, 1)));

        // Inserter from (0,0) -> (0,1).
        let ins_entity = world.spawn((Inserter::new(GridPos::new(0, 0), GridPos::new(0, 1)),));

        // First tick: should pick up.
        inserter_tick_system(&mut world, &reg);
        {
            let ins = world.get::<&Inserter>(ins_entity).unwrap();
            assert_eq!(ins.state, InserterState::Delivering);
            assert_eq!(ins.held_item, Some(ore));
        }

        // Run through delivery ticks.
        for _ in 0..INSERTER_CYCLE_TICKS {
            inserter_tick_system(&mut world, &reg);
        }

        // Should have dropped off.
        {
            let ins = world.get::<&Inserter>(ins_entity).unwrap();
            assert_eq!(ins.state, InserterState::Returning);
            assert!(ins.held_item.is_none());
        }

        // Check inventory received the item.
        for (_, inv) in world.query_mut::<&Inventory>() {
            assert_eq!(inv.count_item(ore), 1);
        }
    }

    #[test]
    fn inserter_respects_filter() {
        let reg = test_registry();
        let ore = reg.id_of("ore").unwrap();
        let mut world = hecs::World::new();

        // Source chest with ore.
        world.spawn((Inventory::new(4), GridPos::new(0, 0)));
        // Insert an item into it.
        for (_, inv) in world.query_mut::<&mut Inventory>() {
            inv.insert(ore, 5, &reg);
        }

        // Destination.
        world.spawn((Inventory::new(4), GridPos::new(1, 0)));

        // Inserter with filter set to something that doesn't exist (item 99).
        let mut ins = Inserter::new(GridPos::new(0, 0), GridPos::new(1, 0));
        ins.filter = Some(99);
        let ins_entity = world.spawn((ins,));

        inserter_tick_system(&mut world, &reg);

        // Should stay idle — filter doesn't match.
        let ins = world.get::<&Inserter>(ins_entity).unwrap();
        assert_eq!(ins.state, InserterState::Idle);
    }
}
