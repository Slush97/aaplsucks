//! Belt/conveyor logistics — transport items along directional belt segments.
//!
//! Each belt entity has a `BeltSegment` component with a direction and item slots.
//! The belt system advances items forward each tick. When the front slot reaches
//! the end, items transfer to the next belt segment's back slot.

use esox_engine::glam::{IVec2, Vec3};
use esox_engine::hecs;

use crate::inventory::ItemId;

/// Number of item slots per belt segment (1 tile = 1 entity).
/// Items move from slot 0 (back) to SLOTS-1 (front).
pub const SLOTS_PER_BELT: usize = 4;

/// Ticks per slot advance. Lower = faster belt. At 60Hz tick rate:
/// - 8 ticks/slot = 7.5 items/sec throughput
/// - 4 ticks/slot = 15 items/sec
pub const TICKS_PER_ADVANCE: u32 = 8;

/// Cardinal direction on the XZ grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dir4 {
    North, // -Z
    East,  // +X
    South, // +Z
    West,  // -X
}

impl Dir4 {
    /// Unit offset in grid coordinates (col, row) where +row = +Z.
    pub fn offset(self) -> IVec2 {
        match self {
            Dir4::North => IVec2::new(0, -1),
            Dir4::East => IVec2::new(1, 0),
            Dir4::South => IVec2::new(0, 1),
            Dir4::West => IVec2::new(-1, 0),
        }
    }

    /// Rotation angle around Y for rendering (radians).
    pub fn angle_y(self) -> f32 {
        match self {
            Dir4::North => 0.0,
            Dir4::East => -std::f32::consts::FRAC_PI_2,
            Dir4::South => std::f32::consts::PI,
            Dir4::West => std::f32::consts::FRAC_PI_2,
        }
    }

    /// Opposite direction.
    pub fn opposite(self) -> Self {
        match self {
            Dir4::North => Dir4::South,
            Dir4::East => Dir4::West,
            Dir4::South => Dir4::North,
            Dir4::West => Dir4::East,
        }
    }

    pub fn rotate_cw(self) -> Self {
        match self {
            Dir4::North => Dir4::East,
            Dir4::East => Dir4::South,
            Dir4::South => Dir4::West,
            Dir4::West => Dir4::North,
        }
    }
}

/// Grid position of a belt/building on the factory floor (column, row).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridPos(pub IVec2);

impl GridPos {
    pub fn new(col: i32, row: i32) -> Self {
        Self(IVec2::new(col, row))
    }

    /// Convert to world-space position (cell centers, Y=0).
    pub fn to_world(self, cell_size: f32) -> Vec3 {
        Vec3::new(
            self.0.x as f32 * cell_size + cell_size * 0.5,
            0.0,
            self.0.y as f32 * cell_size + cell_size * 0.5,
        )
    }

    /// Convert from world-space position to grid position.
    pub fn from_world(pos: Vec3, cell_size: f32) -> Self {
        Self(IVec2::new(
            (pos.x / cell_size).floor() as i32,
            (pos.z / cell_size).floor() as i32,
        ))
    }

    /// Get the neighbor in the given direction.
    pub fn neighbor(self, dir: Dir4) -> Self {
        Self(self.0 + dir.offset())
    }
}

/// A single belt segment occupying one grid cell.
///
/// Items travel from slot 0 (entry) toward slot SLOTS-1 (exit).
/// When the front item reaches the exit and the next belt has an empty back slot,
/// the item transfers.
pub struct BeltSegment {
    /// Direction items travel.
    pub direction: Dir4,
    /// Grid position of this belt.
    pub grid_pos: GridPos,
    /// Item slots. `None` = empty. Index 0 is the back (entry), last is the front (exit).
    pub items: [Option<ItemId>; SLOTS_PER_BELT],
    /// Tick counter for the current advance cycle.
    pub advance_tick: u32,
}

impl BeltSegment {
    pub fn new(grid_pos: GridPos, direction: Dir4) -> Self {
        Self {
            direction,
            grid_pos,
            items: [None; SLOTS_PER_BELT],
            advance_tick: 0,
        }
    }

    /// Whether the back (entry) slot is free to accept an item.
    pub fn can_accept(&self) -> bool {
        self.items[0].is_none()
    }

    /// Push an item into the back slot. Returns false if occupied.
    pub fn push_back(&mut self, item: ItemId) -> bool {
        if self.items[0].is_none() {
            self.items[0] = Some(item);
            true
        } else {
            false
        }
    }

    /// Take the item from the front (exit) slot.
    pub fn take_front(&mut self) -> Option<ItemId> {
        self.items[SLOTS_PER_BELT - 1].take()
    }

    /// Peek at the front slot without removing.
    pub fn peek_front(&self) -> Option<ItemId> {
        self.items[SLOTS_PER_BELT - 1]
    }
}

/// Run the belt advancement system for one tick.
///
/// Two phases:
/// 1. **Transfer**: For each belt with a front item, try to hand it off to the
///    next belt (the neighbor in this belt's direction).
/// 2. **Advance**: Shift items forward within each belt (back toward front).
pub fn belt_tick_system(world: &mut hecs::World) {
    // Collect belt data to avoid borrow conflicts.
    let belt_data: Vec<(hecs::Entity, GridPos, Dir4, [Option<ItemId>; SLOTS_PER_BELT], u32)> = world
        .query_mut::<&BeltSegment>()
        .into_iter()
        .map(|(e, b)| (e, b.grid_pos, b.direction, b.items, b.advance_tick))
        .collect();

    // Build a lookup: grid_pos -> entity for finding neighbors.
    let mut grid_lookup: std::collections::HashMap<GridPos, hecs::Entity> =
        std::collections::HashMap::with_capacity(belt_data.len());
    for &(entity, grid_pos, _, _, _) in &belt_data {
        grid_lookup.insert(grid_pos, entity);
    }

    // Phase 1: Transfer front items to neighbor belts.
    // Collect transfers as (source_entity, dest_entity) pairs.
    let mut transfers: Vec<(hecs::Entity, hecs::Entity)> = Vec::new();
    for &(entity, grid_pos, direction, items, _) in &belt_data {
        if items[SLOTS_PER_BELT - 1].is_some() {
            let neighbor_pos = grid_pos.neighbor(direction);
            if let Some(&neighbor_entity) = grid_lookup.get(&neighbor_pos) {
                // Check that the neighbor can accept (back slot empty).
                // Find neighbor in belt_data.
                if let Some(&(_, _, _, neighbor_items, _)) = belt_data
                    .iter()
                    .find(|(e, _, _, _, _)| *e == neighbor_entity)
                {
                    if neighbor_items[0].is_none() {
                        transfers.push((entity, neighbor_entity));
                    }
                }
            }
        }
    }

    // Execute transfers.
    for (src, dst) in &transfers {
        // Take from source front.
        let item = {
            let mut src_belt = world.get::<&mut BeltSegment>(*src).unwrap();
            src_belt.items[SLOTS_PER_BELT - 1].take()
        };
        if let Some(item) = item {
            let mut dst_belt = world.get::<&mut BeltSegment>(*dst).unwrap();
            dst_belt.items[0] = Some(item);
        }
    }

    // Phase 2: Advance items within each belt.
    for (entity, _, _, _, _) in &belt_data {
        let mut belt = world.get::<&mut BeltSegment>(*entity).unwrap();
        belt.advance_tick += 1;
        if belt.advance_tick >= TICKS_PER_ADVANCE {
            belt.advance_tick = 0;
            // Shift items forward (from back to front), only into empty slots.
            for i in (1..SLOTS_PER_BELT).rev() {
                if belt.items[i].is_none() && belt.items[i - 1].is_some() {
                    belt.items[i] = belt.items[i - 1].take();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn belt_advances_item() {
        let mut world = hecs::World::new();
        let belt_entity = world.spawn((BeltSegment::new(GridPos::new(0, 0), Dir4::East),));

        // Put an item in the back slot.
        world.get::<&mut BeltSegment>(belt_entity).unwrap().items[0] = Some(0);

        // Advance enough ticks.
        for _ in 0..TICKS_PER_ADVANCE {
            belt_tick_system(&mut world);
        }

        let belt = world.get::<&BeltSegment>(belt_entity).unwrap();
        // Item should have moved forward one slot.
        assert!(belt.items[0].is_none());
        assert_eq!(belt.items[1], Some(0));
    }

    #[test]
    fn belt_transfers_to_neighbor() {
        let mut world = hecs::World::new();

        // Two belts: [0,0] -> East, [1,0] -> East
        world.spawn((BeltSegment::new(GridPos::new(0, 0), Dir4::East),));
        world.spawn((BeltSegment::new(GridPos::new(1, 0), Dir4::East),));

        // Put an item at the front of belt 0.
        let belts: Vec<hecs::Entity> = world
            .query_mut::<&BeltSegment>()
            .into_iter()
            .filter(|(_, b)| b.grid_pos == GridPos::new(0, 0))
            .map(|(e, _)| e)
            .collect();
        world.get::<&mut BeltSegment>(belts[0]).unwrap().items[SLOTS_PER_BELT - 1] = Some(0);

        // One tick should transfer.
        belt_tick_system(&mut world);

        // Check: belt 0 front should be empty, belt 1 back should have item.
        for (_, belt) in world.query_mut::<&BeltSegment>() {
            if belt.grid_pos == GridPos::new(0, 0) {
                assert!(belt.items[SLOTS_PER_BELT - 1].is_none(), "source front should be empty");
            }
            if belt.grid_pos == GridPos::new(1, 0) {
                assert_eq!(belt.items[0], Some(0), "dest back should have item");
            }
        }
    }

    #[test]
    fn belt_blocked_no_transfer() {
        let mut world = hecs::World::new();

        world.spawn((BeltSegment::new(GridPos::new(0, 0), Dir4::East),));
        world.spawn((BeltSegment::new(GridPos::new(1, 0), Dir4::East),));

        // Fill both belt fronts.
        for (_, belt) in world.query_mut::<&mut BeltSegment>() {
            if belt.grid_pos == GridPos::new(0, 0) {
                belt.items[SLOTS_PER_BELT - 1] = Some(0);
            }
            if belt.grid_pos == GridPos::new(1, 0) {
                belt.items[0] = Some(1); // blocked
            }
        }

        belt_tick_system(&mut world);

        // Source should still have its item.
        for (_, belt) in world.query_mut::<&BeltSegment>() {
            if belt.grid_pos == GridPos::new(0, 0) {
                assert_eq!(belt.items[SLOTS_PER_BELT - 1], Some(0));
            }
        }
    }

    #[test]
    fn dir4_offset_roundtrip() {
        let pos = GridPos::new(5, 5);
        for dir in [Dir4::North, Dir4::East, Dir4::South, Dir4::West] {
            let neighbor = pos.neighbor(dir);
            let back = neighbor.neighbor(dir.opposite());
            assert_eq!(back, pos);
        }
    }
}
