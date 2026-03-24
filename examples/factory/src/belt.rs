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

/// Maximum tunnel distance for underground belts (in tiles, exclusive of entry/exit).
pub const MAX_UNDERGROUND_DISTANCE: i32 = 5;

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

/// Whether an underground belt is an entry (items go in) or exit (items come out).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UndergroundBeltMode {
    Entry,
    Exit,
}

/// An underground belt that teleports items between an Entry and a paired Exit.
pub struct UndergroundBelt {
    /// Direction items travel (Entry→Exit).
    pub direction: Dir4,
    /// Entry or Exit.
    pub mode: UndergroundBeltMode,
    /// Grid position.
    pub grid_pos: GridPos,
    /// Paired underground belt entity (Entry↔Exit).
    pub pair: Option<hecs::Entity>,
    /// Single item buffer.
    pub held_item: Option<ItemId>,
}

impl UndergroundBelt {
    pub fn new(grid_pos: GridPos, direction: Dir4, mode: UndergroundBeltMode) -> Self {
        Self {
            direction,
            mode,
            grid_pos,
            pair: None,
            held_item: None,
        }
    }
}

/// Run the belt advancement system for one tick.
///
/// Phases:
/// 1. **Underground teleport**: Entry with item + paired Exit empty → teleport.
/// 2. **Transfer**: For each belt with a front item, try to hand it off to the
///    next belt or an Entry underground belt (the neighbor in this belt's direction).
/// 3. **Exit→Belt**: Exit buffers feed into downstream belt back slots.
/// 4. **Advance**: Shift items forward within each belt (back toward front).
pub fn belt_tick_system(world: &mut hecs::World) {
    // ── Phase 0: Underground teleport (Entry → paired Exit) ──
    underground_belt_tick_system(world);

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

    // Collect underground belt data for cross-system transfers.
    let ub_data: Vec<(hecs::Entity, GridPos, Dir4, UndergroundBeltMode, Option<ItemId>)> = world
        .query_mut::<&UndergroundBelt>()
        .into_iter()
        .map(|(e, ub)| (e, ub.grid_pos, ub.direction, ub.mode, ub.held_item))
        .collect();
    let mut ub_grid_lookup: std::collections::HashMap<GridPos, usize> =
        std::collections::HashMap::with_capacity(ub_data.len());
    for (i, &(_, grid_pos, _, _, _)) in ub_data.iter().enumerate() {
        ub_grid_lookup.insert(grid_pos, i);
    }

    // Phase 1: Transfer front items to neighbor belts OR Entry underground belts.
    let mut belt_transfers: Vec<(hecs::Entity, hecs::Entity)> = Vec::new();
    let mut belt_to_ub_transfers: Vec<(hecs::Entity, hecs::Entity)> = Vec::new();
    for &(entity, grid_pos, direction, items, _) in &belt_data {
        if items[SLOTS_PER_BELT - 1].is_some() {
            let neighbor_pos = grid_pos.neighbor(direction);
            // Try neighbor belt first.
            if let Some(&neighbor_entity) = grid_lookup.get(&neighbor_pos) {
                if let Some(&(_, _, _, neighbor_items, _)) = belt_data
                    .iter()
                    .find(|(e, _, _, _, _)| *e == neighbor_entity)
                {
                    if neighbor_items[0].is_none() {
                        belt_transfers.push((entity, neighbor_entity));
                        continue;
                    }
                }
            }
            // No belt neighbor — check for Entry underground belt at neighbor_pos facing same direction.
            if let Some(&ub_idx) = ub_grid_lookup.get(&neighbor_pos) {
                let (ub_entity, _, ub_dir, ub_mode, ub_held) = ub_data[ub_idx];
                if ub_dir == direction && ub_mode == UndergroundBeltMode::Entry && ub_held.is_none() {
                    belt_to_ub_transfers.push((entity, ub_entity));
                }
            }
        }
    }

    // Execute belt→belt transfers.
    for (src, dst) in &belt_transfers {
        let item = {
            let mut src_belt = world.get::<&mut BeltSegment>(*src).unwrap();
            src_belt.items[SLOTS_PER_BELT - 1].take()
        };
        if let Some(item) = item {
            let mut dst_belt = world.get::<&mut BeltSegment>(*dst).unwrap();
            dst_belt.items[0] = Some(item);
        }
    }

    // Execute belt→Entry underground belt transfers.
    for (src, dst) in &belt_to_ub_transfers {
        let item = {
            let mut src_belt = world.get::<&mut BeltSegment>(*src).unwrap();
            src_belt.items[SLOTS_PER_BELT - 1].take()
        };
        if let Some(item) = item {
            let mut ub = world.get::<&mut UndergroundBelt>(*dst).unwrap();
            ub.held_item = Some(item);
        }
    }

    // Phase 2: Exit underground belt → downstream belt back slot.
    let mut ub_to_belt_transfers: Vec<(hecs::Entity, hecs::Entity)> = Vec::new();
    // Re-read underground belt data after teleport phase may have changed buffers.
    let ub_data_fresh: Vec<(hecs::Entity, GridPos, Dir4, UndergroundBeltMode, Option<ItemId>)> = world
        .query_mut::<&UndergroundBelt>()
        .into_iter()
        .map(|(e, ub)| (e, ub.grid_pos, ub.direction, ub.mode, ub.held_item))
        .collect();
    for &(ub_entity, ub_pos, ub_dir, ub_mode, ub_held) in &ub_data_fresh {
        if ub_mode == UndergroundBeltMode::Exit && ub_held.is_some() {
            let downstream_pos = ub_pos.neighbor(ub_dir);
            if let Some(&belt_entity) = grid_lookup.get(&downstream_pos) {
                if let Some(&(_, _, _, _, _)) = belt_data
                    .iter()
                    .find(|(e, _, _, _, _)| *e == belt_entity)
                {
                    // Check current state (belt back slot may have been filled by belt→belt transfer).
                    let back_free = world.get::<&BeltSegment>(belt_entity)
                        .map(|b| b.items[0].is_none())
                        .unwrap_or(false);
                    if back_free {
                        ub_to_belt_transfers.push((ub_entity, belt_entity));
                    }
                }
            }
        }
    }

    // Execute Exit→belt transfers.
    for (src, dst) in &ub_to_belt_transfers {
        let item = {
            let mut ub = world.get::<&mut UndergroundBelt>(*src).unwrap();
            ub.held_item.take()
        };
        if let Some(item) = item {
            let mut belt = world.get::<&mut BeltSegment>(*dst).unwrap();
            belt.items[0] = Some(item);
        }
    }

    // Phase 3: Advance items within each belt.
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

/// Teleport items from Entry underground belts to their paired Exit.
fn underground_belt_tick_system(world: &mut hecs::World) {
    // Collect Entry belts with items and a pair.
    let entries: Vec<(hecs::Entity, hecs::Entity)> = world
        .query_mut::<&UndergroundBelt>()
        .into_iter()
        .filter_map(|(e, ub)| {
            if ub.mode == UndergroundBeltMode::Entry && ub.held_item.is_some() {
                ub.pair.map(|pair| (e, pair))
            } else {
                None
            }
        })
        .collect();

    for (entry_entity, exit_entity) in entries {
        // Check exit has empty buffer.
        let exit_empty = world
            .get::<&UndergroundBelt>(exit_entity)
            .map(|ub| ub.held_item.is_none())
            .unwrap_or(false);
        if exit_empty {
            let item = {
                let mut entry = world.get::<&mut UndergroundBelt>(entry_entity).unwrap();
                entry.held_item.take()
            };
            if let Some(item) = item {
                let mut exit = world.get::<&mut UndergroundBelt>(exit_entity).unwrap();
                exit.held_item = Some(item);
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

    // ── Underground belt tests ──

    #[test]
    fn underground_entry_exit_teleport() {
        let mut world = hecs::World::new();

        let entry_e = world.spawn((
            UndergroundBelt::new(GridPos::new(0, 0), Dir4::East, UndergroundBeltMode::Entry),
        ));
        let exit_e = world.spawn((
            UndergroundBelt::new(GridPos::new(3, 0), Dir4::East, UndergroundBeltMode::Exit),
        ));

        // Pair them.
        world.get::<&mut UndergroundBelt>(entry_e).unwrap().pair = Some(exit_e);
        world.get::<&mut UndergroundBelt>(exit_e).unwrap().pair = Some(entry_e);

        // Put an item in the entry buffer.
        world.get::<&mut UndergroundBelt>(entry_e).unwrap().held_item = Some(0);

        // One tick should teleport to exit.
        belt_tick_system(&mut world);

        {
            let entry = world.get::<&UndergroundBelt>(entry_e).unwrap();
            assert!(entry.held_item.is_none(), "entry should be empty after teleport");
        }
        {
            let exit = world.get::<&UndergroundBelt>(exit_e).unwrap();
            assert_eq!(exit.held_item, Some(0), "exit should have the item");
        }
    }

    #[test]
    fn underground_pairing_within_range() {
        // Test helper: simulate pairing logic (same as main.rs will do).
        fn try_pair(world: &mut hecs::World, new_pos: GridPos, dir: Dir4, mode: UndergroundBeltMode) -> hecs::Entity {
            let entity = world.spawn((UndergroundBelt::new(new_pos, dir, mode),));

            // If placing an Exit, scan backward for unpaired Entry.
            // If placing an Entry, scan forward for unpaired Exit.
            let (scan_dir, target_mode) = match mode {
                UndergroundBeltMode::Exit => (dir.opposite(), UndergroundBeltMode::Entry),
                UndergroundBeltMode::Entry => (dir, UndergroundBeltMode::Exit),
            };

            let mut scan_pos = new_pos;
            let mut found_pair = None;
            for _ in 0..MAX_UNDERGROUND_DISTANCE {
                scan_pos = scan_pos.neighbor(scan_dir);
                // Find underground belt at scan_pos.
                for (e, ub) in world.query::<&UndergroundBelt>().iter() {
                    if e != entity && ub.grid_pos == scan_pos && ub.direction == dir
                        && ub.mode == target_mode && ub.pair.is_none()
                    {
                        found_pair = Some(e);
                        break;
                    }
                }
                if found_pair.is_some() {
                    break;
                }
            }

            if let Some(pair_entity) = found_pair {
                world.get::<&mut UndergroundBelt>(entity).unwrap().pair = Some(pair_entity);
                world.get::<&mut UndergroundBelt>(pair_entity).unwrap().pair = Some(entity);
            }

            entity
        }

        let mut world = hecs::World::new();

        // Place entry at (0,0) facing East.
        let entry = try_pair(&mut world, GridPos::new(0, 0), Dir4::East, UndergroundBeltMode::Entry);
        // Place exit at (3,0) facing East — within range (distance=3).
        let exit = try_pair(&mut world, GridPos::new(3, 0), Dir4::East, UndergroundBeltMode::Exit);

        assert_eq!(world.get::<&UndergroundBelt>(entry).unwrap().pair, Some(exit));
        assert_eq!(world.get::<&UndergroundBelt>(exit).unwrap().pair, Some(entry));
    }

    #[test]
    fn underground_max_distance_exceeded() {
        // Same helper as above.
        fn try_pair(world: &mut hecs::World, new_pos: GridPos, dir: Dir4, mode: UndergroundBeltMode) -> hecs::Entity {
            let entity = world.spawn((UndergroundBelt::new(new_pos, dir, mode),));

            let (scan_dir, target_mode) = match mode {
                UndergroundBeltMode::Exit => (dir.opposite(), UndergroundBeltMode::Entry),
                UndergroundBeltMode::Entry => (dir, UndergroundBeltMode::Exit),
            };

            let mut scan_pos = new_pos;
            let mut found_pair = None;
            for _ in 0..MAX_UNDERGROUND_DISTANCE {
                scan_pos = scan_pos.neighbor(scan_dir);
                for (e, ub) in world.query::<&UndergroundBelt>().iter() {
                    if e != entity && ub.grid_pos == scan_pos && ub.direction == dir
                        && ub.mode == target_mode && ub.pair.is_none()
                    {
                        found_pair = Some(e);
                        break;
                    }
                }
                if found_pair.is_some() {
                    break;
                }
            }

            if let Some(pair_entity) = found_pair {
                world.get::<&mut UndergroundBelt>(entity).unwrap().pair = Some(pair_entity);
                world.get::<&mut UndergroundBelt>(pair_entity).unwrap().pair = Some(entity);
            }

            entity
        }

        let mut world = hecs::World::new();

        let entry = try_pair(&mut world, GridPos::new(0, 0), Dir4::East, UndergroundBeltMode::Entry);
        // Place exit at (6,0) — distance 6, exceeds MAX_UNDERGROUND_DISTANCE (5).
        let exit = try_pair(&mut world, GridPos::new(6, 0), Dir4::East, UndergroundBeltMode::Exit);

        assert!(world.get::<&UndergroundBelt>(entry).unwrap().pair.is_none(), "entry should not pair");
        assert!(world.get::<&UndergroundBelt>(exit).unwrap().pair.is_none(), "exit should not pair");
    }

    #[test]
    fn belt_to_entry_transfer() {
        let mut world = hecs::World::new();

        // Belt at (0,0) facing East, front slot has item.
        let mut belt = BeltSegment::new(GridPos::new(0, 0), Dir4::East);
        belt.items[SLOTS_PER_BELT - 1] = Some(0);
        world.spawn((belt,));

        // Entry underground belt at (1,0) facing East (neighbor in belt's direction).
        let entry_e = world.spawn((
            UndergroundBelt::new(GridPos::new(1, 0), Dir4::East, UndergroundBeltMode::Entry),
        ));

        belt_tick_system(&mut world);

        // Belt front should be empty, entry should have item.
        for (_, b) in world.query_mut::<&BeltSegment>() {
            assert!(b.items[SLOTS_PER_BELT - 1].is_none(), "belt front should be empty");
        }
        let entry = world.get::<&UndergroundBelt>(entry_e).unwrap();
        assert_eq!(entry.held_item, Some(0), "entry should have the item");
    }

    #[test]
    fn exit_to_belt_transfer() {
        let mut world = hecs::World::new();

        // Exit underground belt at (0,0) facing East with item.
        let exit_e = world.spawn((
            UndergroundBelt::new(GridPos::new(0, 0), Dir4::East, UndergroundBeltMode::Exit),
        ));
        world.get::<&mut UndergroundBelt>(exit_e).unwrap().held_item = Some(0);

        // Belt at (1,0) facing East (downstream of exit).
        world.spawn((BeltSegment::new(GridPos::new(1, 0), Dir4::East),));

        belt_tick_system(&mut world);

        // Exit buffer should be empty, belt back slot should have item.
        {
            let exit = world.get::<&UndergroundBelt>(exit_e).unwrap();
            assert!(exit.held_item.is_none(), "exit should be empty");
        }
        for (_, b) in world.query_mut::<&BeltSegment>() {
            assert_eq!(b.items[0], Some(0), "belt back should have item");
        }
    }

    #[test]
    fn unpaired_entry_holds_item() {
        let mut world = hecs::World::new();

        // Entry with no pair.
        let entry_e = world.spawn((
            UndergroundBelt::new(GridPos::new(0, 0), Dir4::East, UndergroundBeltMode::Entry),
        ));
        world.get::<&mut UndergroundBelt>(entry_e).unwrap().held_item = Some(0);

        // Several ticks — item should stay.
        for _ in 0..10 {
            belt_tick_system(&mut world);
        }

        let entry = world.get::<&UndergroundBelt>(entry_e).unwrap();
        assert_eq!(entry.held_item, Some(0), "unpaired entry should keep its item");
    }
}
